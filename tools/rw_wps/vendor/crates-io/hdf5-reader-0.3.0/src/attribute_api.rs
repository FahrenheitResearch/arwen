use crate::error::{Error, Result};
use crate::fractal_heap::FractalHeap;
use crate::global_heap::GlobalHeapCollection;
use crate::io::Cursor;
use crate::messages::attribute::AttributeMessage;
use crate::messages::attribute_info::AttributeInfoMessage;
use crate::messages::dataspace::DataspaceType;
use crate::messages::datatype::{Datatype, StringEncoding, StringPadding, StringSize};
use crate::messages::HdfMessage;
use crate::object_header::ObjectHeader;
use crate::storage::Storage;
use crate::{btree_v2, messages};

/// A parsed, high-level HDF5 attribute.
#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub datatype: Datatype,
    pub shape: Vec<u64>,
    pub raw_data: Vec<u8>,
}

impl Attribute {
    /// Create from a parsed attribute message.
    pub fn from_message(msg: AttributeMessage) -> Self {
        Self::from_message_with_context(msg, None, 0)
    }

    /// Create from a parsed attribute message with optional file context for
    /// resolving variable-length byte attributes stored in the global heap.
    pub fn from_message_with_context(
        msg: AttributeMessage,
        file_data: Option<&[u8]>,
        offset_size: u8,
    ) -> Self {
        let shape = match msg.dataspace.dataspace_type {
            DataspaceType::Scalar => vec![],
            DataspaceType::Null => vec![0],
            DataspaceType::Simple => msg.dataspace.dims.clone(),
        };
        let raw_data =
            if let (Some(file_data), Datatype::VarLen { base }) = (file_data, &msg.datatype) {
                if is_byte_vlen(base) && shape.is_empty() {
                    resolve_vlen_bytes(&msg.raw_data, file_data, offset_size)
                        .unwrap_or_else(|| msg.raw_data.clone())
                } else {
                    msg.raw_data.clone()
                }
            } else {
                msg.raw_data.clone()
            };
        Attribute {
            name: msg.name,
            datatype: msg.datatype,
            shape,
            raw_data,
        }
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> u64 {
        if self.shape.is_empty() {
            1 // scalar
        } else {
            self.shape.iter().product()
        }
    }

    /// Read the attribute value as a scalar of the given type.
    pub fn read_scalar<T: crate::datatype_api::H5Type>(&self) -> Result<T> {
        T::from_bytes(&self.raw_data, &self.datatype)
    }

    /// Read the attribute as a 1-D vector of the given type.
    pub fn read_1d<T: crate::datatype_api::H5Type>(&self) -> Result<Vec<T>> {
        let elem_size = T::element_size(&self.datatype);
        let n = self.num_elements() as usize;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * elem_size;
            let end = start + elem_size;
            if end > self.raw_data.len() {
                return Err(Error::InvalidData(format!(
                    "attribute data too short: need {} bytes, have {}",
                    end,
                    self.raw_data.len()
                )));
            }
            result.push(T::from_bytes(&self.raw_data[start..end], &self.datatype)?);
        }
        Ok(result)
    }

    /// Read the attribute as a string (for string-typed attributes).
    ///
    /// For variable-length strings, use `read_vlen_string()` with the file data
    /// and offset_size — this method will return an error directing you there.
    pub fn read_string(&self) -> Result<String> {
        match &self.datatype {
            Datatype::VarLen { base } if is_byte_vlen(base) => {
                decode_varlen_byte_string(&self.raw_data)
            }
            Datatype::String {
                size,
                encoding,
                padding,
            } => match size {
                StringSize::Fixed(len) => {
                    let len = *len as usize;
                    let bytes = if self.raw_data.len() < len {
                        &self.raw_data
                    } else {
                        &self.raw_data[..len]
                    };
                    decode_string(bytes, *padding, *encoding)
                }
                StringSize::Variable => {
                    // For inline vlen strings in attributes, try direct decode.
                    // If it looks like a global heap reference (>= 12 bytes for
                    // seq_len + addr + index), suggest read_vlen_string instead.
                    if self.raw_data.len() >= 12 {
                        // Try to decode directly first — some files inline the string
                        let trimmed = match padding {
                            StringPadding::NullTerminate => {
                                let end = self
                                    .raw_data
                                    .iter()
                                    .position(|&b| b == 0)
                                    .unwrap_or(self.raw_data.len());
                                &self.raw_data[..end]
                            }
                            _ => &self.raw_data,
                        };
                        if let Ok(s) = String::from_utf8(trimmed.to_vec()) {
                            if s.chars()
                                .all(|c| !c.is_control() || c == '\n' || c == '\r' || c == '\t')
                            {
                                return Ok(s);
                            }
                        }
                    }
                    decode_string(&self.raw_data, *padding, *encoding)
                }
            },
            _ => Err(Error::TypeMismatch {
                expected: "String".into(),
                actual: format!("{:?}", self.datatype),
            }),
        }
    }

    /// Read a variable-length string attribute from the global heap.
    ///
    /// Variable-length strings in HDF5 are stored as references into a global
    /// heap collection. Each reference is: `seq_len(u32) + heap_addr(offset_size) + index(u32)`.
    pub fn read_vlen_string(&self, file_data: &[u8], offset_size: u8) -> Result<String> {
        match &self.datatype {
            Datatype::String {
                size: StringSize::Variable,
                encoding,
                padding,
            } => {
                let ref_size = 4 + offset_size as usize + 4; // seq_len + addr + index
                if self.raw_data.len() < ref_size {
                    // Fallback: try direct decode
                    return decode_string(&self.raw_data, *padding, *encoding);
                }
                let bytes = read_one_vlen_string(
                    &self.raw_data,
                    0,
                    file_data,
                    offset_size,
                    *padding,
                    *encoding,
                )?;
                Ok(bytes)
            }
            Datatype::String {
                size: StringSize::Fixed(_),
                ..
            } => self.read_string(),
            _ => Err(Error::TypeMismatch {
                expected: "String".into(),
                actual: format!("{:?}", self.datatype),
            }),
        }
    }

    /// Read an array of variable-length strings from the global heap.
    pub fn read_vlen_strings(&self, file_data: &[u8], offset_size: u8) -> Result<Vec<String>> {
        match &self.datatype {
            Datatype::String {
                size: StringSize::Variable,
                encoding,
                padding,
            } => {
                let ref_size = 4 + offset_size as usize + 4;
                let n = self.num_elements() as usize;
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let offset = i * ref_size;
                    if offset + ref_size > self.raw_data.len() {
                        break;
                    }
                    result.push(read_one_vlen_string(
                        &self.raw_data,
                        offset,
                        file_data,
                        offset_size,
                        *padding,
                        *encoding,
                    )?);
                }
                Ok(result)
            }
            Datatype::String {
                size: StringSize::Fixed(_),
                ..
            } => self.read_strings(),
            _ => Err(Error::TypeMismatch {
                expected: "String array".into(),
                actual: format!("{:?}", self.datatype),
            }),
        }
    }

    /// Read the attribute as a vector of strings.
    pub fn read_strings(&self) -> Result<Vec<String>> {
        match &self.datatype {
            Datatype::String {
                size: StringSize::Fixed(len),
                encoding,
                padding,
            } => {
                let len = *len as usize;
                let n = self.num_elements() as usize;
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let start = i * len;
                    let end = (start + len).min(self.raw_data.len());
                    if start >= self.raw_data.len() {
                        break;
                    }
                    result.push(decode_string(
                        &self.raw_data[start..end],
                        *padding,
                        *encoding,
                    )?);
                }
                Ok(result)
            }
            _ => Err(Error::TypeMismatch {
                expected: "String array".into(),
                actual: format!("{:?}", self.datatype),
            }),
        }
    }

    /// Read an attribute as f64 (with automatic promotion from int types).
    pub fn read_as_f64(&self) -> Result<f64> {
        match &self.datatype {
            Datatype::FloatingPoint { size, .. } => {
                let val: f64 = match size {
                    4 => {
                        let v = self.read_scalar::<f32>()?;
                        v as f64
                    }
                    8 => self.read_scalar::<f64>()?,
                    _ => {
                        return Err(Error::TypeMismatch {
                            expected: "f32 or f64".into(),
                            actual: format!("FloatingPoint(size={})", size),
                        })
                    }
                };
                Ok(val)
            }
            Datatype::FixedPoint { size, signed, .. } => {
                let val = match (size, signed) {
                    (1, true) => self.read_scalar::<i8>()? as f64,
                    (1, false) => self.read_scalar::<u8>()? as f64,
                    (2, true) => self.read_scalar::<i16>()? as f64,
                    (2, false) => self.read_scalar::<u16>()? as f64,
                    (4, true) => self.read_scalar::<i32>()? as f64,
                    (4, false) => self.read_scalar::<u32>()? as f64,
                    (8, true) => self.read_scalar::<i64>()? as f64,
                    (8, false) => self.read_scalar::<u64>()? as f64,
                    _ => {
                        return Err(Error::TypeMismatch {
                            expected: "numeric".into(),
                            actual: format!("FixedPoint(size={})", size),
                        })
                    }
                };
                Ok(val)
            }
            _ => Err(Error::TypeMismatch {
                expected: "numeric".into(),
                actual: format!("{:?}", self.datatype),
            }),
        }
    }
}

#[allow(dead_code)]
pub(crate) fn collect_attribute_messages(
    header: &ObjectHeader,
    file_data: &[u8],
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<AttributeMessage>> {
    let mut attributes = Vec::new();
    let mut attribute_info = None;

    for msg in &header.messages {
        match msg {
            HdfMessage::Attribute(attr) => attributes.push(attr.clone()),
            HdfMessage::AttributeInfo(info) => attribute_info = Some(info.clone()),
            _ => {}
        }
    }

    if let Some(info) = attribute_info {
        attributes.extend(load_dense_attribute_messages(
            &info,
            file_data,
            offset_size,
            length_size,
        )?);
    }

    Ok(attributes)
}

pub(crate) fn collect_attribute_messages_storage(
    header: &ObjectHeader,
    storage: &dyn Storage,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<AttributeMessage>> {
    let mut attributes = Vec::new();
    let mut attribute_info = None;

    for msg in &header.messages {
        match msg {
            HdfMessage::Attribute(attr) => attributes.push(attr.clone()),
            HdfMessage::AttributeInfo(info) => attribute_info = Some(info.clone()),
            _ => {}
        }
    }

    if let Some(info) = attribute_info {
        attributes.extend(load_dense_attribute_messages_storage(
            &info,
            storage,
            offset_size,
            length_size,
        )?);
    }

    Ok(attributes)
}

#[allow(dead_code)]
fn load_dense_attribute_messages(
    info: &AttributeInfoMessage,
    file_data: &[u8],
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<AttributeMessage>> {
    if Cursor::is_undefined_offset(info.fractal_heap_address, offset_size) {
        return Ok(Vec::new());
    }

    let mut heap_cursor = Cursor::new(file_data);
    heap_cursor.set_position(info.fractal_heap_address);
    let heap = FractalHeap::parse(&mut heap_cursor, offset_size, length_size)?;

    let records =
        load_dense_attribute_records(info, file_data, offset_size, length_size).unwrap_or_default();

    let mut attributes = Vec::new();
    for record in records {
        let heap_id = match record {
            btree_v2::BTreeV2Record::AttributeNameHash { heap_id, .. }
            | btree_v2::BTreeV2Record::AttributeCreationOrder { heap_id, .. } => heap_id,
            _ => continue,
        };

        let managed_bytes =
            match heap.get_managed_object(&heap_id, file_data, offset_size, length_size) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };

        let mut attr_cursor = Cursor::new(&managed_bytes);
        if let Ok(attr) = messages::attribute::parse(
            &mut attr_cursor,
            offset_size,
            length_size,
            managed_bytes.len(),
        ) {
            attributes.push(attr);
        }
    }

    Ok(attributes)
}

fn load_dense_attribute_messages_storage(
    info: &AttributeInfoMessage,
    storage: &dyn Storage,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<AttributeMessage>> {
    if Cursor::is_undefined_offset(info.fractal_heap_address, offset_size) {
        return Ok(Vec::new());
    }

    let heap = FractalHeap::parse_at_storage(
        storage,
        info.fractal_heap_address,
        offset_size,
        length_size,
    )?;

    let records = load_dense_attribute_records_storage(info, storage, offset_size, length_size)
        .unwrap_or_default();

    let mut attributes = Vec::new();
    for record in records {
        let heap_id = match record {
            btree_v2::BTreeV2Record::AttributeNameHash { heap_id, .. }
            | btree_v2::BTreeV2Record::AttributeCreationOrder { heap_id, .. } => heap_id,
            _ => continue,
        };

        let managed_bytes =
            match heap.get_managed_object_storage(&heap_id, storage, offset_size, length_size) {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };

        let mut attr_cursor = Cursor::new(&managed_bytes);
        if let Ok(attr) = messages::attribute::parse(
            &mut attr_cursor,
            offset_size,
            length_size,
            managed_bytes.len(),
        ) {
            attributes.push(attr);
        }
    }

    Ok(attributes)
}

#[allow(dead_code)]
fn load_dense_attribute_records(
    info: &AttributeInfoMessage,
    file_data: &[u8],
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<btree_v2::BTreeV2Record>> {
    let mut addrs = vec![info.btree_name_index_address];
    if let Some(creation_order_addr) = info.btree_creation_order_address {
        addrs.push(creation_order_addr);
    }

    for addr in addrs {
        if Cursor::is_undefined_offset(addr, offset_size) {
            continue;
        }

        let mut btree_cursor = Cursor::new(file_data);
        btree_cursor.set_position(addr);
        let header =
            match btree_v2::BTreeV2Header::parse(&mut btree_cursor, offset_size, length_size) {
                Ok(header) => header,
                Err(_) => continue,
            };

        if let Ok(records) = btree_v2::collect_btree_v2_records(
            file_data,
            &header,
            offset_size,
            length_size,
            None,
            &[],
            None,
        ) {
            return Ok(records);
        }
    }

    Ok(Vec::new())
}

fn load_dense_attribute_records_storage(
    info: &AttributeInfoMessage,
    storage: &dyn Storage,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<btree_v2::BTreeV2Record>> {
    let mut addrs = vec![info.btree_name_index_address];
    if let Some(creation_order_addr) = info.btree_creation_order_address {
        addrs.push(creation_order_addr);
    }

    for addr in addrs {
        if Cursor::is_undefined_offset(addr, offset_size) {
            continue;
        }

        let header = match btree_v2::BTreeV2Header::parse_at_storage(
            storage,
            addr,
            offset_size,
            length_size,
        ) {
            Ok(header) => header,
            Err(_) => continue,
        };

        if let Ok(records) = btree_v2::collect_btree_v2_records_storage(
            storage,
            &header,
            offset_size,
            length_size,
            None,
            &[],
            None,
        ) {
            return Ok(records);
        }
    }

    Ok(Vec::new())
}

/// Read one variable-length string from a vlen reference in raw_data.
pub(crate) fn read_one_vlen_string(
    raw_data: &[u8],
    offset: usize,
    file_data: &[u8],
    offset_size: u8,
    padding: StringPadding,
    encoding: StringEncoding,
) -> Result<String> {
    let mut cursor = Cursor::new(&raw_data[offset..]);
    let _seq_len = cursor.read_u32_le()?;
    let heap_addr = cursor.read_offset(offset_size)?;
    let obj_index = cursor.read_u32_le()?;

    if Cursor::is_undefined_offset(heap_addr, offset_size) || obj_index == 0 {
        return Ok(String::new());
    }

    let mut heap_cursor = Cursor::new(file_data);
    heap_cursor.set_position(heap_addr);
    let collection = GlobalHeapCollection::parse(&mut heap_cursor, offset_size, offset_size)?;

    match collection.get_object(obj_index as u16) {
        Some(obj) => decode_string(&obj.data, padding, encoding),
        None => Ok(String::new()),
    }
}

pub(crate) fn read_one_vlen_string_storage(
    raw_data: &[u8],
    offset: usize,
    storage: &dyn Storage,
    offset_size: u8,
    length_size: u8,
    padding: StringPadding,
    encoding: StringEncoding,
) -> Result<String> {
    let mut cursor = Cursor::new(&raw_data[offset..]);
    let _seq_len = cursor.read_u32_le()?;
    let heap_addr = cursor.read_offset(offset_size)?;
    let obj_index = cursor.read_u32_le()?;

    if Cursor::is_undefined_offset(heap_addr, offset_size) || obj_index == 0 {
        return Ok(String::new());
    }

    let collection =
        GlobalHeapCollection::parse_at_storage(storage, heap_addr, offset_size, length_size)?;
    match collection.get_object(obj_index as u16) {
        Some(obj) => decode_string(&obj.data, padding, encoding),
        None => Ok(String::new()),
    }
}

/// Decode a byte slice into a String, handling padding and encoding.
///
/// HDF5 supports ASCII and UTF-8 string encodings. Both are valid UTF-8
/// (ASCII is a strict subset), so we decode uniformly via `from_utf8`.
pub(crate) fn decode_string(
    bytes: &[u8],
    padding: StringPadding,
    _encoding: StringEncoding,
) -> Result<String> {
    let trimmed = match padding {
        StringPadding::NullTerminate => {
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            &bytes[..end]
        }
        StringPadding::NullPad => {
            let end = bytes.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
            &bytes[..end]
        }
        StringPadding::SpacePad => {
            let end = bytes.iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);
            &bytes[..end]
        }
    };

    String::from_utf8(trimmed.to_vec())
        .map_err(|e| Error::InvalidData(format!("invalid string data: {e}")))
}

fn is_byte_vlen(base: &Datatype) -> bool {
    matches!(base, Datatype::FixedPoint { size: 1, .. })
}

pub(crate) fn decode_varlen_byte_string(bytes: &[u8]) -> Result<String> {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8(bytes[..end].to_vec())
        .map_err(|e| Error::InvalidData(format!("invalid string data: {e}")))
}

pub(crate) fn resolve_vlen_bytes(
    raw_data: &[u8],
    file_data: &[u8],
    offset_size: u8,
) -> Option<Vec<u8>> {
    if raw_data.len() < 4 + offset_size as usize + 4 {
        return None;
    }

    let mut cursor = Cursor::new(raw_data);
    let seq_len = cursor.read_u32_le().ok()? as usize;
    let heap_addr = cursor.read_offset(offset_size).ok()?;
    let obj_index = cursor.read_u32_le().ok()? as u16;

    if Cursor::is_undefined_offset(heap_addr, offset_size) || obj_index == 0 {
        return Some(Vec::new());
    }

    let mut heap_cursor = Cursor::new(file_data);
    heap_cursor.set_position(heap_addr);
    let collection =
        GlobalHeapCollection::parse(&mut heap_cursor, offset_size, offset_size).ok()?;
    let object = collection.get_object(obj_index)?;
    Some(object.data[..object.data.len().min(seq_len)].to_vec())
}

pub(crate) fn resolve_vlen_bytes_storage(
    raw_data: &[u8],
    storage: &dyn Storage,
    offset_size: u8,
    length_size: u8,
) -> Option<Vec<u8>> {
    if raw_data.len() < 4 + offset_size as usize + 4 {
        return None;
    }

    let mut cursor = Cursor::new(raw_data);
    let seq_len = cursor.read_u32_le().ok()? as usize;
    let heap_addr = cursor.read_offset(offset_size).ok()?;
    let obj_index = cursor.read_u32_le().ok()? as u16;

    if Cursor::is_undefined_offset(heap_addr, offset_size) || obj_index == 0 {
        return Some(Vec::new());
    }

    let collection =
        GlobalHeapCollection::parse_at_storage(storage, heap_addr, offset_size, length_size)
            .ok()?;
    let object = collection.get_object(obj_index)?;
    Some(object.data[..object.data.len().min(seq_len)].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ByteOrder;
    use std::f64::consts::PI;

    #[test]
    fn test_scalar_f64_attribute() {
        let value: f64 = PI;
        let raw_data = value.to_le_bytes().to_vec();
        let attr = Attribute {
            name: "pi".to_string(),
            datatype: Datatype::FloatingPoint {
                size: 8,
                byte_order: ByteOrder::LittleEndian,
            },
            shape: vec![],
            raw_data,
        };
        let val = attr.read_scalar::<f64>().unwrap();
        assert!((val - PI).abs() < 1e-10);
    }

    #[test]
    fn test_1d_i32_attribute() {
        let values = [1i32, 2, 3, 4];
        let mut raw_data = Vec::new();
        for v in &values {
            raw_data.extend_from_slice(&v.to_le_bytes());
        }
        let attr = Attribute {
            name: "data".to_string(),
            datatype: Datatype::FixedPoint {
                size: 4,
                signed: true,
                byte_order: ByteOrder::LittleEndian,
            },
            shape: vec![4],
            raw_data,
        };
        let result = attr.read_1d::<i32>().unwrap();
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_string_attribute() {
        let attr = Attribute {
            name: "units".to_string(),
            datatype: Datatype::String {
                size: StringSize::Fixed(10),
                encoding: StringEncoding::Ascii,
                padding: StringPadding::NullPad,
            },
            shape: vec![],
            raw_data: b"meters\0\0\0\0".to_vec(),
        };
        assert_eq!(attr.read_string().unwrap(), "meters");
    }

    #[test]
    fn test_varlen_byte_string_attribute() {
        let attr = Attribute {
            name: "name".to_string(),
            datatype: Datatype::VarLen {
                base: Box::new(Datatype::FixedPoint {
                    size: 1,
                    signed: false,
                    byte_order: ByteOrder::LittleEndian,
                }),
            },
            shape: vec![],
            raw_data: b"test_dataset".to_vec(),
        };
        assert_eq!(attr.read_string().unwrap(), "test_dataset");
    }

    #[test]
    fn test_read_as_f64_from_int() {
        let raw_data = 42i32.to_le_bytes().to_vec();
        let attr = Attribute {
            name: "count".to_string(),
            datatype: Datatype::FixedPoint {
                size: 4,
                signed: true,
                byte_order: ByteOrder::LittleEndian,
            },
            shape: vec![],
            raw_data,
        };
        let val = attr.read_as_f64().unwrap();
        assert!((val - 42.0).abs() < 1e-10);
    }
}
