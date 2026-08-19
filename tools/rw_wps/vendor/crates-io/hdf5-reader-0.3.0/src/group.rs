use std::sync::Arc;

use crate::attribute_api::{
    collect_attribute_messages_storage, resolve_vlen_bytes_storage, Attribute,
};
use crate::btree_v1;
use crate::btree_v2;
use crate::dataset::Dataset;
use crate::error::{Error, Result};
use crate::fractal_heap::FractalHeap;
use crate::io::Cursor;
use crate::local_heap::LocalHeap;
use crate::messages::link::{self, LinkMessage, LinkTarget};
use crate::messages::link_info::LinkInfoMessage;
use crate::messages::symbol_table_msg::SymbolTableMessage;
use crate::messages::HdfMessage;
use crate::storage::Storage;
use crate::FileContext;

/// A group within an HDF5 file.
#[derive(Clone)]
pub struct Group {
    context: Arc<FileContext>,
    pub(crate) name: String,
    pub(crate) address: u64,
    /// Address of the root group's object header, used for resolving soft links.
    pub(crate) root_address: u64,
}

#[derive(Debug, Clone)]
struct ChildEntry {
    name: String,
    address: u64,
}

impl Group {
    /// Create a group from a known object header address.
    pub(crate) fn new(
        context: Arc<FileContext>,
        address: u64,
        name: String,
        root_address: u64,
    ) -> Self {
        Group {
            context,
            name,
            address,
            root_address,
        }
    }

    /// Group name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Object header address of this group within the file.
    pub fn address(&self) -> u64 {
        self.address
    }

    /// Materialize the full file backing this group.
    pub fn file_data(&self) -> Result<crate::storage::StorageBuffer> {
        self.context.full_file_data()
    }

    /// Access the underlying random-access storage backend.
    pub fn storage(&self) -> &dyn Storage {
        self.context.storage.as_ref()
    }

    /// Size of file offsets in bytes.
    pub fn offset_size(&self) -> u8 {
        self.context.superblock.offset_size
    }

    /// Size of file lengths in bytes.
    pub fn length_size(&self) -> u8 {
        self.context.superblock.length_size
    }

    /// Parse (or retrieve from cache) the object header at the given address.
    fn cached_header(&self, addr: u64) -> Result<Arc<crate::object_header::ObjectHeader>> {
        self.context.get_or_parse_header(addr)
    }

    /// List all child groups.
    pub fn groups(&self) -> Result<Vec<Group>> {
        let (groups, _) = self.resolve_member_objects()?;
        Ok(groups)
    }

    /// List all child members, partitioned into groups and datasets.
    pub fn members(&self) -> Result<(Vec<Group>, Vec<Dataset>)> {
        self.resolve_member_objects()
    }

    fn resolve_member_objects(&self) -> Result<(Vec<Group>, Vec<Dataset>)> {
        let children = self.resolve_children()?;
        let mut groups = Vec::new();
        let mut datasets = Vec::new();
        for child in &children {
            if self.child_is_group(child)? {
                groups.push(Group::new(
                    self.context.clone(),
                    child.address,
                    child.name.clone(),
                    self.root_address,
                ));
            } else if let Some(dataset) = self.try_open_child_dataset(child) {
                datasets.push(dataset);
            }
        }
        Ok((groups, datasets))
    }

    /// Get a child group by name.
    pub fn group(&self, name: &str) -> Result<Group> {
        let children = self.resolve_children()?;
        for child in &children {
            if child.name == name {
                if self.is_group_at(child.address)? {
                    return Ok(Group::new(
                        self.context.clone(),
                        child.address,
                        child.name.clone(),
                        self.root_address,
                    ));
                } else {
                    return Err(Error::GroupNotFound(format!(
                        "'{}' is a dataset, not a group",
                        name
                    )));
                }
            }
        }
        Err(Error::GroupNotFound(name.to_string()))
    }

    /// List all child datasets.
    pub fn datasets(&self) -> Result<Vec<Dataset>> {
        let (_, datasets) = self.resolve_member_objects()?;
        Ok(datasets)
    }

    /// Get a child dataset by name.
    pub fn dataset(&self, name: &str) -> Result<Dataset> {
        let children = self.resolve_children()?;
        for child in &children {
            if child.name == name {
                if let Some(dataset) = self.try_open_child_dataset(child) {
                    return Ok(dataset);
                }
                return Err(Error::DatasetNotFound(name.to_string()));
            }
        }
        Err(Error::DatasetNotFound(name.to_string()))
    }

    /// List attributes on this group.
    pub fn attributes(&self) -> Result<Vec<Attribute>> {
        let mut header = (*self.cached_header(self.address)?).clone();
        header.resolve_shared_messages_storage(
            self.context.storage.as_ref(),
            self.offset_size(),
            self.length_size(),
        )?;
        Ok(collect_attribute_messages_storage(
            &header,
            self.context.storage.as_ref(),
            self.offset_size(),
            self.length_size(),
        )?
        .into_iter()
        .map(|attr| {
            let raw_data = match &attr.datatype {
                crate::messages::datatype::Datatype::VarLen { base }
                    if matches!(
                        base.as_ref(),
                        crate::messages::datatype::Datatype::FixedPoint { size: 1, .. }
                    ) && attr.dataspace.num_elements() == 1 =>
                {
                    resolve_vlen_bytes_storage(
                        &attr.raw_data,
                        self.context.storage.as_ref(),
                        self.offset_size(),
                        self.length_size(),
                    )
                    .unwrap_or_else(|| attr.raw_data.clone())
                }
                _ => attr.raw_data.clone(),
            };
            Attribute {
                name: attr.name,
                datatype: attr.datatype,
                shape: match attr.dataspace.dataspace_type {
                    crate::messages::dataspace::DataspaceType::Scalar => vec![],
                    crate::messages::dataspace::DataspaceType::Null => vec![0],
                    crate::messages::dataspace::DataspaceType::Simple => attr.dataspace.dims,
                },
                raw_data,
            }
        })
        .collect())
    }

    /// Find an attribute by name.
    pub fn attribute(&self, name: &str) -> Result<Attribute> {
        let attrs = self.attributes()?;
        attrs
            .into_iter()
            .find(|a| a.name == name)
            .ok_or_else(|| Error::AttributeNotFound(name.to_string()))
    }

    /// Resolve children from the object header.
    /// Handles both old-style (symbol table) and new-style (link messages) groups.
    fn resolve_children(&self) -> Result<Vec<ChildEntry>> {
        self.resolve_children_with_link_depth(0)
    }

    /// Resolve children with a soft-link depth counter to prevent cycles.
    fn resolve_children_with_link_depth(&self, link_depth: u32) -> Result<Vec<ChildEntry>> {
        let header = self.cached_header(self.address)?;

        let mut children = Vec::new();

        // Check for old-style groups (symbol table message)
        let mut found_symbol_table = false;
        // Check for new-style groups (link messages)
        let mut link_info: Option<LinkInfoMessage> = None;
        let mut links: Vec<LinkMessage> = Vec::new();

        for msg in &header.messages {
            match msg {
                HdfMessage::SymbolTable(st) => {
                    found_symbol_table = true;
                    children = self.resolve_old_style_group_storage(st)?;
                }
                HdfMessage::Link(link) => {
                    links.push(link.clone());
                }
                HdfMessage::LinkInfo(li) => {
                    link_info = Some(li.clone());
                }
                _ => {}
            }
        }

        if !found_symbol_table {
            // New-style group: use compact links from header messages
            self.resolve_link_targets(&links, link_depth, &mut children);

            // Dense-link storage can coexist with compact links, so merge both.
            if let Some(ref li) = link_info {
                if !Cursor::is_undefined_offset(li.fractal_heap_address, self.offset_size()) {
                    for child in self.resolve_dense_links_storage(li, link_depth)? {
                        let is_duplicate = children.iter().any(|existing| {
                            existing.name == child.name && existing.address == child.address
                        });
                        if !is_duplicate {
                            children.push(child);
                        }
                    }
                }
            }
        }

        Ok(children)
    }

    /// Resolve link targets (hard and soft), appending to `children`.
    fn resolve_link_targets(
        &self,
        links: &[LinkMessage],
        link_depth: u32,
        children: &mut Vec<ChildEntry>,
    ) {
        for link in links {
            match &link.target {
                LinkTarget::Hard { address } => {
                    children.push(ChildEntry {
                        name: link.name.clone(),
                        address: *address,
                    });
                }
                LinkTarget::Soft { path } => {
                    if let Ok(address) = self.resolve_soft_link_depth(path, link_depth) {
                        children.push(ChildEntry {
                            name: link.name.clone(),
                            address,
                        });
                    }
                }
                LinkTarget::External { .. } => {
                    // External links reference other files; skip.
                }
            }
        }
    }

    /// Resolve old-style group children via B-tree v1 + local heap.
    #[allow(dead_code)]
    fn resolve_old_style_group(
        &self,
        st: &SymbolTableMessage,
        file_data: &[u8],
    ) -> Result<Vec<ChildEntry>> {
        let mut heap_cursor = Cursor::new(file_data);
        heap_cursor.set_position(st.heap_address);
        let heap = LocalHeap::parse(&mut heap_cursor, self.offset_size(), self.length_size())?;

        let leaves = btree_v1::collect_btree_v1_leaves(
            file_data,
            st.btree_address,
            self.offset_size(),
            self.length_size(),
            None,
            &[],
            None,
        )?;

        let mut children = Vec::new();
        for (_key, snod_address) in &leaves {
            let mut cursor = Cursor::new(file_data);
            cursor.set_position(*snod_address);
            let snod = crate::symbol_table::SymbolTableNode::parse(
                &mut cursor,
                self.offset_size(),
                self.length_size(),
            )?;

            for entry in &snod.entries {
                let name = heap.get_string(entry.link_name_offset, file_data)?;
                children.push(ChildEntry {
                    name,
                    address: entry.object_header_address,
                });
            }
        }

        Ok(children)
    }

    fn resolve_old_style_group_storage(&self, st: &SymbolTableMessage) -> Result<Vec<ChildEntry>> {
        let heap = LocalHeap::parse_at_storage(
            self.context.storage.as_ref(),
            st.heap_address,
            self.offset_size(),
            self.length_size(),
        )?;

        let leaves = btree_v1::collect_btree_v1_leaves_storage(
            self.context.storage.as_ref(),
            st.btree_address,
            self.offset_size(),
            self.length_size(),
            None,
            &[],
            None,
        )?;

        let mut children = Vec::new();
        for (_key, snod_address) in &leaves {
            let header_len = 8 + 2 * usize::from(self.offset_size());
            let prefix = self.context.read_range(*snod_address, header_len)?;
            let mut prefix_cursor = Cursor::new(prefix.as_ref());
            let sig = prefix_cursor.read_bytes(4)?;
            if sig != *b"SNOD" {
                return Err(Error::InvalidData(format!(
                    "expected SNOD signature at offset {:#x}",
                    snod_address
                )));
            }
            let version = prefix_cursor.read_u8()?;
            if version != 1 {
                return Err(Error::InvalidData(format!(
                    "unsupported symbol table node version {}",
                    version
                )));
            }
            prefix_cursor.skip(1)?;
            let num_symbols = prefix_cursor.read_u16_le()?;
            let node_len =
                8 + usize::from(num_symbols) * (2 * usize::from(self.offset_size()) + 4 + 4 + 16);
            let bytes = self.context.read_range(*snod_address, node_len)?;
            let mut cursor = Cursor::new(bytes.as_ref());
            let snod = crate::symbol_table::SymbolTableNode::parse(
                &mut cursor,
                self.offset_size(),
                self.length_size(),
            )?;

            for entry in &snod.entries {
                let name =
                    heap.get_string_storage(entry.link_name_offset, self.context.storage.as_ref())?;
                children.push(ChildEntry {
                    name,
                    address: entry.object_header_address,
                });
            }
        }

        Ok(children)
    }

    /// Resolve dense links from a fractal heap + B-tree v2.
    #[allow(dead_code)]
    fn resolve_dense_links(
        &self,
        link_info: &LinkInfoMessage,
        link_depth: u32,
        file_data: &[u8],
    ) -> Result<Vec<ChildEntry>> {
        let mut heap_cursor = Cursor::new(file_data);
        heap_cursor.set_position(link_info.fractal_heap_address);
        let heap = FractalHeap::parse(&mut heap_cursor, self.offset_size(), self.length_size())?;

        let mut btree_cursor = Cursor::new(file_data);
        btree_cursor.set_position(link_info.btree_name_index_address);
        let btree_header = btree_v2::BTreeV2Header::parse(
            &mut btree_cursor,
            self.offset_size(),
            self.length_size(),
        )?;

        let records = btree_v2::collect_btree_v2_records(
            file_data,
            &btree_header,
            self.offset_size(),
            self.length_size(),
            None,
            &[],
            None,
        )?;

        let mut children = Vec::new();
        for record in &records {
            let heap_id = match record {
                btree_v2::BTreeV2Record::LinkNameHash { heap_id, .. } => heap_id,
                btree_v2::BTreeV2Record::CreationOrder { heap_id, .. } => heap_id,
                _ => continue,
            };

            let managed_bytes = heap.get_managed_object(
                heap_id,
                file_data,
                self.offset_size(),
                self.length_size(),
            )?;

            let mut link_cursor = Cursor::new(&managed_bytes);
            let link_msg = link::parse(
                &mut link_cursor,
                self.offset_size(),
                self.length_size(),
                managed_bytes.len(),
            )?;

            match &link_msg.target {
                LinkTarget::Hard { address } => {
                    children.push(ChildEntry {
                        name: link_msg.name.clone(),
                        address: *address,
                    });
                }
                LinkTarget::Soft { path } => {
                    if let Ok(address) = self.resolve_soft_link_depth(path, link_depth) {
                        children.push(ChildEntry {
                            name: link_msg.name.clone(),
                            address,
                        });
                    }
                }
                LinkTarget::External { .. } => {}
            }
        }

        Ok(children)
    }

    fn resolve_dense_links_storage(
        &self,
        link_info: &LinkInfoMessage,
        link_depth: u32,
    ) -> Result<Vec<ChildEntry>> {
        let heap = FractalHeap::parse_at_storage(
            self.context.storage.as_ref(),
            link_info.fractal_heap_address,
            self.offset_size(),
            self.length_size(),
        )?;

        let btree_header = btree_v2::BTreeV2Header::parse_at_storage(
            self.context.storage.as_ref(),
            link_info.btree_name_index_address,
            self.offset_size(),
            self.length_size(),
        )?;

        let records = btree_v2::collect_btree_v2_records_storage(
            self.context.storage.as_ref(),
            &btree_header,
            self.offset_size(),
            self.length_size(),
            None,
            &[],
            None,
        )?;

        let mut children = Vec::new();
        for record in &records {
            let heap_id = match record {
                btree_v2::BTreeV2Record::LinkNameHash { heap_id, .. }
                | btree_v2::BTreeV2Record::CreationOrder { heap_id, .. } => heap_id,
                _ => continue,
            };

            let managed_bytes = heap.get_managed_object_storage(
                heap_id,
                self.context.storage.as_ref(),
                self.offset_size(),
                self.length_size(),
            )?;

            let mut link_cursor = Cursor::new(&managed_bytes);
            let link_msg = link::parse(
                &mut link_cursor,
                self.offset_size(),
                self.length_size(),
                managed_bytes.len(),
            )?;

            match &link_msg.target {
                LinkTarget::Hard { address } => {
                    children.push(ChildEntry {
                        name: link_msg.name.clone(),
                        address: *address,
                    });
                }
                LinkTarget::Soft { path } => {
                    if let Ok(address) = self.resolve_soft_link_depth(path, link_depth) {
                        children.push(ChildEntry {
                            name: link_msg.name.clone(),
                            address,
                        });
                    }
                }
                LinkTarget::External { .. } => {}
            }
        }

        Ok(children)
    }

    pub fn child_name_by_address(&self, address: u64) -> Result<Option<String>> {
        Ok(self
            .resolve_children()?
            .into_iter()
            .find(|child| child.address == address)
            .map(|child| child.name))
    }

    /// Check if the object at the given address is a group (vs a dataset).
    fn is_group_at(&self, address: u64) -> Result<bool> {
        let mut header = (*self.cached_header(address)?).clone();
        header.resolve_shared_messages_storage(
            self.context.storage.as_ref(),
            self.offset_size(),
            self.length_size(),
        )?;
        for msg in &header.messages {
            match msg {
                HdfMessage::SymbolTable(_)
                | HdfMessage::Link(_)
                | HdfMessage::LinkInfo(_)
                | HdfMessage::GroupInfo(_) => return Ok(true),
                HdfMessage::DataLayout(_) => return Ok(false),
                _ => {}
            }
        }
        Ok(true)
    }

    fn try_open_child_dataset(&self, child: &ChildEntry) -> Option<Dataset> {
        let header = self.cached_header(child.address).ok()?;
        Dataset::from_parsed_header(
            crate::dataset::DatasetParseContext {
                context: self.context.clone(),
            },
            child.address,
            child.name.clone(),
            header.as_ref(),
        )
        .ok()
    }

    fn child_is_group(&self, child: &ChildEntry) -> Result<bool> {
        match self.is_group_at(child.address) {
            Ok(is_group) => Ok(is_group),
            Err(_) => Ok(self.try_open_child_dataset(child).is_none()),
        }
    }

    /// Maximum nesting depth for soft link resolution.
    const MAX_SOFT_LINK_DEPTH: u32 = 16;

    fn resolve_soft_link_depth(&self, path: &str, depth: u32) -> Result<u64> {
        if depth >= Self::MAX_SOFT_LINK_DEPTH {
            return Err(Error::Other(format!(
                "soft link resolution exceeded maximum depth ({}) — possible cycle at '{}'",
                Self::MAX_SOFT_LINK_DEPTH,
                path,
            )));
        }

        let parts: Vec<&str> = path
            .trim_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        if parts.is_empty() {
            return Ok(self.root_address);
        }

        let start_addr = if path.starts_with('/') {
            self.root_address
        } else {
            self.address
        };

        let mut current_group = Group::new(
            self.context.clone(),
            start_addr,
            String::new(),
            self.root_address,
        );

        for &part in &parts[..parts.len() - 1] {
            current_group = current_group.group(part)?;
        }

        let target_name = parts[parts.len() - 1];
        let children = current_group.resolve_children_with_link_depth(depth + 1)?;
        for child in &children {
            if child.name == target_name {
                return Ok(child.address);
            }
        }

        Err(Error::Other(format!(
            "soft link target '{}' not found",
            path
        )))
    }
}
