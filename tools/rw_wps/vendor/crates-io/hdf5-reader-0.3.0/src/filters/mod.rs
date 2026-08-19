pub mod deflate;
pub mod fletcher32;
#[cfg(feature = "lz4")]
pub mod lz4;
pub mod nbit;
pub mod scaleoffset;
pub mod shuffle;

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::messages::filter_pipeline::FilterDescription;

/// Standard HDF5 filter IDs.
pub const FILTER_DEFLATE: u16 = 1;
pub const FILTER_SHUFFLE: u16 = 2;
pub const FILTER_FLETCHER32: u16 = 3;
pub const FILTER_SZIP: u16 = 4;
pub const FILTER_NBIT: u16 = 5;
pub const FILTER_SCALEOFFSET: u16 = 6;
/// HDF5 registered LZ4 filter.
pub const FILTER_LZ4: u16 = 32004;

/// A user-supplied filter function.
///
/// Takes the filter description, input data, and element size, then returns
/// the decoded output.
pub type FilterFn = Box<dyn Fn(&FilterDescription, &[u8], usize) -> Result<Vec<u8>> + Send + Sync>;

/// A registry of filter implementations.
///
/// Comes pre-loaded with deflate, shuffle, and fletcher32. Users can register
/// additional filters (e.g., Blosc, LZ4, ZFP) before reading datasets.
pub struct FilterRegistry {
    filters: HashMap<u16, FilterFn>,
}

impl FilterRegistry {
    /// Create a new registry with the built-in filters pre-registered.
    pub fn new() -> Self {
        let mut registry = FilterRegistry {
            filters: HashMap::new(),
        };
        registry.register(
            FILTER_DEFLATE,
            Box::new(|_, data, _| deflate::decompress(data)),
        );
        registry.register(
            FILTER_SHUFFLE,
            Box::new(|_, data, elem_size| Ok(shuffle::unshuffle(data, elem_size))),
        );
        registry.register(
            FILTER_FLETCHER32,
            Box::new(|_, data, _| fletcher32::verify_and_strip(data)),
        );
        registry.register(
            FILTER_NBIT,
            Box::new(|filter, data, _| nbit::decompress(data, &filter.client_data)),
        );
        registry.register(
            FILTER_SCALEOFFSET,
            Box::new(|filter, data, _| scaleoffset::decompress(data, &filter.client_data)),
        );
        #[cfg(feature = "lz4")]
        registry.register(FILTER_LZ4, Box::new(|_, data, _| lz4::decompress(data)));
        registry
    }

    /// Register a custom filter implementation for the given filter ID.
    ///
    /// Overwrites any previously registered filter with the same ID.
    pub fn register(&mut self, id: u16, f: FilterFn) {
        self.filters.insert(id, f);
    }

    /// Apply a single filter by ID.
    pub fn apply(
        &self,
        filter: &FilterDescription,
        data: &[u8],
        element_size: usize,
    ) -> Result<Vec<u8>> {
        match self.filters.get(&filter.id) {
            Some(f) => f(filter, data, element_size),
            None => Err(Error::UnsupportedFilter(format!("filter id {}", filter.id))),
        }
    }
}

impl Default for FilterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply the filter pipeline in reverse (decompression direction) to a chunk.
///
/// HDF5 stores filters in the order they were applied during writing.
/// On read, we apply them in reverse order.
///
/// If `registry` is `None`, the built-in filter set is used.
///
/// `filter_mask` is a bitmask where bit N being set means filter N should be skipped.
pub fn apply_pipeline(
    data: &[u8],
    filters: &[FilterDescription],
    filter_mask: u32,
    element_size: usize,
    registry: Option<&FilterRegistry>,
) -> Result<Vec<u8>> {
    // Count active filters to decide on single-buffer vs double-buffer strategy.
    let active_count = filters
        .iter()
        .enumerate()
        .rev()
        .filter(|(i, _)| filter_mask & (1 << i) == 0)
        .count();

    if active_count == 0 {
        return Ok(data.to_vec());
    }

    // For a single active filter, avoid the double-buffer overhead.
    if active_count == 1 {
        for (i, filter) in filters.iter().enumerate().rev() {
            if filter_mask & (1 << i) != 0 {
                continue;
            }
            return if let Some(reg) = registry {
                reg.apply(filter, data, element_size)
            } else {
                apply_builtin_filter(filter, data, element_size)
            };
        }
    }

    // Multi-filter pipeline: the first stage reads from the borrowed input
    // slice (avoiding a copy), subsequent stages consume the previous output.
    // Each filter stage necessarily allocates (output sizes are unpredictable),
    // but we avoid the initial data.to_vec() copy.
    let mut owned: Option<Vec<u8>> = None;

    for (i, filter) in filters.iter().enumerate().rev() {
        if filter_mask & (1 << i) != 0 {
            continue;
        }

        let input: &[u8] = match &owned {
            Some(buf) => buf,
            None => data,
        };

        owned = Some(if let Some(reg) = registry {
            reg.apply(filter, input, element_size)?
        } else {
            apply_builtin_filter(filter, input, element_size)?
        });
    }

    Ok(owned.unwrap_or_else(|| data.to_vec()))
}

fn apply_builtin_filter(
    filter: &FilterDescription,
    data: &[u8],
    element_size: usize,
) -> Result<Vec<u8>> {
    match filter.id {
        FILTER_DEFLATE => deflate::decompress(data),
        FILTER_SHUFFLE => Ok(shuffle::unshuffle(data, element_size)),
        FILTER_FLETCHER32 => fletcher32::verify_and_strip(data),
        FILTER_SZIP => Err(Error::UnsupportedFilter("szip".into())),
        FILTER_NBIT => nbit::decompress(data, &filter.client_data),
        FILTER_SCALEOFFSET => scaleoffset::decompress(data, &filter.client_data),
        #[cfg(feature = "lz4")]
        FILTER_LZ4 => lz4::decompress(data),
        id => Err(Error::UnsupportedFilter(format!("filter id {}", id))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_registry_default() {
        let registry = FilterRegistry::new();
        // Built-in filters should be registered
        assert!(registry.filters.contains_key(&FILTER_DEFLATE));
        assert!(registry.filters.contains_key(&FILTER_SHUFFLE));
        assert!(registry.filters.contains_key(&FILTER_FLETCHER32));
        assert!(registry.filters.contains_key(&FILTER_NBIT));
        assert!(registry.filters.contains_key(&FILTER_SCALEOFFSET));
    }

    #[test]
    fn test_filter_registry_custom() {
        let mut registry = FilterRegistry::new();
        // Register a no-op custom filter
        registry.register(32000, Box::new(|_, data, _| Ok(data.to_vec())));
        let filter = FilterDescription {
            id: 32000,
            name: None,
            client_data: Vec::new(),
        };
        let result = registry.apply(&filter, &[1, 2, 3], 1).unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_filter_registry_unknown() {
        let registry = FilterRegistry::new();
        let filter = FilterDescription {
            id: 9999,
            name: None,
            client_data: Vec::new(),
        };
        let err = registry.apply(&filter, &[1, 2, 3], 1).unwrap_err();
        assert!(matches!(err, Error::UnsupportedFilter(_)));
    }
}
