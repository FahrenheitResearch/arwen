//! # unsafe_cell_slice
//!
//! A Rust microlibrary for creating multiple mutable references to a [`slice`].
//!
//! ### Motivation
//! The rust borrow checker forbids creating multiple mutable references of a [`slice`].
//! For example, this fails to compile with ```cannot borrow `data` as mutable more than once at a time```:
//! ```rust,compile_fail
//! let mut data = vec![0u8; 2];
//! let data_a: &mut u8 = &mut data[0];
//! let data_b: &mut u8 = &mut data[1];
//! *data_a = 0;
//! *data_b = 1;
//! ```
//!
//! There are use cases for acquiring multiple mutable references of a [`slice`], such as for writing independent elements in parallel.
//! A safe approach is to borrow non-overlapping slices via [`slice::split_at_mut`], [`slice::chunks_mut`], etc.
//! However, such approaches may not be applicable in complicated use cases, such as writing to interleaved elements.
//!
//! ### [`UnsafeCellSlice`]
//! An [`UnsafeCellSlice`] can be created from a mutable slice or the spare capacity in a [`Vec`].
//! It has unsafe [`get_mut`](UnsafeCellSlice::get_mut) and [`index_mut`](UnsafeCellSlice::index_mut) methods that permit creating multiple mutable references of subslices or elements.
//!
//! ```rust
//! # use unsafe_cell_slice::UnsafeCellSlice;
//! let mut data = vec![0u8; 2];
//! {
//!     let data = UnsafeCellSlice::new(&mut data);
//!     let data_a: &mut u8 = unsafe { data.index_mut(0) };
//!     let data_b: &mut u8 = unsafe { data.index_mut(1) };
//!     *data_a = 0;
//!     *data_b = 1;
//! }
//! assert_eq!(data[0], 0);
//! assert_eq!(data[1], 1);
//! ```
//!
//! Note that this is very unsafe and bypasses Rust's safety guarantees!
//! It is the responsibility of the caller of [`UnsafeCellSlice`] methods to avoid data races and undefined behavior by not requesting overlapping subslices/elements.
//!
//! Under the hood, [`UnsafeCellSlice`] is a reference to a [`std::cell::UnsafeCell`] slice, hence the name of the crate.
//!
//! ## Licence
//! `unsafe_cell_slice` is licensed under either of
//!  - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/unsafe_cell_slice/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//!  - the MIT license [LICENSE-MIT](https://docs.rs/crate/unsafe_cell_slice/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

mod slice_index;

pub use slice_index::SliceIndex;

/// An unsafe cell slice. Permits acquisition of multiple mutable references of a slice.
///
/// This is inherently unsafe.
/// It is the responsibility of the caller to only access non-overlapping subslices/elements to avoid data races and undefined behavior.
#[derive(Copy, Clone)]
pub struct UnsafeCellSlice<'a, T>(&'a [std::cell::UnsafeCell<T>]);

unsafe impl<T: Send + Sync> Send for UnsafeCellSlice<'_, T> {}
unsafe impl<T: Send + Sync> Sync for UnsafeCellSlice<'_, T> {}

impl<'a, T> UnsafeCellSlice<'a, T> {
    /// Create a new [`UnsafeCellSlice`] from a mutable slice.
    #[must_use]
    pub fn new(slice: &'a mut [T]) -> Self {
        // Rust 1.76: std::ptr::from_mut::<[T]>(slice)
        let ptr = slice as *mut [T] as *const [std::cell::UnsafeCell<T>];
        Self(unsafe { &*ptr })
    }

    /// Create a new [`UnsafeCellSlice`] from the spare capacity in a [`Vec`].
    #[must_use]
    pub fn new_from_vec_with_spare_capacity(vec: &'a mut Vec<T>) -> Self {
        Self::new(unsafe { vec_spare_capacity_to_mut_slice(vec) })
    }

    /// Return the length of the underlying slice.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Reutrn whether the underlying slice is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a mutable reference to a subslice or element of the underlying slice.
    ///
    /// Returns `None` if the index is out of bounds.
    ///
    /// # Safety
    /// This is very unsafe because it is capable of creating multiple mutable references to the same data.
    /// It is the responsibility of the caller to only access non-overlapping subslices to avoid data races and undefined behavior.
    #[must_use]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn get_mut<I>(&self, index: I) -> Option<&mut I::Output>
    where
        I: SliceIndex<T>,
    {
        index.get_mut(self)
    }

    /// Get a mutable reference to a subslice or element of the underlying slice.
    ///
    /// Note that unlike [`std::ops::IndexMut::index_mut`], `self` is not a mutable reference.
    /// Thus, this method does not support desuraging.
    ///
    /// # Safety
    /// This is very unsafe because it is capable of creating multiple mutable references to the same data.
    /// It is the responsibility of the caller to only access non-overlapping subslices to avoid data races and undefined behavior.
    ///
    /// # Panics
    /// May panic if the index is out of bounds.
    #[must_use]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn index_mut<I>(&self, index: I) -> &mut I::Output
    where
        I: SliceIndex<T>,
    {
        index.index_mut(self)
    }
}

/// Get a mutable slice of the spare capacity in a vector.
///
/// # Safety
/// Returned elements are uninitialised.
#[allow(unused_unsafe)]
unsafe fn vec_spare_capacity_to_mut_slice<T>(vec: &mut Vec<T>) -> &mut [T] {
    let spare_capacity = vec.spare_capacity_mut();
    unsafe {
        std::slice::from_raw_parts_mut(
            spare_capacity.as_mut_ptr().cast::<T>(),
            spare_capacity.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn empty() {
        {
            let mut data: Vec<u8> = vec![];
            let data = super::UnsafeCellSlice::new(&mut data);
            assert!(data.is_empty());
        }
        {
            let mut data: Vec<u8> = vec![1];
            let data = super::UnsafeCellSlice::new(&mut data);
            assert!(!data.is_empty());
        }
    }
}
