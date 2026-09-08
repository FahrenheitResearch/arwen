use crate::UnsafeCellSlice;

mod private_slice_index {
    pub trait Sealed {}
}

/// A trait for indexing into an [`UnsafeCellSlice`].
///
/// # Safety
/// Callers of the trait methods must ensure that overlapping subslices/elements are not requested.
pub unsafe trait SliceIndex<T>: private_slice_index::Sealed {
    type Output: ?Sized;

    /// Returns a mutable reference to the output at this location, if in bounds.
    fn get_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> Option<&'a mut Self::Output>;

    /// Returns a mutable reference to the output at this location, panicking
    /// if out of bounds.
    #[allow(clippy::mut_from_ref)]
    fn index_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> &'a mut Self::Output;
}

// impl private_slice_index::Sealed for (Bound<usize>, Bound<usize>) {}
impl private_slice_index::Sealed for usize {}
impl private_slice_index::Sealed for std::ops::Range<usize> {}
impl private_slice_index::Sealed for std::ops::RangeFrom<usize> {}
impl private_slice_index::Sealed for std::ops::RangeFull {}
impl private_slice_index::Sealed for std::ops::RangeInclusive<usize> {}
impl private_slice_index::Sealed for std::ops::RangeTo<usize> {}
impl private_slice_index::Sealed for std::ops::RangeToInclusive<usize> {}

// TODO (Bound<usize>, Bound<usize>) is not implemented

unsafe impl<T> SliceIndex<T> for usize {
    type Output = T;

    fn get_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> Option<&'a mut Self::Output> {
        slice.0.get(self).map(|v| unsafe { &mut *v.get() })
    }

    fn index_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> &'a mut Self::Output {
        self.get_mut(slice).expect("index out of bounds")
    }
}

unsafe impl<T> SliceIndex<T> for std::ops::Range<usize> {
    type Output = [T];

    fn get_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> Option<&'a mut Self::Output> {
        slice
            .0
            .get(self)
            .map(|s| unsafe { std::slice::from_raw_parts_mut(s.as_ptr() as *mut T, s.len()) })
    }

    fn index_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> &'a mut Self::Output {
        self.get_mut(slice).expect("index out of bounds")
    }
}

unsafe impl<T> SliceIndex<T> for std::ops::RangeFrom<usize> {
    type Output = [T];

    fn get_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> Option<&'a mut Self::Output> {
        slice
            .0
            .get(self)
            .map(|s| unsafe { std::slice::from_raw_parts_mut(s.as_ptr() as *mut T, s.len()) })
    }

    fn index_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> &'a mut Self::Output {
        self.get_mut(slice).expect("index out of bounds")
    }
}

unsafe impl<T> SliceIndex<T> for std::ops::RangeFull {
    type Output = [T];

    fn get_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> Option<&'a mut Self::Output> {
        slice
            .0
            .get(self)
            .map(|s| unsafe { std::slice::from_raw_parts_mut(s.as_ptr() as *mut T, s.len()) })
    }

    fn index_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> &'a mut Self::Output {
        self.get_mut(slice).expect("index out of bounds")
    }
}

unsafe impl<T> SliceIndex<T> for std::ops::RangeInclusive<usize> {
    type Output = [T];

    fn get_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> Option<&'a mut Self::Output> {
        slice
            .0
            .get(self)
            .map(|s| unsafe { std::slice::from_raw_parts_mut(s.as_ptr() as *mut T, s.len()) })
    }

    fn index_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> &'a mut Self::Output {
        self.get_mut(slice).expect("index out of bounds")
    }
}

unsafe impl<T> SliceIndex<T> for std::ops::RangeTo<usize> {
    type Output = [T];

    fn get_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> Option<&'a mut Self::Output> {
        slice
            .0
            .get(self)
            .map(|s| unsafe { std::slice::from_raw_parts_mut(s.as_ptr() as *mut T, s.len()) })
    }

    fn index_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> &'a mut Self::Output {
        self.get_mut(slice).expect("index out of bounds")
    }
}

unsafe impl<T> SliceIndex<T> for std::ops::RangeToInclusive<usize> {
    type Output = [T];

    fn get_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> Option<&'a mut Self::Output> {
        slice
            .0
            .get(self)
            .map(|s| unsafe { std::slice::from_raw_parts_mut(s.as_ptr() as *mut T, s.len()) })
    }

    fn index_mut<'a>(self, slice: &'a UnsafeCellSlice<T>) -> &'a mut Self::Output {
        self.get_mut(slice).expect("index out of bounds")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_index_usize() {
        let mut data = vec![0i64, 1i64];
        let data = UnsafeCellSlice::new(&mut data);
        let data_a: &mut i64 = unsafe { data.index_mut(0) };
        let data_b: &mut i64 = unsafe { data.index_mut(1) };
        assert_eq!(*data_a, 0);
        assert_eq!(*data_b, 1);
        assert!(unsafe { data.get_mut(0) }.is_some());
        assert!(unsafe { data.get_mut(1) }.is_some());
        assert!(unsafe { data.get_mut(2) }.is_none());
    }

    #[test]
    fn slice_index_range() {
        let mut data = vec![0i64, 1i64, 2i64];
        let data = UnsafeCellSlice::new(&mut data);
        {
            let data_ab: &mut [i64] = unsafe { data.index_mut(0..2) };
            assert_eq!(data_ab.len(), 2);
            assert_eq!(data_ab, [0, 1]);
        }
        {
            let data_a: &mut [i64] = unsafe { data.index_mut(0..1) };
            assert_eq!(data_a.len(), 1);
            assert_eq!(data_a, [0]);
        }
        {
            let data_b: &mut [i64] = unsafe { data.index_mut(1..2) };
            assert_eq!(data_b.len(), 1);
            assert_eq!(data_b, [1]);
        }
        assert_eq!(unsafe { data.get_mut(0..2) }.unwrap().len(), 2);
        assert_eq!(unsafe { data.get_mut(0..1) }.unwrap().len(), 1);
        assert_eq!(unsafe { data.get_mut(1..2) }.unwrap().len(), 1);
        assert_eq!(unsafe { data.get_mut(0..3) }.unwrap().len(), 3);
        assert!(unsafe { data.get_mut(0..4) }.is_none());
        assert!(unsafe { data.get_mut(2..4) }.is_none());
        assert!(unsafe { data.get_mut(2..0) }.is_none());
    }

    #[test]
    fn slice_index_range_from() {
        let mut data = vec![0i64, 1i64, 2i64];
        let data = UnsafeCellSlice::new(&mut data);
        {
            let data_ab: &mut [i64] = unsafe { data.index_mut(0..) };
            assert_eq!(data_ab.len(), 3);
            assert_eq!(data_ab, [0, 1, 2]);
        }
        {
            let data_b: &mut [i64] = unsafe { data.index_mut(1..) };
            assert_eq!(data_b.len(), 2);
            assert_eq!(data_b, [1, 2]);
        }
        {
            let data_b: &mut [i64] = unsafe { data.index_mut(2..) };
            assert_eq!(data_b.len(), 1);
            assert_eq!(data_b, [2]);
        }
        assert_eq!(unsafe { data.get_mut(0..) }.unwrap().len(), 3);
        assert_eq!(unsafe { data.get_mut(1..) }.unwrap().len(), 2);
        assert!(unsafe { data.get_mut(2..) }.is_some());
        assert_eq!(unsafe { data.get_mut(3..) }.unwrap().len(), 0);
        assert!(unsafe { data.get_mut(4..) }.is_none());
    }

    #[test]
    fn slice_index_range_full() {
        {
            let mut data = vec![0i64, 1i64];
            let data = UnsafeCellSlice::new(&mut data);
            let data_ab: &mut [i64] = unsafe { data.index_mut(..) };
            assert_eq!(data_ab.len(), 2);
            assert_eq!(data_ab, [0, 1]);
            assert_eq!(unsafe { data.get_mut(..) }.unwrap().len(), 2);
        }
        {
            let mut data: Vec<u8> = vec![];
            let data = UnsafeCellSlice::new(&mut data);
            assert_eq!(unsafe { data.get_mut(..) }.unwrap().len(), 0);
        }
    }

    #[test]
    fn slice_index_range_inclusive() {
        let mut data = vec![0i64, 1i64, 2i64, 3i64];
        let data = UnsafeCellSlice::new(&mut data);
        {
            let data_abc: &mut [i64] = unsafe { data.index_mut(0..=2) };
            assert_eq!(data_abc.len(), 3);
            assert_eq!(data_abc, [0, 1, 2]);
        }
        {
            let data_cd: &mut [i64] = unsafe { data.index_mut(2..=3) };
            assert_eq!(data_cd.len(), 2);
            assert_eq!(data_cd, [2, 3]);
        }
        {
            let data_ab: &mut [i64] = unsafe { data.index_mut(1..=2) };
            assert_eq!(data_ab.len(), 2);
            assert_eq!(data_ab, [1, 2]);
        }
        {
            let data_a: &mut [i64] = unsafe { data.index_mut(0..=0) };
            assert_eq!(data_a.len(), 1);
            assert_eq!(data_a, [0]);
        }
        {
            let data_b: &mut [i64] = unsafe { data.index_mut(1..=1) };
            assert_eq!(data_b.len(), 1);
            assert_eq!(data_b, [1]);
        }
        assert_eq!(unsafe { data.get_mut(0..=1) }.unwrap().len(), 2);
        assert_eq!(unsafe { data.get_mut(1..=1) }.unwrap().len(), 1);
        assert!(unsafe { data.get_mut(0..=4) }.is_none());
        assert!(unsafe { data.get_mut(2..=4) }.is_none());
        assert!(unsafe { data.get_mut(2..=0) }.is_none());
    }

    #[test]
    fn slice_index_range_to() {
        let mut data = vec![0i64, 1i64];
        let data = UnsafeCellSlice::new(&mut data);
        {
            let data_ab: &mut [i64] = unsafe { data.index_mut(..2) };
            assert_eq!(data_ab.len(), 2);
            assert_eq!(data_ab, [0, 1]);
        }
        {
            let data_a: &mut [i64] = unsafe { data.index_mut(..1) };
            assert_eq!(data_a.len(), 1);
            assert_eq!(data_a, [0]);
        }
        {
            let data_0: &mut [i64] = unsafe { data.index_mut(..0) };
            assert_eq!(data_0.len(), 0);
            assert_eq!(data_0, []);
        }
        assert_eq!(unsafe { data.get_mut(..2) }.unwrap().len(), 2);
        assert!(unsafe { data.get_mut(..3) }.is_none());
    }

    #[test]
    fn slice_index_range_to_inclusive() {
        let mut data = vec![0i64, 1i64, 2i64, 3i64];
        let data = UnsafeCellSlice::new(&mut data);
        {
            let data_abcd: &mut [i64] = unsafe { data.index_mut(..=3) };
            assert_eq!(data_abcd.len(), 4);
            assert_eq!(data_abcd, [0, 1, 2, 3]);
        }
        {
            let data_abc: &mut [i64] = unsafe { data.index_mut(..=2) };
            assert_eq!(data_abc.len(), 3);
            assert_eq!(data_abc, [0, 1, 2]);
        }
        {
            let data_ab: &mut [i64] = unsafe { data.index_mut(..=1) };
            assert_eq!(data_ab.len(), 2);
            assert_eq!(data_ab, [0, 1]);
        }
        {
            let data_a: &mut [i64] = unsafe { data.index_mut(..=0) };
            assert_eq!(data_a.len(), 1);
            assert_eq!(data_a, [0]);
        }
        assert_eq!(unsafe { data.get_mut(..=0) }.unwrap().len(), 1);
        assert_eq!(unsafe { data.get_mut(..=1) }.unwrap().len(), 2);
        assert_eq!(unsafe { data.get_mut(..=2) }.unwrap().len(), 3);
        assert_eq!(unsafe { data.get_mut(..=3) }.unwrap().len(), 4);
        assert!(unsafe { data.get_mut(..=4) }.is_none());
    }
}
