use super::device::StridedArray;
use crate::shapes::Shape;
use std::vec::Vec;

pub(crate) struct NdIndex<S: Shape> {
    pub(crate) indices: S::Concrete,
    pub(crate) shape: S::Concrete,
    pub(crate) strides: S::Concrete,
    pub(crate) next: Option<usize>,
    pub(crate) contiguous: Option<usize>,
}

impl<S: Shape> NdIndex<S> {
    #[inline]
    pub(crate) fn new(shape: S, strides: S::Concrete) -> Self {
        Self {
            indices: Default::default(),
            shape: shape.concrete(),
            strides,
            next: Some(0),
            contiguous: (strides == shape.strides()).then(|| shape.num_elements()),
        }
    }
}

impl<S: Shape> NdIndex<S> {
    #[inline(always)]
    pub(crate) fn next(&mut self) -> Option<usize> {
        match self.contiguous {
            Some(numel) => match self.next.as_mut() {
                Some(i) => {
                    let idx = *i;
                    let next = idx + 1;
                    if next >= numel {
                        self.next = None;
                    } else {
                        *i = next;
                    }
                    Some(idx)
                }
                None => None,
            },
            None => self.next_with_idx().map(|(i, _)| i),
        }
    }

    #[inline(always)]
    fn next_with_idx(&mut self) -> Option<(usize, S::Concrete)> {
        match (S::NUM_DIMS, self.next.as_mut()) {
            (_, None) => None,
            (0, Some(i)) => {
                let idx = (*i, self.indices);
                self.next = None;
                Some(idx)
            }
            (_, Some(i)) => {
                let idx = (*i, self.indices);
                let mut dim = S::NUM_DIMS - 1;
                loop {
                    self.indices[dim] += 1;
                    *i += self.strides[dim];

                    if self.indices[dim] < self.shape[dim] {
                        break;
                    }

                    *i -= self.shape[dim] * self.strides[dim];
                    self.indices[dim] = 0;

                    if dim == 0 {
                        self.next = None;
                        break;
                    }

                    dim -= 1;
                }
                Some(idx)
            }
        }
    }
}

pub(crate) struct StridedRefIter<'a, S: Shape, E> {
    data: &'a Vec<E>,
    index: NdIndex<S>,
}

pub(crate) struct StridedMutIter<'a, S: Shape, E> {
    data: &'a mut Vec<E>,
    index: NdIndex<S>,
}

pub(crate) struct StridedRefIndexIter<'a, S: Shape, E> {
    data: &'a Vec<E>,
    index: NdIndex<S>,
}

pub(crate) struct StridedMutIndexIter<'a, S: Shape, E> {
    data: &'a mut Vec<E>,
    index: NdIndex<S>,
}

impl<S: Shape, E: Clone> StridedArray<S, E> {
    #[inline]
    pub(crate) fn buf_iter(&self) -> std::slice::Iter<'_, E> {
        self.data.iter()
    }

    #[inline]
    pub(crate) fn buf_iter_mut(&mut self) -> std::slice::IterMut<'_, E> {
        std::sync::Arc::make_mut(&mut self.data).iter_mut()
    }

    #[inline]
    pub(crate) fn iter(&self) -> StridedRefIter<S, E> {
        StridedRefIter {
            data: self.data.as_ref(),
            index: NdIndex::new(self.shape, self.strides),
        }
    }

    #[inline]
    pub(crate) fn iter_mut(&mut self) -> StridedMutIter<S, E> {
        StridedMutIter {
            data: std::sync::Arc::make_mut(&mut self.data),
            index: NdIndex::new(self.shape, self.strides),
        }
    }

    #[inline]
    pub(crate) fn iter_with_index(&self) -> StridedRefIndexIter<S, E> {
        StridedRefIndexIter {
            data: self.data.as_ref(),
            index: NdIndex::new(self.shape, self.strides),
        }
    }

    #[inline]
    pub(crate) fn iter_mut_with_index(&mut self) -> StridedMutIndexIter<S, E> {
        StridedMutIndexIter {
            data: std::sync::Arc::make_mut(&mut self.data),
            index: NdIndex::new(self.shape, self.strides),
        }
    }
}

pub(crate) trait LendingIterator {
    type Item<'a>
    where
        Self: 'a;
    fn next(&'_ mut self) -> Option<Self::Item<'_>>;
}

impl<'q, S: Shape, E> LendingIterator for StridedRefIter<'q, S, E> {
    type Item<'a> = &'a E where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index.next().map(|i| &self.data[i])
    }
}

impl<'q, S: Shape, E> LendingIterator for StridedMutIter<'q, S, E> {
    type Item<'a> = &'a mut E where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index.next().map(|i| &mut self.data[i])
    }
}

impl<'q, S: Shape, E> LendingIterator for StridedRefIndexIter<'q, S, E> {
    type Item<'a> = (&'a E, S::Concrete) where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index
            .next_with_idx()
            .map(|(i, idx)| (&self.data[i], idx))
    }
}

impl<'q, S: Shape, E> LendingIterator for StridedMutIndexIter<'q, S, E> {
    type Item<'a> = (&'a mut E, S::Concrete) where Self: 'a;
    #[inline(always)]
    fn next(&'_ mut self) -> Option<Self::Item<'_>> {
        self.index
            .next_with_idx()
            .map(|(i, idx)| (&mut self.data[i], idx))
    }
}

#[cfg(test)]
mod tests {
    use crate::shapes::{Rank0, Rank1, Rank2, Rank3};

    use super::*;

    #[test]
    fn test_0d_contiguous_iter() {
        let s: StridedArray<Rank0, f32> = StridedArray {
            data: Arc::new([0.0].to_vec()),
            shape: (),
            strides: ().strides(),
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&0.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_1d_contiguous_iter() {
        let shape = Default::default();
        let s: StridedArray<Rank1<3>, f32> = StridedArray {
            data: Arc::new([0.0, 1.0, 2.0].to_vec()),
            shape,
            strides: shape.strides(),
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&0.0));
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&2.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_contiguous_iter() {
        let shape = Default::default();
        let s: StridedArray<Rank2<2, 3>, f32> = StridedArray {
            data: Arc::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].to_vec()),
            shape,
            strides: shape.strides(),
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&2.0));
        assert_eq!(i.next(), Some(&3.0));
        assert_eq!(i.next(), Some(&4.0));
        assert_eq!(i.next(), Some(&5.0));
        assert_eq!(i.next(), Some(&6.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_broadcasted_0_iter() {
        let s: StridedArray<Rank2<2, 3>, f32> = StridedArray {
            data: Arc::new([1.0, 0.0, -1.0].to_vec()),
            shape: Default::default(),
            strides: [0, 1],
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&0.0));
        assert_eq!(i.next(), Some(&-1.0));
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&0.0));
        assert_eq!(i.next(), Some(&-1.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_broadcasted_1_iter() {
        let s: StridedArray<Rank2<2, 3>, f32> = StridedArray {
            data: Arc::new([1.0, -1.0].to_vec()),
            shape: Default::default(),
            strides: [1, 0],
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&-1.0));
        assert_eq!(i.next(), Some(&-1.0));
        assert_eq!(i.next(), Some(&-1.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_2d_permuted_iter() {
        let s: StridedArray<Rank2<3, 2>, f32> = StridedArray {
            data: Arc::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].to_vec()),
            shape: Default::default(),
            strides: [1, 3],
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&4.0));
        assert_eq!(i.next(), Some(&2.0));
        assert_eq!(i.next(), Some(&5.0));
        assert_eq!(i.next(), Some(&3.0));
        assert_eq!(i.next(), Some(&6.0));
        assert!(i.next().is_none());
    }

    #[test]
    fn test_3d_broadcasted_iter() {
        let s: StridedArray<Rank3<3, 1, 2>, f32> = StridedArray {
            data: Arc::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0].to_vec()),
            shape: Default::default(),
            strides: [2, 0, 1],
        };
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1.0));
        assert_eq!(i.next(), Some(&2.0));
        assert_eq!(i.next(), Some(&3.0));
        assert_eq!(i.next(), Some(&4.0));
        assert_eq!(i.next(), Some(&5.0));
        assert_eq!(i.next(), Some(&6.0));
        assert!(i.next().is_none());
    }
}
