//! Implementations of selecting either 1 or Z elements from an axis of an nd array.
//!
//! # Implementation Details
//! There are three cases to handle:
//!
//! ## Selecting 1 element from the 0th axis
//!
//! Just index into input using the single index and assign to output.
//!
//! ## Selecting Z elements from the 0th axis
//!
//! Just index into input for each index and assing to `output[z]`
//!
//! ## Selecting either 1 or Z elements from a non-zero axis
//!
//! Then all three arrays with have the same dimension as the 0th axis.
//! Do a for loop over the 0th axis and recurse!

use super::{Cpu, ForEachElement};
use crate::arrays::CountElements;

pub mod modes {
    pub struct Index;
    pub struct Recurse<const N: usize>;
    pub struct Broadcast<const N: usize>;
}

use modes::*;

pub trait DeviceSelect<T, I, Mode> {
    type Result;

    /// Equivalent to psuedocode `out = inp[indices]`
    fn select_axis(inp: &T, indices: &I, out: &mut Self::Result);

    /// `inp[indices] += out`
    fn select_add(inp: &mut T, indices: &I, out: &Self::Result);
}

impl<T, const M: usize> DeviceSelect<[T; M], usize, Index> for Cpu
where
    Self: ForEachElement<T>,
    T: Copy + CountElements,
    T::Dtype: for<'a> std::ops::AddAssign<&'a T::Dtype>,
{
    type Result = T;

    fn select_axis(inp: &[T; M], indices: &usize, out: &mut Self::Result) {
        *out = inp[*indices];
    }
    fn select_add(inp: &mut [T; M], indices: &usize, out: &Self::Result) {
        Self::foreach_mr(&mut inp[*indices], out, &mut |a, b| *a += b);
    }
}

impl<T, const M: usize, const Z: usize> DeviceSelect<[T; M], [usize; Z], Index> for Cpu
where
    Self: ForEachElement<T>,
    T: Copy + CountElements,
    T::Dtype: for<'a> std::ops::AddAssign<&'a T::Dtype>,
{
    type Result = [T; Z];
    fn select_axis(inp: &[T; M], indices: &[usize; Z], out: &mut Self::Result) {
        for z in 0..Z {
            out[z] = inp[indices[z]];
        }
    }

    fn select_add(inp: &mut [T; M], indices: &[usize; Z], out: &Self::Result) {
        for z in 0..Z {
            Self::foreach_mr(&mut inp[indices[z]], &out[z], &mut |a, b| *a += b);
        }
    }
}

macro_rules! nd_recurse {
    ($Mode:ty, $SubMode:ty) => {
        impl<T, I, const M: usize> DeviceSelect<[T; M], [I; M], $Mode> for Cpu
        where
            Self: DeviceSelect<T, I, $SubMode>,
        {
            type Result = [<Self as DeviceSelect<T, I, $SubMode>>::Result; M];

            fn select_axis(inp: &[T; M], indices: &[I; M], out: &mut Self::Result) {
                for m in 0..M {
                    Self::select_axis(&inp[m], &indices[m], &mut out[m]);
                }
            }
            fn select_add(inp: &mut [T; M], indices: &[I; M], out: &Self::Result) {
                for m in 0..M {
                    Self::select_add(&mut inp[m], &indices[m], &out[m]);
                }
            }
        }
    };
}

nd_recurse!(Recurse<0>, Index);
nd_recurse!(Recurse<1>, Recurse<0>);
nd_recurse!(Recurse<2>, Recurse<1>);
nd_recurse!(Recurse<3>, Recurse<2>);

impl<T, I, const M: usize> DeviceSelect<T, [I; M], Broadcast<0>> for Cpu
where
    Self: DeviceSelect<T, I, Index>,
{
    type Result = [<Self as DeviceSelect<T, I, Index>>::Result; M];

    fn select_axis(inp: &T, indices: &[I; M], out: &mut Self::Result) {
        for m in 0..M {
            Self::select_axis(inp, &indices[m], &mut out[m]);
        }
    }
    fn select_add(inp: &mut T, indices: &[I; M], out: &Self::Result) {
        for m in 0..M {
            Self::select_add(inp, &indices[m], &out[m]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::ZeroElements;

    #[test]
    fn test_select_1d_0() {
        let a = [1.0, 2.0, 3.0];
        let mut b = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &1usize, &mut b);
        assert_eq!(b, 2.0);
    }

    #[test]
    fn test_select_1d_0z() {
        let a = [1.0f32, 2.0, 3.0];
        let mut b = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Index>>::select_axis(&a, &[0, 1, 2, 2, 1, 0], &mut b);
        assert_eq!(b, [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]);
    }

    const A_2D: [[f32; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

    #[test]
    fn test_select_2d_0() {
        let a = A_2D;
        let mut b = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &0, &mut b);
        assert_eq!(b, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_select_2d_0z() {
        let a = A_2D;
        let mut b = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Index>>::select_axis(&a, &[0, 0, 1], &mut b);
        assert_eq!(b, [a[0], a[0], a[1]]);
    }

    #[test]
    fn test_select_2d_1() {
        let a = A_2D;
        let mut b = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Recurse<0>>>::select_axis(&a, &[0, 1], &mut b);
        assert_eq!(b, [1.0, 5.0]);
    }

    #[test]
    fn test_select_2d_1z() {
        let a = A_2D;
        let mut b = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Recurse<0>>>::select_axis(&a, &[[0, 2], [1, 1]], &mut b);
        assert_eq!(b, [[1.0, 3.0], [5.0, 5.0]]);
    }

    #[test]
    fn test_select_broadcast_2d() {
        let a = [[1.0], [2.0]];
        let i = [[0, 1, 0], [1, 1, 1], [0, 0, 0], [1, 0, 1]];
        let mut b = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Broadcast<0>>>::select_axis(&a, &i, &mut b);
        #[rustfmt::skip]
        assert_eq!(b, [[[1.], [2.], [1.]], [[2.], [2.], [2.]], [[1.], [1.], [1.]], [[2.], [1.], [2.]]]);
    }

    #[test]
    fn test_select_add_2d() {
        let mut a = [[0.0; 3]; 2];
        let b = [[1.0, 3.0], [5.0, 5.0]];
        let i = [[0, 2], [1, 1]];
        <Cpu as DeviceSelect<_, _, Recurse<0>>>::select_add(&mut a, &i, &b);
        assert_eq!(a, [[1.0, 0.0, 3.0], [0.0, 10.0, 0.0]]);
    }

    const A_3D: [[[f32; 3]; 2]; 4] = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[-1.0, 2.0, 3.0], [4.0, -5.0, 6.0]],
        [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
        [[1.0, 2.0, -3.0], [-4.0, -5.0, -6.0]],
    ];

    #[test]
    fn test_select_3d_0() {
        let a = A_3D;
        let mut b = ZeroElements::ZEROS;
        Cpu::select_axis(&a, &0, &mut b);
        assert_eq!(b, A_3D[0]);
    }

    #[test]
    fn test_select_3d_0z() {
        let a = A_3D;
        let mut b = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Index>>::select_axis(&a, &[0, 0, 1, 2, 3, 3], &mut b);
        assert_eq!(b, [A_3D[0], A_3D[0], A_3D[1], A_3D[2], A_3D[3], A_3D[3]]);
    }

    #[test]
    fn test_select_3d_1() {
        let a = A_3D;
        let mut b = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Recurse<0>>>::select_axis(&a, &[0, 0, 1, 1], &mut b);
        assert_eq!(b, [A_3D[0][0], A_3D[1][0], A_3D[2][1], A_3D[3][1]]);
    }

    #[test]
    fn test_select_3d_1z() {
        let a = A_3D;
        let mut b = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Recurse<0>>>::select_axis(&a, &[[0], [0], [1], [1]], &mut b);
        assert_eq!(b, [[A_3D[0][0]], [A_3D[1][0]], [A_3D[2][1]], [A_3D[3][1]]]);
    }

    #[test]
    fn test_select_3d_2() {
        let a = A_3D;
        let mut b = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Recurse<1>>>::select_axis(
            &a,
            &[[1, 0], [0, 1], [0, 0], [1, 1]],
            &mut b,
        );
        assert_eq!(
            b,
            [
                [A_3D[0][0][1], A_3D[0][1][0]],
                [A_3D[1][0][0], A_3D[1][1][1]],
                [A_3D[2][0][0], A_3D[2][1][0]],
                [A_3D[3][0][1], A_3D[3][1][1]],
            ]
        );
    }

    #[test]
    fn test_select_3d_2z() {
        let a = A_3D;
        let mut b: [[[f32; 1]; 2]; 4] = ZeroElements::ZEROS;
        <Cpu as DeviceSelect<_, _, Recurse<1>>>::select_axis(
            &a,
            &[[[1], [0]], [[0], [1]], [[0], [0]], [[1], [1]]],
            &mut b,
        );
        assert_eq!(
            b,
            [
                [[A_3D[0][0][1]], [A_3D[0][1][0]]],
                [[A_3D[1][0][0]], [A_3D[1][1][1]]],
                [[A_3D[2][0][0]], [A_3D[2][1][0]]],
                [[A_3D[3][0][1]], [A_3D[3][1][1]]],
            ]
        );
    }
}
