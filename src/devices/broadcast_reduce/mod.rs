mod accumulator;
mod indexing;

use super::fill::FillElements;
use super::Cpu;
use crate::arrays::{Axes2, Axes3, Axes4, Axis, CountElements};
pub(crate) use accumulator::*;
use indexing::*;

pub(crate) trait DeviceReduce<T: CountElements, Axes> {
    type Reduced;
    fn reduce_into<A: Accumulator<T::Dtype>>(r: &mut Self::Reduced, t: &T);
    fn broadcast_into<A: Accumulator<T::Dtype>>(t: &mut T, r: &Self::Reduced);
}

macro_rules! impl_reduce {
    ($ArrTy:ty, $AxesTy:ty, $RedTy:ty, $Accum:tt, {$($Const:tt),*}) => {
        impl<$(const $Const: usize, )*> DeviceReduce<$ArrTy, $AxesTy> for Cpu {
            type Reduced = $RedTy;
            fn reduce_into<A: Accumulator<f32>>(r: &mut Self::Reduced, t: &$ArrTy) {
                Self::fill(r, &mut |x| *x = A::INIT);
                let mut b = BroadcastMut::<_, $AxesTy>::new(r);
                $Accum::<A, _, _, $($Const, )*>(&mut b, t);
            }
            fn broadcast_into<A: Accumulator<f32>>(t: &mut $ArrTy, r: &Self::Reduced) {
                Self::fill(t, &mut |x| *x = A::INIT);
                let b = BroadcastRef::<_, $AxesTy>::new(r);
                $Accum::<A, _, _, $($Const, )*>(t, &b);
            }
        }
    };
}

// 1d -> 0d
impl_reduce!([f32; M], Axis<0>, f32, accum1d, { M });
impl_reduce!([f32; M], Axis<-1>, f32, accum1d, { M });

// 2d -> 1d
impl_reduce!([[f32; N]; M], Axis<0>, [f32; N], accum2d, {M, N});
impl_reduce!([[f32; N]; M], Axis<1>, [f32; M], accum2d, {M, N});
impl_reduce!([[f32; N]; M], Axis<-1>, [f32; M], accum2d, {M, N});

// 2d -> 0d
impl_reduce!([[f32; N]; M], Axes2<0, 1>, f32, accum2d, {M, N});

// 3d -> 2d
impl_reduce!([[[f32; O]; N]; M], Axis<0>, [[f32; O]; N], accum3d, {M, N, O});
impl_reduce!([[[f32; O]; N]; M], Axis<1>, [[f32; O]; M], accum3d, {M, N, O});
impl_reduce!([[[f32; O]; N]; M], Axis<2>, [[f32; N]; M], accum3d, {M, N, O});
impl_reduce!([[[f32; O]; N]; M], Axis<-1>, [[f32; N]; M], accum3d, {M, N, O});

// 3d -> 1d
impl_reduce!([[[f32; O]; N]; M], Axes2<0, 1>, [f32; O], accum3d, {M, N, O});
impl_reduce!([[[f32; O]; N]; M], Axes2<0, 2>, [f32; N], accum3d, {M, N, O});
impl_reduce!([[[f32; O]; N]; M], Axes2<1, 2>, [f32; M], accum3d, {M, N, O});

// 3d -> 0d
impl_reduce!([[[f32; O]; N]; M], Axes3<0, 1, 2>, f32, accum3d, {M, N, O});

// 4d -> 3d
impl_reduce!([[[[f32; P]; O]; N]; M], Axis<0>, [[[f32; P]; O]; N], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axis<1>, [[[f32; P]; O]; M], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axis<2>, [[[f32; P]; N]; M], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axis<3>, [[[f32; O]; N]; M], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axis<-1>, [[[f32; O]; N]; M], accum4d, {M, N, O, P});

// 4d -> 2d
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<0, 1>, [[f32; P]; O], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<0, 2>, [[f32; P]; N], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<0, 3>, [[f32; O]; N], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<1, 2>, [[f32; P]; M], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<1, 3>, [[f32; O]; M], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes2<2, 3>, [[f32; N]; M], accum4d, {M, N, O, P});

// 4d -> 1d
impl_reduce!([[[[f32; P]; O]; N]; M], Axes3<0, 1, 2>, [f32; P], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes3<0, 1, 3>, [f32; O], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes3<0, 2, 3>, [f32; N], accum4d, {M, N, O, P});
impl_reduce!([[[[f32; P]; O]; N]; M], Axes3<1, 2, 3>, [f32; M], accum4d, {M, N, O, P});

// 4d -> 0d
impl_reduce!([[[[f32; P]; O]; N]; M], Axes4<0, 1, 2, 3>, f32, accum4d, {M, N, O, P});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arrays::ZeroElements;

    #[test]
    fn test_reduce_all_1d() {
        let t = [1.0, 2.0, 3.0, 4.0];
        let mut r = 0.0;
        <Cpu as DeviceReduce<_, Axis<0>>>::reduce_into::<MulAccum>(&mut r, &t);
        assert_eq!(r, 24.0);
    }

    #[test]
    fn test_reduce_all_2d() {
        let t = [[1.0, 2.0, 3.0, 4.0], [5.0, -1.0, 2.0, 0.0]];
        let mut r = 0.0;
        <Cpu as DeviceReduce<_, Axes2<0, 1>>>::reduce_into::<AddAccum>(&mut r, &t);
        assert_eq!(r, 16.0);
    }

    #[test]
    fn test_reduce_all_3d() {
        let t = [[[1.0, 2.0], [2.0, 3.0]], [[1.0, 0.5], [0.5, 1.0 / 3.0]]];
        let mut r = 0.0;
        <Cpu as DeviceReduce<_, Axes3<0, 1, 2>>>::reduce_into::<MulAccum>(&mut r, &t);
        assert_eq!(r, 1.0);
    }

    #[test]
    fn test_1d_reductions() {
        let inp = [2., -1., 0., 1., -2.];
        let mut out = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<0>>>::reduce_into::<MaxAccum>(&mut out, &inp);
        assert_eq!(out, 2.);
    }

    #[test]
    fn test_2d_reductions() {
        type T = [[f32; 3]; 2];
        let inp: T = [[-1., 2., -3.], [1., -2., 3.]];

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<0>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(out0, [1., 2., 3.]);

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<1>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(out0, [2., 3.]);
    }

    #[test]
    fn test_3d_reductions() {
        type T = [[[f32; 3]; 2]; 2];
        let inp: T = [
            [[-1., 2., -3.], [1., -2., 3.]],
            [[4., -5., -3.], [-1., 6., -3.]],
        ];

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<0>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(out0, [[4., 2., -3.], [1., 6., 3.]]);

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<1>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(out0, [[1., 2., 3.], [4., 6., -3.]]);

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<2>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(out0, [[2., 3.], [4., 6.]]);
    }

    #[test]
    fn test_4d_reductions() {
        type T = [[[[f32; 3]; 2]; 3]; 2];
        let inp: T = [
            [
                [[-1., 2., -3.], [1., -2., 3.]],
                [[4., -5., -3.], [-1., 6., -3.]],
                [[-2., 3., 4.], [-6., -7., 3.]],
            ],
            [
                [[1., -2., 3.], [-1., -2., -3.]],
                [[-4., 5., 3.], [-1., -6., -3.]],
                [[2., -3., -4.], [-6., 7., -3.]],
            ],
        ];

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<0>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(
            out0,
            [
                [[1., 2., 3.], [1., -2., 3.]],
                [[4., 5., 3.], [-1., 6., -3.]],
                [[2., 3., 4.], [-6., 7., 3.]]
            ]
        );

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<1>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(
            out0,
            [[[4., 3., 4.], [1., 6., 3.]], [[2., 5., 3.], [-1., 7., -3.]]]
        );

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<2>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(
            out0,
            [
                [[1., 2., 3.], [4., 6., -3.], [-2., 3., 4.]],
                [[1., -2., 3.], [-1., 5., 3.], [2., 7., -3.]]
            ]
        );

        let mut out0 = ZeroElements::ZEROS;
        <Cpu as DeviceReduce<_, Axis<3>>>::reduce_into::<MaxAccum>(&mut out0, &inp);
        assert_eq!(
            out0,
            [
                [[2., 3.], [4., 6.], [4., 3.]],
                [[3., -1.], [5., -1.], [2., 7.]]
            ]
        );
    }

    #[test]
    fn test_broadcast_0d_to_1d() {
        let mut a = [-1.0; 3];
        <Cpu as DeviceReduce<_, Axis<0>>>::broadcast_into::<CopyAccum>(&mut a, &1.0);
        assert_eq!(a, [1.0; 3]);

        let mut a = -1.0;
        <Cpu as DeviceReduce<_, Axis<0>>>::reduce_into::<AddAccum>(&mut a, &[1.0, -2.0, 3.0]);
        assert_eq!(a, 2.0);
    }

    #[test]
    fn test_broadcast_1d_to_2d() {
        let mut a = [[0.0; 3]; 2];
        <Cpu as DeviceReduce<_, Axis<0>>>::broadcast_into::<CopyAccum>(&mut a, &[1.0, 2.0, 3.0]);
        assert_eq!(a, [[1.0, 2.0, 3.0]; 2]);

        let mut a = [[0.0; 3]; 2];
        <Cpu as DeviceReduce<_, Axis<1>>>::broadcast_into::<CopyAccum>(&mut a, &[1.0, 2.0]);
        assert_eq!(a, [[1.0; 3], [2.0; 3]]);

        let mut b = [-1.0, 2.0];
        <Cpu as DeviceReduce<_, Axis<1>>>::reduce_into::<AddAccum>(&mut b, &a);
        assert_eq!(b, [3.0, 6.0]);
    }

    #[test]
    fn test_broadcast_1d_to_3d() {
        let mut a = [[[0.0; 3]; 2]; 1];
        <Cpu as DeviceReduce<_, Axes2<0, 1>>>::broadcast_into::<CopyAccum>(
            &mut a,
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(a, [[[1.0, 2.0, 3.0]; 2]]);

        let mut a = [[[0.0; 3]; 2]; 1];
        <Cpu as DeviceReduce<_, Axes2<0, 2>>>::broadcast_into::<CopyAccum>(&mut a, &[1.0, 2.0]);
        assert_eq!(a, [[[1.0; 3], [2.0; 3]]]);

        let mut a = [[[0.0; 3]; 2]; 3];
        <Cpu as DeviceReduce<_, Axes2<1, 2>>>::broadcast_into::<CopyAccum>(
            &mut a,
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(a, [[[1.0; 3]; 2], [[2.0; 3]; 2], [[3.0; 3]; 2]]);
    }

    #[test]
    fn test_broadcast_1d_to_4d() {
        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as DeviceReduce<_, Axes3<0, 1, 2>>>::broadcast_into::<CopyAccum>(
            &mut a,
            &[1.0, 2.0, 3.0],
        );
        assert_eq!(a, [[[[1.0, 2.0, 3.0]; 2]; 1]; 4]);

        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as DeviceReduce<_, Axes3<0, 1, 3>>>::broadcast_into::<CopyAccum>(&mut a, &[1.0, 2.0]);
        assert_eq!(a, [[[[1.0; 3], [2.0; 3]]; 1]; 4]);

        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as DeviceReduce<_, Axes3<0, 2, 3>>>::broadcast_into::<CopyAccum>(&mut a, &[1.0]);
        assert_eq!(a, [[[[1.0; 3]; 2]; 1]; 4]);

        let mut a = [[[[0.0; 3]; 2]; 1]; 4];
        <Cpu as DeviceReduce<_, Axes3<1, 2, 3>>>::broadcast_into::<CopyAccum>(
            &mut a,
            &[1.0, 2.0, 3.0, 4.0],
        );
        assert_eq!(
            a,
            [
                [[[1.0; 3]; 2]; 1],
                [[[2.0; 3]; 2]; 1],
                [[[3.0; 3]; 2]; 1],
                [[[4.0; 3]; 2]; 1]
            ]
        );
    }

    #[test]
    fn test_broadcast_2d_to_3d() {
        let mut a = [[[0.0; 2]; 2]; 1];
        <Cpu as DeviceReduce<_, Axis<0>>>::broadcast_into::<CopyAccum>(
            &mut a,
            &[[1.0, 2.0], [-1.0, -2.0]],
        );
        assert_eq!(a, [[[1.0, 2.0], [-1.0, -2.0]]]);

        let mut a = [[[0.0; 2]; 2]; 1];
        <Cpu as DeviceReduce<_, Axis<1>>>::broadcast_into::<CopyAccum>(&mut a, &[[1.0, 2.0]]);
        assert_eq!(a, [[[1.0, 2.0], [1.0, 2.0]]]);

        let mut a = [[[0.0; 2]; 2]; 1];
        <Cpu as DeviceReduce<_, Axis<2>>>::broadcast_into::<CopyAccum>(&mut a, &[[1.0, 2.0]]);
        assert_eq!(a, [[[1.0, 1.0], [2.0, 2.0]]]);
    }
}
