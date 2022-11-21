use crate::{
    arrays::{Dtype, Shape},
    devices::{
        device::{HasErr, UnaryKernel},
        unary_ops, Device,
    },
    gradients::Tape,
    tensor::Tensor,
};

use super::utils::try_unary_op;

/// **Requires Nightly** Reshape `Self` into `T`.
pub trait ReshapeTo<T>: HasErr {
    fn reshape(self) -> T {
        self.try_reshape().unwrap()
    }
    fn try_reshape(self) -> Result<T, Self::Err>;
}

impl<Src: Shape, Dst: Shape + Default, E: Dtype, D: Device, T: Tape<D>>
    ReshapeTo<Tensor<Dst, E, D, T>> for Tensor<Src, E, D, T>
where
    D: UnaryKernel<unary_ops::Reshape<Dst>, Src, Dst, E>,
{
    fn try_reshape(self) -> Result<Tensor<Dst, E, D, T>, Self::Err> {
        try_unary_op(unary_ops::Reshape(Default::default()), self)
    }
}

#[cfg(test)]
mod tests {
    use crate::devices::{AsArray, Zeros};
    use crate::tensor::*;
    use crate::tensor_ops::impl_backward::TryBackward;
    use crate::tensor_ops::impl_mean::MeanTo;
    use crate::tensor_ops::map::TryExp;
    use crate::tests::build_test_device;

    use super::*;

    #[test]
    fn test_valid_reshapes() {
        let dev = build_test_device!();

        let t: Tensor0D<_> = dev.zeros();
        let _: Tensor1D<1, _> = t.clone().reshape();
        let _: Tensor2D<1, 1, _> = t.clone().reshape();
        let _: Tensor3D<1, 1, 1, _> = t.clone().reshape();
        let _: Tensor4D<1, 1, 1, 1, _> = t.clone().reshape();

        let t: Tensor1D<16, _> = dev.zeros();
        let _: Tensor1D<16, _> = t.clone().reshape();
        let _: Tensor2D<2, 8, _> = t.clone().reshape();
        let _: Tensor3D<2, 2, 4, _> = t.clone().reshape();
        let _: Tensor4D<2, 2, 2, 2, _> = t.clone().reshape();

        let t: Tensor2D<2, 8, _> = dev.zeros();
        let _: Tensor1D<16, _> = t.clone().reshape();
        let _: Tensor2D<8, 2, _> = t.clone().reshape();
        let _: Tensor3D<2, 2, 4, _> = t.clone().reshape();
        let _: Tensor4D<2, 2, 2, 2, _> = t.clone().reshape();

        let t: Tensor3D<2, 2, 4, _> = dev.zeros();
        let _: Tensor1D<16, _> = t.clone().reshape();
        let _: Tensor2D<2, 8, _> = t.clone().reshape();
        let _: Tensor3D<4, 2, 2, _> = t.clone().reshape();
        let _: Tensor4D<2, 2, 2, 2, _> = t.clone().reshape();

        let t: Tensor4D<2, 2, 2, 2, _> = dev.zeros();
        let _: Tensor1D<16, _> = t.clone().reshape();
        let _: Tensor2D<2, 8, _> = t.clone().reshape();
        let _: Tensor3D<2, 2, 4, _> = t.clone().reshape();
        let _: Tensor4D<4, 1, 2, 2, _> = t.clone().reshape();
    }

    #[test]
    fn test_1d_reshape() {
        let dev = build_test_device!();
        let a = dev.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let b: Tensor2D<2, 3, _, _> = a.trace().reshape();
        assert_eq!(b.as_array(), [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
        let g = b.exp().mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [0.18419516, 0.20356713, 0.22497648, 0.24863747, 0.2747869, 0.3036865]
        )
    }
}
