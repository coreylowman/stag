use super::Device;
use crate::{arrays::*, gradients::Tape, tensor::Tensor};

/// Computes the [softmax function](https://en.wikipedia.org/wiki/Softmax_function) across
/// `Axes`.
///
/// Equivalent to `exp(log_softmax(t))`.
///
/// **Pytorch equivalent**: `t.softmax(Axes)`
///
/// **Related functions**: [logsumexp()], [log_softmax()]
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor3D<2, 3, 5> = TensorCreator::zeros();
/// let _ = t.softmax::<Axis<2>>();
/// ```
///
/// Using multi axis softmax:
/// ```rust
/// # use dfdx::prelude::*;
/// # let t: Tensor3D<2, 3, 5> = TensorCreator::zeros();
/// let _ = t.softmax::<Axes2<1, 2>>();
/// ```
pub fn softmax<Ax: Axes, S: Shape, E: Dtype, D: Device<E>, T: Tape<D>>(
    t: Tensor<S, E, D, T>,
) -> Tensor<S, E, D, T>
where
    S: ReduceShape<Ax>,
{
    t.softmax::<Ax>()
}

impl<S: Shape, E: Dtype, D: Device<E>, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn softmax<Ax: Axes>(self) -> Self
    where
        S: ReduceShape<Ax>,
    {
        self.try_softmax::<Ax>().unwrap()
    }

    pub fn try_softmax<Ax: Axes>(self) -> Result<Self, D::Err>
    where
        S: ReduceShape<Ax>,
    {
        self.try_log_softmax::<Ax>()?.try_exp()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        arrays::{Axes2, Axis},
        devices::{AsArray, Randn},
        tensor::*,
        tensor_ops::*,
        tests::{assert_close, build_test_device},
    };

    #[test]
    fn test_softmax_1d() {
        let dev = build_test_device!();
        let a = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = a.trace().softmax();
        assert_eq!(
            r.as_array(),
            [0.011656232, 0.031684924, 0.086128555, 0.23412168, 0.6364087]
        );
        let l = r * dev.tensor([0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(l.as_array(), [0.0, 0.0, 0.086128555, 0.0, 0.0]);
        let g = l.mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [
                -0.00020078686,
                -0.00054579525,
                0.015742086,
                -0.0040329117,
                -0.010962591
            ]
        );
    }

    #[test]
    fn test_softmax_2d() {
        let dev = build_test_device!();
        let a = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.trace().softmax::<Axis<1>>();
        assert_eq!(
            r.as_array(),
            [
                [0.09003058, 0.24472849, 0.66524094],
                [0.002355633, 0.047314156, 0.9503302]
            ]
        );
        let l = r * dev.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(
            l.as_array(),
            [[0.09003058, 0.0, 0.0], [0.0, 0.047314156, 0.0]]
        );
        let g = l.mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [
                [0.01365418, -0.0036721744, -0.009982005],
                [-1.85758e-5, 0.0075125876, -0.0074940124]
            ]
        );
    }

    #[test]
    fn test_softmax_2d_0th_axis() {
        let dev = build_test_device!();
        let a = dev.tensor([[-2.0, -1.0, 0.0], [1.0, 4.0, 7.0]]);
        let r = a.trace().softmax::<Axis<0>>();
        assert_eq!(
            r.as_array(),
            [
                [0.047425874, 0.0066928514, 0.0009110514],
                [0.95257413, 0.9933072, 0.9990892]
            ]
        );
        let l = r * dev.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_eq!(
            l.as_array(),
            [[0.047425874, 0.0, 0.0], [0.0, 0.9933072, 0.0]]
        );
        let g = l.mean().backward();
        assert_eq!(
            g.get(&a).as_array(),
            [
                [0.0075294436, -0.0011080095, 0.0],
                [-0.0075294436, 0.0011080056, 0.0]
            ]
        );
    }

    #[test]
    fn test_softmax_3d_to_1d_12() {
        let dev = build_test_device!();
        let t: Tensor3D<2, 3, 4, _> = dev.randn();
        let r = t.trace().softmax::<Axes2<1, 2>>();
        #[rustfmt::skip]
        assert_close(
            &r.as_array(),
            &[
                [[0.08535644, 0.0987266, 0.00366116, 0.04927256], [0.01169326, 0.1515922, 0.00951258, 0.07721686], [0.0776206, 0.23813945, 0.19471556, 0.00249278]],
                [[0.01881982, 0.25171953, 0.02559674, 0.03725754], [0.04064152, 0.314442, 0.02427996, 0.04708378], [0.02791536, 0.14462142, 0.02221143, 0.04541067]],
            ],
        );
    }
}
