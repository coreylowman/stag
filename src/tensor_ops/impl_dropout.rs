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

/// Zeros elements with probability `p` and scales all elements by `1 / (1 - p)`.
///
/// Described in paper: [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # use rand::prelude::*;
/// let mut rng = StdRng::seed_from_u64(4);
/// let t = tensor([1.0, 2.0, 3.0, 4.0]);
///
/// // no tape in t, this won't do anything
/// let a = dropout(t.clone(), 0.5, &mut rng);
/// assert_eq!(a.data(), t.data());
///
/// // now t has the tape, dropout!
/// let a = dropout(t.trace(), 0.5, &mut rng);
/// assert_eq!(a.data(), &[2.0, 4.0, 0.0, 8.0]);
/// ```
///
/// ### Implementation details:
///
/// To reduce memory usage, this function first samples a u64 seed from `rng`,
/// and then instantiates two identical [StdRng] with that seed. These rngs
/// are used in both the forward pass and backward pass to generate identical
/// random numbers, so the masking is the same for both.
pub trait TryDropout: HasErr {
    fn dropout(self, prob: f32) -> Self {
        self.try_dropout(prob).unwrap()
    }
    fn try_dropout(self, prob: f32) -> Result<Self, Self::Err>;
}

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> TryDropout for Tensor<S, E, D, T>
where
    D: UnaryKernel<unary_ops::Dropout, S, S, E>,
{
    fn try_dropout(self, prob: f32) -> Result<Self, Self::Err> {
        let seed = self.device.random_u64();
        try_unary_op(unary_ops::Dropout { seed, prob }, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::AsArray;
    use crate::tensor::*;
    use crate::tensor_ops::impl_backward::TryBackward;
    use crate::tensor_ops::impl_mean::MeanTo;
    use crate::tensor_ops::map::TryExp;
    use crate::tests::{assert_close, build_test_device};

    #[test]
    fn test_dropout_all_0d() {
        let dev = build_test_device!();
        let t: Tensor0D<_> = dev.tensor(3.0);
        let r = t.trace().dropout(1.0);
        assert_eq!(r.as_array(), 0.0);
        let g = r.backward();
        assert_eq!(g.get(&t).as_array(), 0.0);
    }

    #[test]
    fn test_dropout_none_0d() {
        let dev = build_test_device!();
        let t: Tensor0D<_> = dev.tensor(3.0);
        let r = t.trace().dropout(0.0);
        assert_eq!(r.as_array(), 3.0);
        let g = r.backward();
        assert_eq!(g.get(&t).as_array(), 1.0);
    }

    #[test]
    fn test_dropout_1d_with_non_positive_values() {
        let dev = build_test_device!();
        let t = dev.tensor([0.0, 2.0, -3.0, -4.0, 0.0]);
        let r = t.trace().dropout(0.5);
        assert_eq!(r.as_array(), [0.0, 4.0, -6.0, 0.0, 0.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&t).as_array(), [0.4, 0.4, 0.4, 0.0, 0.0]);
    }

    #[test]
    fn test_dropout_2d() {
        let dev = build_test_device!();
        let t = dev.tensor([[0.05, 0.1, -0.2], [0.3, -0.4, 0.5]]);
        let r = t.trace().dropout(0.6);
        assert_close(&r.as_array(), &[[0.125, 0.25, -0.5], [0.0, 0.0, 1.25]]);
        // NOTE: .exp() so we ensure result grad is used properly
        let g = r.exp().mean().backward();
        assert_eq!(
            g.get(&t).as_array(),
            [[0.47214523, 0.5350107, 0.2527211], [0.0, 0.0, 1.4543099]]
        );
    }
}
