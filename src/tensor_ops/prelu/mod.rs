mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use core::fmt::Debug;

use crate::{shapes::*, tensor::*};

use super::{
    ops::{try_binary_op, try_unary_op, BinaryKernel, UnaryKernel},
    Device,
};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct PReLUKernelOp;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct LeakyReLUKernelOp<E> {
    slope: E,
}

/// [Parametric Rectified Linear Unit (PReLU)](https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html). `max(0, t) + a*min(0, t)`
///
/// Examples:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let t = dev.tensor([-1.0, 0.0, 1.0, 2.0]);
/// let a = dev.tensor([0.05,0.05,0.05,0.05]);
/// let r = prelu(t, a);
/// assert_eq!(r.array(), [-0.05, 0.0, 1.0, 2.0]);
/// ```

pub fn prelu<
    S: Shape,
    E: Dtype,
    D: Device<E> + BinaryKernel<PReLUKernelOp, E>,
    T: Tape<E, D> + Merge<R>,
    R: Default,
>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, R>,
) -> Tensor<S, E, D, T> {
    lhs.prelu(rhs)
}

pub fn leakyrelu<
    S: Shape,
    E: Dtype,
    D: Device<E> + UnaryKernel<LeakyReLUKernelOp<E>, E>,
    T: Tape<E, D>,
>(
    lhs: Tensor<S, E, D, T>,
    rhs: E,
) -> Tensor<S, E, D, T> {
    lhs.prelu(rhs)
}

pub trait TryPReLU<T = Self>: HasErr {
    type Output;

    fn try_prelu(self, rhs: T) -> Result<Self::Output, Self::Err>;

    fn prelu(self, rhs: T) -> Self::Output {
        self.try_prelu(rhs).unwrap()
    }
}

impl<S: Shape, E: Dtype, D, LhsTape: Tape<E, D>, R> TryPReLU<Tensor<S, E, D, R>>
    for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<PReLUKernelOp, E>,
    LhsTape: Merge<R>,
{
    type Output = Self;

    /// See [prelu]
    fn try_prelu(self, rhs: Tensor<S, E, D, R>) -> Result<Self, Self::Err> {
        try_binary_op(PReLUKernelOp, self, rhs)
    }
}

impl<S: Shape, E: Dtype, D: UnaryKernel<LeakyReLUKernelOp<E>, E>, T: Tape<E, D>> TryPReLU<E>
    for Tensor<S, E, D, T>
{
    type Output = Self;

    /// See [prelu]
    fn try_prelu(self, rhs: E) -> Result<Self, Self::Err> {
        try_unary_op(LeakyReLUKernelOp { slope: rhs }, self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tensor::*,
        tensor_ops::{prelu::TryPReLU, *},
        tests::*,
    };

    #[test]
    fn test_prelu() {
        let dev: TestDevice = Default::default();
        let x: Tensor<_, TestDtype, _> = dev.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y: Tensor<_, TestDtype, _> = dev.tensor([0.05, 0.05, 0.05, 0.05, 0.05]);
        let r = x.leaky_trace().prelu(y.clone());
        assert_eq!(r.array(), [-0.1, -0.05, 0.0, 1.0, 2.0]);
        // NOTE: call .exp() to make sure we cover cases where .prelu() uses the result's gradient
        let g = r.exp().mean().backward();
        assert_close(
            &g.get(&x).array(),
            &[0.00904837, 0.00951229, 0.2, 0.54365635, 1.4778112],
        );
        // TODO: confirm this
        assert_close(&g.get(&y).array(), &[-0.3619348, -0.1902458, 0.0, 0.0, 0.0]);
    }
}