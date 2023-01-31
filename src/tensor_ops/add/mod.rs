mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use super::ops::*;
use crate::{gradients::*, shapes::*, tensor::*};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct BinaryAddKernelOp;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarAddKernelOp<E> {
    scalar: E,
}

/// Element wise and scalar addition.
///
/// Example:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r = a + dev.ones();
/// assert_eq!(r.array(), [[2.0, 3.0, 4.0], [0.0, -1.0, -2.0]]);
/// ```
///
/// Adding a scalar:
/// ```rust
/// # use dfdx::prelude::*;
/// # let dev: Cpu = Default::default();
/// let a = dev.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]);
/// let r = a + 1.0;
/// assert_eq!(r.array(), [[2.0, 3.0, 4.0], [0.0, -1.0, -2.0]]);
/// ```
pub fn add<S: ConstShape, E: Dtype, D, T: Tape<D> + Merge<RhsTape>, RhsTape: Tape<D>>(
    lhs: Tensor<S, E, D, T>,
    rhs: Tensor<S, E, D, RhsTape>,
) -> Tensor<S, E, D, T>
where
    D: BinaryKernel<BinaryAddKernelOp, E>,
{
    lhs + rhs
}

/// Fallible version of std::ops::Add with compile time known shapes
pub trait TryConstAdd<Rhs = Self>: HasErr {
    fn try_add(self, rhs: Rhs) -> Result<Self, Self::Err>;
}

/// Runtime shape checked addition
pub trait TryCheckedAdd<Rhs = Self>: HasErr {
    fn checked_add(self, rhs: Rhs) -> Option<Self> {
        self.try_checked_add(rhs).map(Result::unwrap)
    }
    fn try_checked_add(self, rhs: Rhs) -> Option<Result<Self, Self::Err>>;
}

/// We don't need to check anything for compile time known shapes
impl<S: ConstShape, E: Dtype, D, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryConstAdd<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<BinaryAddKernelOp, E>,
    LhsTape: Merge<RhsTape>,
{
    /// Requires compile time shapes. See [add]
    fn try_add(self, rhs: Tensor<S, E, D, RhsTape>) -> Result<Self, Self::Err> {
        try_binary_op(BinaryAddKernelOp, self, rhs)
    }
}

/// We require checked operations with dynamic shapes
impl<S: Shape, E: Dtype, D, LhsTape: Tape<D>, RhsTape: Tape<D>>
    TryCheckedAdd<Tensor<S, E, D, RhsTape>> for Tensor<S, E, D, LhsTape>
where
    D: BinaryKernel<BinaryAddKernelOp, E>,
    LhsTape: Merge<RhsTape>,
{
    /// Runtime shape checked. See [add]
    fn try_checked_add(self, rhs: Tensor<S, E, D, RhsTape>) -> Option<Result<Self, Self::Err>> {
        try_checked_binary_op(BinaryAddKernelOp, self, rhs)
    }
}

/// For scalar addition, we support any shape
impl<S: Shape, E: Dtype, D: UnaryKernel<ScalarAddKernelOp<E>, E>, T: Tape<D>> TryConstAdd<E>
    for Tensor<S, E, D, T>
{
    /// See [add]
    fn try_add(self, rhs: E) -> Result<Self, Self::Err> {
        try_unary_op(ScalarAddKernelOp { scalar: rhs }, self)
    }
}

impl<S: Shape, E: Dtype, D: DeviceStorage, LhsTape: Tape<D>, Rhs> std::ops::Add<Rhs>
    for Tensor<S, E, D, LhsTape>
where
    Self: TryConstAdd<Rhs>,
{
    type Output = Self;
    /// See [add]
    fn add(self, rhs: Rhs) -> Self::Output {
        self.try_add(rhs).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::{shapes::*, tensor::*, tensor_ops::*, tests::*};

    #[test]
    fn test_add_0d() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor(1.0);
        let b = dev.tensor(1.0);

        let r = a.trace() + b.clone();
        assert_eq!(r.array(), 2.0);
        let g = r.backward();
        assert_eq!(g.get(&a).array(), 1.0);
        assert_eq!(g.get(&b).array(), 1.0);
    }

    #[test]
    fn test_add_1d() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([1.0, 2.0, 3.0]);
        let b = dev.tensor([1.0, -1.0, 0.0]);

        let r = a.trace() + b.clone();
        assert_eq!(r.array(), [2.0, 1.0, 3.0]);
        let g = r.mean().backward();
        assert_eq!(g.get(&a).array(), [1.0 / 3.0; 3]);
        assert_eq!(g.get(&b).array(), [1.0 / 3.0; 3]);
    }

    #[test]
    fn test_add_2d() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = dev.tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let r = a.trace() + b.clone();
        assert_eq!(
            r.array(),
            [[1.1769, 0.5552, 0.5259], [1.3917, 1.0692, 0.873]]
        );
        let g = r.mean().backward();
        assert_eq!(g.get(&a).array(), [[1.0 / 6.0; 3]; 2]);
        assert_eq!(g.get(&b).array(), [[1.0 / 6.0; 3]; 2]);
    }

    #[test]
    fn test_add_broadcast_bottom() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = dev.tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let a2: Tensor<Rank3<2, 3, 4>, f32, TestDevice> = a.broadcast();
        let b2: Tensor<Rank3<2, 3, 4>, f32, TestDevice> = b.broadcast();

        let r = a2.trace() + b2.clone();
        assert_eq!(
            r.array(),
            [
                [[1.1769f32; 4], [0.5552; 4], [0.5259; 4]],
                [[1.3917; 4], [1.0692; 4], [0.873; 4]]
            ]
        );
        let g = r.mean().backward();
        assert_eq!(g.get(&a2).array(), [[[1.0 / 6.0; 4]; 3]; 2]);
        assert_eq!(g.get(&b2).array(), [[[1.0 / 6.0; 4]; 3]; 2]);
    }

    #[test]
    fn test_add_broadcast_top() {
        let dev: TestDevice = Default::default();
        let a = dev.tensor([[0.6570, 0.1708, 0.1500], [0.5658, 0.7010, 0.8342]]);
        let b = dev.tensor([[0.5199, 0.3844, 0.3759], [0.8259, 0.3682, 0.0388]]);

        let a2: Tensor<Rank3<4, 2, 3>, f32, TestDevice> = a.broadcast();
        let b2: Tensor<Rank3<4, 2, 3>, f32, TestDevice> = b.broadcast();

        let r = a2.trace() + b2.clone();
        assert_eq!(
            r.array(),
            [[[1.1769f32, 0.5552, 0.5259], [1.3917, 1.0692, 0.873]]; 4]
        );
        let g = r.mean().backward();
        assert_eq!(g.get(&a2).array(), [[[1.0 / 6.0; 3]; 2]; 4]);
        assert_eq!(g.get(&b2).array(), [[[1.0 / 6.0; 3]; 2]; 4]);
    }

    #[test]
    fn test_scalar_add_0d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor(0.0);
        let r = x.trace() + 1.0;
        assert_eq!(r.array(), 1.0);
        let g = r.exp().backward();
        assert_eq!(g.get(&x).array(), 1.0f32.exp());
    }

    #[test]
    fn test_scalar_add_1d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([0.0, 1.0, 2.0]);
        let r = x.trace() + 0.5;
        assert_eq!(r.array(), [0.5, 1.5, 2.5]);
        let g = r.exp().sum().backward();
        assert_eq!(g.get(&x).array(), [1.6487212, 4.481689, 12.182494]);
    }

    #[test]
    fn test_scalar_add_2d() {
        let dev: TestDevice = Default::default();
        let x = dev.tensor([[0.0; 2]; 3]);
        let r = x.trace() + 0.5;
        assert_eq!(r.array(), [[0.5; 2]; 3]);
        let g = r.exp().sum().backward();
        assert_eq!(g.get(&x).array(), [[1.6487212; 2]; 3]);
    }
}
