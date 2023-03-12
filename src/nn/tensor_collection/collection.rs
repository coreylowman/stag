#![allow(clippy::type_complexity)]

use crate::{
    prelude::Device,
    shapes::{Dtype, Shape},
    tensor::{OneFillStorage, Tensor, ZeroFillStorage},
};

use super::visitor::TensorVisitor;

#[macro_export]
macro_rules! try_some {
    {$e:expr} => {{
        let f = move || {Some($e)};
        f()
    }};
}

pub type ModuleVisitorOutput<Func, Mod, E, D, E2, D2> = Result<
    Option<<Mod as TensorCollection<E, D>>::Output<E2, D2>>,
    <Func as TensorVisitor<E, D, E2, D2>>::Err,
>;

pub type TensorVisitorOutput<F, S, E, D, E2, D2> =
    Result<Option<Tensor<S, E2, D2>>, <F as TensorVisitor<E, D, E2, D2>>::Err>;

/// A collection of named tensors. Implementing this trait will enable anything
/// that operates on tensors, like resetting, EMA, counting number of params,
/// gradient updates, etc.
pub trait TensorCollection<E: Dtype, D: Device<E>>: Sized {
    type Output<E2: Dtype, D2: Device<E2>>;

    fn iter_tensors<E2: Dtype, D2: Device<E2>, V: ModuleVisitor<Self, E, D, E2, D2>>(
        visitor: &mut V,
    ) -> ModuleVisitorOutput<V::Func, Self, E, D, E2, D2>;
}

/// An object that can visit [TensorCollection]s and [Tensor]s recursively.
pub trait ModuleVisitor<
    T: TensorCollection<E, D>,
    E: Dtype,
    D: Device<E>,
    E2: Dtype,
    D2: Device<E2>,
>: Sized
{
    type Err;
    type Func: TensorVisitor<E, D, E2, D2>;

    /// Visit a [TensorCollection]
    fn visit_module<Field, GetRef, GetMut>(
        &mut self,
        name: &str,
        get_refs: GetRef,
        get_muts: GetMut,
    ) -> ModuleVisitorOutput<Self::Func, Field, E, D, E2, D2>
    where
        GetRef: FnMut(&T) -> &Field,
        GetMut: FnMut(&mut T) -> &mut Field,
        Field: TensorCollection<E, D>;

    /// Visits an actual named [Tensor]
    fn visit_tensor<S: Shape, GetRef, GetMut>(
        &mut self,
        name: &str,
        get_refs: GetRef,
        get_muts: GetMut,
        opts: TensorOptions<S, E, D>,
    ) -> TensorVisitorOutput<Self::Func, S, E, D, E2, D2>
    where
        GetRef: FnMut(&T) -> &Tensor<S, E, D>,
        GetMut: FnMut(&mut T) -> &mut Tensor<S, E, D>;
}

impl<S: Shape, E: Dtype, D: Device<E>> TensorCollection<E, D> for Tensor<S, E, D> {
    type Output<E2: Dtype, D2: Device<E2>> = Tensor<S, E2, D2>;

    fn iter_tensors<E2: Dtype, D2: Device<E2>, V: ModuleVisitor<Self, E, D, E2, D2>>(
        visitor: &mut V,
    ) -> ModuleVisitorOutput<V::Func, Self, E, D, E2, D2> {
        visitor.visit_tensor(
            "",
            |s| s,
            |s| s,
            TensorOptions {
                do_gradient_update: true,
                reset: |_| Ok(()),
                shape: Default::default(),
            },
        )
    }
}

/// Options to change behavior of [ModuleVisitor]
#[non_exhaustive]
pub struct TensorOptions<S: Shape, E: Dtype, D: Device<E>> {
    /// Whether the tensor should be updated with gradients
    pub do_gradient_update: bool,

    /// How to reset the tensor in the future.
    pub reset: fn(&'_ mut Tensor<S, E, D>) -> Result<(), D::Err>,

    /// The [Shape] that [BuildModule] uses to construct the tensor
    pub shape: S,
}

impl<S: Shape, E: Dtype, D: Device<E>> TensorOptions<S, E, D> {
    /// A tensor that should be updated with gradients & reset to 0
    pub fn reset_to_zeros() -> Self
    where
        D: ZeroFillStorage<E>,
    {
        TensorOptions {
            do_gradient_update: true,
            reset: |t| t.try_fill_with_zeros(),
            shape: Default::default(),
        }
    }

    /// A tensor that should be updated with gradients & reset to 1
    pub fn reset_to_ones() -> Self
    where
        D: OneFillStorage<E>,
    {
        TensorOptions {
            do_gradient_update: true,
            reset: |t| t.try_fill_with_ones(),
            shape: Default::default(),
        }
    }

    /// A tensor that should be updated with gradients & reset with the fn passed in
    pub fn reset_with(reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>) -> Self {
        TensorOptions {
            do_gradient_update: true,
            reset,
            shape: Default::default(),
        }
    }

    /// A tensor that should **NOT** be updated with gradients & reset with the fn passed in
    pub fn detached(reset: fn(&mut Tensor<S, E, D>) -> Result<(), D::Err>) -> Self {
        TensorOptions {
            do_gradient_update: false,
            reset,
            shape: Default::default(),
        }
    }

    /// Sets `self.shape` to `shape`
    pub fn with_shape(mut self, shape: S) -> Self {
        self.shape = shape;
        self
    }
}
