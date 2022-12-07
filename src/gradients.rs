//! Implementations of [GradientTape] and generic Nd array containers via [Gradients].
#![allow(clippy::type_complexity)]

use core::marker::PhantomData;
use std::collections::HashMap;
use std::{boxed::Box, vec::Vec};

use crate::shapes::{HasDtype, HasShape};
use crate::tensor::storage_traits::{AllocGrad, DeviceStorage};
use crate::unique_id::{HasUniqueId, UniqueId};

/// A generic container for keeping variable sized arrays associated with a [UniqueId].
///
/// You can:
/// 1. Insert array values into it
/// 2. Remove entries
/// 3. Access references to arrays
/// 4. Access mutable references to arrays
///
/// This structure is similar to a HashMap, where all the methods require a key
/// implementing [UniqueId] and [HasArrayType].
///
/// Under the hood, it actually is a HashMap, and stores values as Box<dyn Any>. The
/// important part of key's implementing [HasArrayType] is that the associated type
/// of that trait is used to downcast the box to the expected value.
#[derive(Debug, Default)]
pub struct Gradients<D: DeviceStorage> {
    gradient_by_id: HashMap<UniqueId, Box<dyn std::any::Any>>,
    device: PhantomData<*const D>,
}

impl<D: DeviceStorage> Gradients<D> {
    pub(crate) fn get_or_alloc_mut<T>(
        &mut self,
        t: &T,
    ) -> Result<&mut D::Storage<T::Shape, T::Dtype>, D::Err>
    where
        T: HasUniqueId + AllocGrad<D>,
    {
        self.try_alloc_for(t)?;
        Ok(self.get_mut(t))
    }

    pub(crate) fn try_alloc_for<T>(&mut self, t: &T) -> Result<(), D::Err>
    where
        T: HasUniqueId + AllocGrad<D>,
    {
        if !self.gradient_by_id.contains_key(t.id()) {
            let grad = t.try_alloc_grad()?;
            self.gradient_by_id.insert(*t.id(), Box::new(grad));
        }
        Ok(())
    }

    /// Removes and returns the data associated with `t.id()`.
    ///
    /// **Panics** if data associated with `t` is not found. This indicates an unrecoverable bug.
    ///
    /// Example usage:
    /// ```
    /// # use dfdx::{prelude::*, gradients::*};
    /// let t = Tensor1D::new([1.0, 2.0, 3.0]);
    /// let mut gradients: Gradients = Default::default();
    /// *gradients.mut_gradient(&t) = [-4.0, 5.0, -6.0];
    /// assert_eq!(gradients.remove(&t).expect("").as_ref(), &[-4.0, 5.0, -6.0]);
    /// ```
    pub fn remove<T: HasUniqueId + HasShape + HasDtype>(
        &mut self,
        t: &T,
    ) -> Option<D::Storage<T::Shape, T::Dtype>> {
        self.gradient_by_id
            .remove_entry(t.id())
            .map(|e| *e.1.downcast().unwrap())
    }

    /// Returns a mutable reference to the data associated with `t`.
    ///
    /// If no data is associated with `t`, then [AllocateZeros::zeros] is called
    /// to allocate the data.
    ///
    /// Example usage:
    /// ```
    /// # use dfdx::{prelude::*, gradients::*};
    /// let t = Tensor1D::new([1.0, 2.0, 3.0]);
    /// let mut gradients: Gradients = Default::default();
    /// let g: &mut [f32; 3] = gradients.get_mut(&t);
    /// assert_eq!(g, &mut [0.0, 0.0, 0.0]);
    /// g[0] = 1.0;
    /// assert_eq!(gradients.ref_gradient(&t), &[1.0, 0.0, 0.0]);
    /// ```
    pub fn get_mut<T>(&mut self, t: &T) -> &mut D::Storage<T::Shape, T::Dtype>
    where
        T: HasUniqueId + HasDtype + HasShape,
    {
        self.gradient_by_id
            .get_mut(t.id())
            .unwrap()
            .downcast_mut()
            .unwrap()
    }

    /// Returns a reference to the data associated with `t`.
    ///
    /// # Panics
    ///
    /// If no data is associated with `t` yet, this will panic due to an unwrap()
    /// on a .get() to the underlying hashmap.
    ///
    /// # Example usage:
    /// ```
    /// # use dfdx::{prelude::*, gradients::*};
    /// let t = Tensor1D::new([1.0, 2.0, 3.0]);
    /// let mut gradients: Gradients = Default::default();
    /// gradients.mut_gradient(&t);
    /// assert_eq!(gradients.grad(&t), &[0.0, 0.0, 0.0]);
    /// ```
    pub fn get<T: HasUniqueId + HasDtype + HasShape>(
        &self,
        t: &T,
    ) -> &D::Storage<T::Shape, T::Dtype> {
        self.gradient_by_id
            .get(t.id())
            .unwrap()
            .as_ref()
            .downcast_ref()
            .unwrap()
    }

    /// Borrows a pair of a gradients `(&mut L, &R)`.
    /// `l` is the gradient to update, and `r` is the gradient to backprop.
    ///
    /// **Panics** if `l` and `r` have the same id.
    ///
    /// Examples:
    /// ```rust
    /// # use dfdx::{prelude::*, gradients::*};
    /// let a = Tensor1D::new([1.0, 2.0, 3.0]);
    /// let b: Tensor1D<5> = Tensor1D::zeros();
    /// let mut gradients: Gradients = Default::default();
    /// *gradients.mut_gradient(&a) = [-4.0, 5.0, -6.0];
    /// *gradients.mut_gradient(&b) = [1.0, 2.0, 3.0, 4.0, 5.0];
    /// let (g_a, g_b) = gradients.mut_and_ref(&a, &b);
    /// assert_eq!(g_a, &mut [-4.0, 5.0, -6.0]);
    /// assert_eq!(g_b, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// ```
    pub fn mut_and_ref<L, R>(
        &mut self,
        l: &L,
        r: &R,
    ) -> (
        &mut D::Storage<L::Shape, L::Dtype>,
        &D::Storage<R::Shape, R::Dtype>,
    )
    where
        L: HasUniqueId + HasShape + HasDtype,
        R: HasUniqueId + HasShape + HasDtype,
    {
        assert_ne!(l.id(), r.id());
        let l_ptr = self.get_mut(l) as *mut _;
        let r_ptr = self.get(r) as *const _;
        let l_ref = unsafe { &mut *l_ptr };
        let r_ref = unsafe { &*r_ptr };
        (l_ref, r_ref)
    }

    pub fn muts_and_ref<L1, L2, R>(
        &mut self,
        l1: &L1,
        l2: &L2,
        r: &R,
    ) -> (
        &mut D::Storage<L1::Shape, L1::Dtype>,
        &mut D::Storage<L2::Shape, L2::Dtype>,
        &D::Storage<R::Shape, R::Dtype>,
    )
    where
        L1: HasUniqueId + HasShape + HasDtype,
        L2: HasUniqueId + HasShape + HasDtype,
        R: HasUniqueId + HasShape + HasDtype,
    {
        assert_ne!(l1.id(), l2.id());
        assert_ne!(l1.id(), r.id());
        assert_ne!(l2.id(), r.id());
        let l1_ptr = self.get_mut(l1) as *mut _;
        let l2_ptr = self.get_mut(l2) as *mut _;
        let r_ptr = self.get(r) as *const _;
        let l1_ref = unsafe { &mut *l1_ptr };
        let l2_ref = unsafe { &mut *l2_ptr };
        let r_ref = unsafe { &*r_ptr };
        (l1_ref, l2_ref, r_ref)
    }
}

/// Records gradient computations to execute later.
///
/// The only two things you can do with this are:
/// 1. Adding an operation (an operation is a FnOnce that acts on &mut [Gradients])
/// 2. Executing all the operations to produce [Gradients]
///
/// The reason for this design, which forces users to specify gradient computations, as opposed to having
/// a fixed set of *kinds* of computations are these:
/// 1. Different tensor sizes. The tensors size information would have to be stored inside the operation somehow.
///     Instead, the operation themselves must query with a sized tensor, so sizes are still known at compile time instead of dynamically.
/// 2. Slightly different operations. It'd have to support broadcasting operations, etc which can get needlessly complex.
/// 3. Optimizations are harder. With operations having control over everything, they can be optimized by hand separately.
///
/// An example for how these two are used is the following from the negate operation (ie. multiply all values by -1).
///
/// ```ignore
/// tape.add_backward_op(move |grads| {
///     let (t_grad, result_grad) = grads.mut_and_ref(&t, &_result);
///     // addmul_assign is equivalent to: t_grad += t.data() * result_grad;
///     T::Device::addmul(t_grad, t.data(), result_grad);
/// });
/// ```
///
/// This is implementing the chain rule, which is normally defined as `gradient(t) += deriv * gradient(result)` with
/// the following optimizations:
/// 1. instead of allocating new data for the derivative (which is just -1 everywhere), we can reuse the `t` tensor since the negate
///     function owns it.
/// 2. We can combine computing the derivative and multiplying by the `gradient(result)` by just setting `t` to `-gradient(result)`
///
/// This would not be possible if these chain rule operations were inside of GradientTape!
#[allow(clippy::type_complexity)]
pub struct GradientTape<D: DeviceStorage> {
    operations: Vec<Box<dyn FnOnce(&mut Gradients<D>) -> Result<(), D::Err>>>,
    gradients: Gradients<D>,
}

impl<D: DeviceStorage> Default for GradientTape<D> {
    fn default() -> Self {
        Self {
            operations: Vec::new(),
            gradients: Default::default(),
        }
    }
}

impl<D: DeviceStorage> std::fmt::Debug for GradientTape<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientTape")
            .field("num_operations", &self.operations.len())
            .finish()
    }
}

impl<D: DeviceStorage> GradientTape<D> {
    /// Add an operation to be executed later. Implementation is all left to the caller,
    /// but the operation should likely call [Gradients::ref_gradient] and [Gradients::mut_gradient].
    ///
    /// # Arguments
    /// * `operation` - A FnOnce that acts on [Gradients].
    ///
    /// See src/tensor_ops for implementation examples.
    pub(crate) fn add_backward_op<F: 'static + FnOnce(&mut Gradients<D>) -> Result<(), D::Err>>(
        &mut self,
        operation: F,
    ) {
        self.operations.push(Box::new(operation));
    }

    /// Compute the [Gradients]! This just runs all the operations on a new [Gradients] struct.
    ///
    /// Note that this method takes ownership of self, so it can't be called twice!
    pub fn execute(mut self) -> Result<Gradients<D>, D::Err> {
        for operation in self.operations.drain(..).rev() {
            (operation)(&mut self.gradients)?;
        }
        Ok(self.gradients)
    }

    /// Moves all the operations from `other` into self. Leaves `other` empty.
    pub fn append(&mut self, other: &mut Self) {
        self.gradients
            .gradient_by_id
            .extend(other.gradients.gradient_by_id.drain());
        self.operations.append(&mut other.operations);
    }
}

/// Contains a boxed [GradientTape]. When [Tape::add_backward_op] is called,
/// this function passes the operation directly to [GradientTape].
#[derive(Debug, Default)]
pub struct OwnedTape<D: DeviceStorage>(pub(crate) Box<GradientTape<D>>);

/// Contains nothing. When [Tape::add_backward_op] is called, this struct does nothing.
#[derive(Default, Debug, Clone, Copy)]
pub struct NoneTape;

/// Something that can add a gradient operation to [GradientTape].
pub trait Tape<D: DeviceStorage>: Default + Merge<Self> + Merge<NoneTape> {
    /// Whether this object currently owns the [GradientTape]. This is known at compile time.
    const OWNS_TAPE: bool;
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients<D>) -> Result<(), D::Err>>(
        &mut self,
        operation: F,
    );
    fn try_alloc_grad<T: HasUniqueId + AllocGrad<D>>(&mut self, t: &T) -> Result<(), D::Err>;
}

impl<D: DeviceStorage> Tape<D> for OwnedTape<D> {
    const OWNS_TAPE: bool = true;
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients<D>) -> Result<(), D::Err>>(
        &mut self,
        operation: F,
    ) {
        self.0.add_backward_op(operation)
    }
    fn try_alloc_grad<T: HasUniqueId + AllocGrad<D>>(&mut self, t: &T) -> Result<(), D::Err> {
        self.0.gradients.try_alloc_for(t)
    }
}

impl<D: DeviceStorage> Tape<D> for NoneTape {
    const OWNS_TAPE: bool = false;
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients<D>) -> Result<(), D::Err>>(
        &mut self,
        _: F,
    ) {
    }
    fn try_alloc_grad<T: HasUniqueId + AllocGrad<D>>(&mut self, _: &T) -> Result<(), D::Err> {
        Ok(())
    }
}

pub trait Merge<T: ?Sized> {
    /// Merges `T` into `self`
    fn merge(self, other: T) -> Self;
}

impl Merge<NoneTape> for NoneTape {
    fn merge(self, _: NoneTape) -> Self {
        self
    }
}

impl<D: DeviceStorage> Merge<NoneTape> for OwnedTape<D> {
    fn merge(self, _: NoneTape) -> Self {
        self
    }
}

impl<D: DeviceStorage> Merge<OwnedTape<D>> for OwnedTape<D> {
    fn merge(mut self, mut other: Self) -> Self {
        self.0.append(other.0.as_mut());
        self
    }
}
