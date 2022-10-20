//! Implementations of [GradientTape] and generic Nd array containers via [Gradients].

use std::collections::HashMap;
use std::{boxed::Box, vec::Vec};

use crate::arrays::{HasArrayData, HasArrayType};
use crate::devices::{AllocateZeros, HasDevice};
use crate::unique_id::{HasUniqueId, UniqueId};

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
#[derive(Default)]
#[allow(clippy::type_complexity)]
pub struct GradientTape {
    operations: Vec<Box<dyn FnOnce(&mut Gradients)>>,
}

impl std::fmt::Debug for GradientTape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientTape")
            .field("num_operations", &self.operations.len())
            .finish()
    }
}

impl GradientTape {
    /// Add an operation to be executed later. Implementation is all left to the caller,
    /// but the operation should likely call [Gradients::ref_gradient] and [Gradients::mut_gradient].
    ///
    /// # Arguments
    /// * `operation` - A FnOnce that acts on [Gradients].
    ///
    /// See src/tensor_ops for implementation examples.
    pub(crate) fn add_backward_op<F: 'static + FnOnce(&mut Gradients)>(&mut self, operation: F) {
        self.operations.push(Box::new(operation));
    }

    /// Compute the [Gradients]! This just runs all the operations on a new [Gradients] struct.
    ///
    /// Note that this method takes ownership of self, so it can't be called twice!
    pub fn execute(mut self) -> Gradients {
        let mut gradients: Gradients = Default::default();
        for operation in self.operations.drain(..).rev() {
            (operation)(&mut gradients);
        }
        gradients
    }

    /// Moves all the operations from `other` into self. Leaves `other` empty.
    pub fn append(&mut self, other: &mut Self) {
        self.operations.append(&mut other.operations);
    }
}

/// Contains a boxed [GradientTape]. When [Tape::add_backward_op] is called,
/// this function passes the operation directly to [GradientTape].
#[derive(Default, Debug)]
pub struct OwnedTape(pub(crate) Box<GradientTape>);

/// Contains nothing. When [Tape::add_backward_op] is called, this struct does nothing.
#[derive(Default, Debug, Clone, Copy)]
pub struct NoneTape;

/// Something that can add a gradient operation to [GradientTape].
pub trait Tape {
    /// Whether this object currently owns the [GradientTape]. This is known at compile time.
    const OWNS_TAPE: bool;
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients)>(&mut self, operation: F);
}

impl Tape for OwnedTape {
    const OWNS_TAPE: bool = true;
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients)>(&mut self, operation: F) {
        self.0.add_backward_op(operation)
    }
}

impl Tape for NoneTape {
    const OWNS_TAPE: bool = false;
    fn add_backward_op<F: 'static + FnOnce(&mut Gradients)>(&mut self, _operation: F) {}
}

impl<T: Tape> Merge<NoneTape> for T {
    type Output = T;
    fn merge(self, _: NoneTape) -> Self::Output {
        self
    }
}

pub trait Merge<T> {
    type Output: Tape;

    /// Merges `T` into `self`
    fn merge(self, other: T) -> Self::Output;
}

impl Merge<OwnedTape> for NoneTape {
    type Output = OwnedTape;
    fn merge(self, other: OwnedTape) -> Self::Output {
        other
    }
}

impl Merge<Self> for OwnedTape {
    type Output = Self;
    fn merge(mut self, mut other: Self) -> Self::Output {
        self.0.append(other.0.as_mut());
        self
    }
}

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
pub struct Gradients {
    gradient_by_id: HashMap<UniqueId, Box<dyn std::any::Any>>,
}

impl Gradients {
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
    pub fn mut_and_ref<L, R>(&mut self, l: &L, r: &R) -> (&mut L::Array, &R::Array)
    where
        L: HasUniqueId + HasArrayType + HasDevice,
        R: HasUniqueId + HasArrayType,
    {
        assert_ne!(l.id(), r.id());
        let l_ptr = self.mut_gradient(l) as *mut L::Array;
        let r_ptr = self.ref_gradient(r) as *const R::Array;
        let l_ref = unsafe { &mut *l_ptr };
        let r_ref = unsafe { &*r_ptr };
        (l_ref, r_ref)
    }

    pub fn muts_and_ref<L1, L2, L3, R>(
        &mut self,
        l1: &L1,
        l2: &L2,
        l3: &L3,
        r: &R,
    ) -> (&mut L1::Array, &mut L2::Array, &mut L3::Array, &R::Array)
    where
        L1: HasUniqueId + HasArrayType + HasDevice,
        L2: HasUniqueId + HasArrayType + HasDevice,
        L3: HasUniqueId + HasArrayType + HasDevice,
        R: HasUniqueId + HasArrayType,
    {
        // assert_ne!(l1.id(), r.id());
        let l1_ptr = self.mut_gradient(l1) as *mut L1::Array;
        let l2_ptr = self.mut_gradient(l2) as *mut L2::Array;
        let l3_ptr = self.mut_gradient(l3) as *mut L3::Array;
        let r_ptr = self.ref_gradient(r) as *const R::Array;
        let l1_ref = unsafe { &mut *l1_ptr };
        let l2_ref = unsafe { &mut *l2_ptr };
        let l3_ref = unsafe { &mut *l3_ptr };
        let r_ref = unsafe { &*r_ptr };
        (l1_ref, l2_ref, l3_ref, r_ref)
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
    pub fn remove<T: HasUniqueId + HasArrayType>(&mut self, t: &T) -> Option<Box<T::Array>> {
        self.gradient_by_id
            .remove_entry(t.id())
            .map(|e| e.1.downcast().unwrap())
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
    /// let g: &mut [f32; 3] = gradients.mut_gradient(&t);
    /// assert_eq!(g, &mut [0.0, 0.0, 0.0]);
    /// g[0] = 1.0;
    /// assert_eq!(gradients.ref_gradient(&t), &[1.0, 0.0, 0.0]);
    /// ```
    pub fn mut_gradient<T: HasUniqueId + HasArrayType + HasDevice>(
        &mut self,
        t: &T,
    ) -> &mut T::Array {
        self.gradient_by_id
            .entry(*t.id())
            .or_insert_with(|| T::Device::zeros::<T::Array>())
            .as_mut()
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
    /// assert_eq!(gradients.ref_gradient(&t), &[0.0, 0.0, 0.0]);
    /// ```
    pub fn ref_gradient<T: HasUniqueId + HasArrayType>(&self, t: &T) -> &T::Array {
        self.gradient_by_id
            .get(t.id())
            .unwrap()
            .as_ref()
            .downcast_ref()
            .unwrap()
    }
}

/// Represents something that can return a gradient for a given key.
///
/// This is very similar to what [Gradients] does, however the intention
/// is that any this object be passed to [CanUpdateWithGradients].
///
/// [Gradients] does **not** implement this, so you *have* to go through
/// an optimizer to update a [CanUpdateWithGradients]. Although it very easily
/// could.
///
/// See [crate::optim::Sgd] and [crate::optim::Adam] for examples on implementing this.
pub trait GradientProvider {
    /// Retrieves the data associated with `p` if there is any.
    /// This can modify `self`, for instance if velocities are calculated
    /// based on the associated data!
    fn gradient<P>(&mut self, p: &P) -> Option<Box<P::Array>>
    where
        P: HasUniqueId + HasArrayType<Dtype = f32> + HasDevice + HasArrayData;
}

/// Represents something that can be updated with [GradientProvider].
///
/// Most implementations of this trait will have sub structs that also
/// implement [CanUpdateWithGradients].
pub trait CanUpdateWithGradients {
    /// Updates self given the [GradientProvider]. When any parameters that
    /// are NOT present in `G`, then this function should
    /// add the tensor's [UniqueId] to [UnusedTensors].
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors);
}

/// Holds [UniqueId] of tensors that were missing gradients during
/// [CanUpdateWithGradients::update()], and therefore are unused
#[derive(Debug, Default)]
pub struct UnusedTensors {
    pub ids: Vec<UniqueId>,
}

impl UnusedTensors {
    /// Adds a single unnammed parameter
    pub fn add<T: HasUniqueId>(&mut self, t: &T) {
        self.ids.push(*t.id());
    }

    /// Returns `true` if there are no missing gradients present
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::Cpu;
    use crate::unique_id::unique_id;

    struct Tensor {
        id: UniqueId,
    }

    impl HasUniqueId for Tensor {
        fn id(&self) -> &UniqueId {
            &self.id
        }
    }

    impl HasArrayType for Tensor {
        type Array = [f32; 5];
        type Dtype = f32;
    }

    impl HasDevice for Tensor {
        type Device = Cpu;
    }

    #[test]
    fn test_backward() {
        let id = unique_id();
        let t1: Tensor = Tensor { id };
        let _t1: Tensor = Tensor { id };

        let mut tape = GradientTape::default();
        tape.add_backward_op(move |g| {
            let t_grad = g.mut_gradient(&_t1);
            for x in t_grad.iter_mut() {
                *x += 1.0;
            }
        });
        let g = tape.execute();
        assert_eq!(g.ref_gradient(&t1), &[1.0; 5]);
    }
}
