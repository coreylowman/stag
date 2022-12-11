//! Ergonomics & safety focused deep learning in Rust. Main features include:
//! 1. Const generic tensor library with tensors up to 4d!
//! 2. A large library of tensor operations (matrix multiplication, arithmetic, activation functions, etc).
//! 3. Safe & easy to use neural network building blocks.
//! 4. Standard deep learning optimizers such as Sgd and Adam.
//! 5. Reverse mode auto differentiation implementation.
//! 6. Serialization to/from `.npy` and `.npz` for transferring models to/from python.
//!
//! # A quick tutorial
//!
//! 1. [crate::tensor::Tensor]s can be created with normal rust arrays. See [crate::tensor].
//! ```rust
//! # use dfdx::prelude::*;
//! let x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let y: Tensor2D<2, 3> = TensorCreator::ones();
//! ```
//!
//! 2. Neural networks are built with types. Tuples are sequential models. See [crate::nn].
//! ```rust
//! # use dfdx::prelude::*;
//! type Mlp = (
//!     Linear<5, 3>,
//!     ReLU,
//!     Linear<3, 2>,
//! );
//! ```
//!
//! 3. Instantiate models with [Default], and randomize with [crate::nn::ResetParams]
//! ```rust
//! # use dfdx::prelude::*;
//! # let mut rng = rand::thread_rng();
//! let mut mlp: Linear<5, 2> = Default::default();
//! mlp.reset_params(&mut rng);
//! ```
//!
//! 4. Pass data through networks with [crate::nn::Module]
//! ```rust
//! # use dfdx::prelude::*;
//! let mut mlp: Linear<5, 2> = Default::default();
//! let x = Tensor1D::zeros(); // compiler knows that x is a `Tensor1D<5>`
//! let y = mlp.forward(x); // compiler knows that `y` must be `Tensor1D<2>`
//! ```
//!
//! 5. Trace gradients using [crate::tensor::trace()]
//! ```rust
//! # use dfdx::prelude::*;
//! # let mut rng = rand::thread_rng();
//! # let model: Linear<10, 5> = Default::default();
//! # let y_true: Tensor1D<5> = Tensor1D::randn(&mut rng).softmax();
//! // tensors default to not having a tape
//! let x: Tensor1D<10, NoneTape> = TensorCreator::zeros();
//!
//! // `.trace()` clones `x` and inserts a gradient tape.
//! let x_t: Tensor1D<10, OwnedTape> = x.trace();
//!
//! // The tape from the input is moved through the network during .forward().
//! let y: Tensor1D<5, NoneTape> = model.forward(x);
//! let y_t: Tensor1D<5, OwnedTape> = model.forward(x_t);
//! ```
//!
//! 6. Compute gradients with [crate::tensor_ops::backward()]. See [crate::tensor_ops].
//! ```rust
//! # use dfdx::{prelude::*, gradients::Gradients};
//! # let mut rng = rand::thread_rng();
//! # let model: Linear<10, 5> = Default::default();
//! # let y_true: Tensor1D<5> = Tensor1D::randn(&mut rng).softmax();
//! # let y: Tensor1D<5, OwnedTape> = model.forward(Tensor1D::zeros().trace());
//! // compute cross entropy loss
//! let loss: Tensor0D<OwnedTape> = cross_entropy_with_logits_loss(y, y_true);
//!
//! // call `backward()` to compute gradients. The tensor *must* have `OwnedTape`!
//! let gradients: Gradients = loss.backward();
//! ```
//! 7. Use an optimizer from [crate::optim] to optimize your network!
//! ```rust
//! # use dfdx::{prelude::*, gradients::Gradients};
//! # let mut rng = rand::thread_rng();
//! # let mut model: Linear<10, 5> = Default::default();
//! # let x: Tensor1D<10> = Tensor1D::zeros();
//! # let y_true: Tensor1D<5> = Tensor1D::randn(&mut rng).softmax();
//! # let y: Tensor1D<5, OwnedTape> = model.forward(x.trace());
//! # let loss = cross_entropy_with_logits_loss(y, y_true);
//! # let gradients: Gradients = loss.backward();
//! // Use stochastic gradient descent (Sgd), with a learning rate of 1e-2, and 0.9 momentum.
//! let mut opt = Sgd::new(SgdConfig {
//!     lr: 1e-2,
//!     momentum: Some(Momentum::Classic(0.9)),
//!     weight_decay: None,
//! });
//!
//! // pass the gradients & the model into the optimizer's update method
//! opt.update(&mut model, gradients);
//! ```

#![no_std]
#![allow(incomplete_features)]
#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

extern crate alloc;
extern crate no_std_compat as std;

pub mod data;
pub mod feature_flags;
pub mod gradients;
pub mod losses;
pub mod nn;
pub mod optim;
pub mod shapes;
pub mod tensor;
pub mod tensor_ops;
pub mod unique_id;

/// Contains all public exports.
pub mod prelude {
    pub use crate::gradients::{NoneTape, OwnedTape};
    pub use crate::nn::*;
    pub use crate::optim::prelude::*;
    pub use crate::shapes::*;
    pub use crate::tensor::*;
    pub use crate::tensor_ops::*;
}

/// Sets a CPU `sse` flag to flush denormal floating point numbers to zero. The opposite of this is [keep_denormals()].
///
/// Some resources:
/// 1. [Effects of Flush-To-Zero mode](https://developer.arm.com/documentation/dui0473/c/neon-and-vfp-programming/the-effects-of-using-flush-to-zero-mode?lang=en)
/// 2. [When to use Flush-To-Zero mode](https://developer.arm.com/documentation/dui0473/c/neon-and-vfp-programming/when-to-use-flush-to-zero-mode?lang=en)
pub fn flush_denormals_to_zero() {
    #[cfg(all(target_arch = "x86", target_feature = "sse"))]
    {
        use std::arch::x86::{_MM_FLUSH_ZERO_ON, _MM_SET_FLUSH_ZERO_MODE};
        unsafe { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON) }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    {
        use std::arch::x86_64::{_MM_FLUSH_ZERO_ON, _MM_SET_FLUSH_ZERO_MODE};
        unsafe { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON) }
    }
}

/// Sets a CPU flag to keep denormal floating point numbers. The opposite of this is [flush_denormals_to_zero()].
///
/// Some resources:
/// 1. [Effects of Flush-To-Zero mode](https://developer.arm.com/documentation/dui0473/c/neon-and-vfp-programming/the-effects-of-using-flush-to-zero-mode?lang=en)
/// 2. [When to use Flush-To-Zero mode](https://developer.arm.com/documentation/dui0473/c/neon-and-vfp-programming/when-to-use-flush-to-zero-mode?lang=en)
pub fn keep_denormals() {
    #[cfg(all(target_arch = "x86", target_feature = "sse"))]
    {
        use std::arch::x86::{_MM_FLUSH_ZERO_OFF, _MM_SET_FLUSH_ZERO_MODE};
        unsafe { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF) }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse"))]
    {
        use std::arch::x86_64::{_MM_FLUSH_ZERO_OFF, _MM_SET_FLUSH_ZERO_MODE};
        unsafe { _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF) }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    const TOLERANCE: f32 = 1e-6;

    pub type TestDevice = crate::tensor::Cpu;

    macro_rules! build_test_device {
        () => {{
            use crate::tensor::Cpu;
            Cpu::default()
        }};
        ($Seed:expr) => {{
            use crate::tensor::Cpu;
            Cpu::seed_from_u64($Seed)
        }};
    }

    pub(crate) use build_test_device;

    pub trait AssertClose {
        fn get_far_pair(&self, rhs: &Self, tolerance: f32) -> Option<(f32, f32)>;
        fn assert_close(&self, rhs: &Self, tolerance: f32)
        where
            Self: std::fmt::Debug,
        {
            if let Some((l, r)) = self.get_far_pair(rhs, tolerance) {
                panic!("lhs != rhs | {l} != {r}\n\n{self:?}\n\n{rhs:?}");
            }
        }
    }

    impl<const M: usize> AssertClose for [f32; M] {
        fn get_far_pair(&self, rhs: &Self, tolerance: f32) -> Option<(f32, f32)> {
            for (l, r) in self.iter().zip(rhs.iter()) {
                if (l - r).abs() > tolerance {
                    return Some((*l, *r));
                }
            }
            None
        }
    }

    impl<T: AssertClose, const M: usize> AssertClose for [T; M] {
        fn get_far_pair(&self, rhs: &Self, tolerance: f32) -> Option<(f32, f32)> {
            for (l, r) in self.iter().zip(rhs.iter()) {
                if let Some(pair) = l.get_far_pair(r, tolerance) {
                    return Some(pair);
                }
            }
            None
        }
    }

    pub fn assert_close<T: AssertClose + std::fmt::Debug>(a: &T, b: &T) {
        a.assert_close(b, TOLERANCE);
    }
}

/// Used to assert things about const generics
pub struct Assert<const C: bool>;
pub trait ConstTrue {}
impl ConstTrue for Assert<true> {}
