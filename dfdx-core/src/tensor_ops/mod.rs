//! Operations on tensors like [relu()], [matmul()], [softmax()], and more.
//!
//! # Generic function and struct methods
//!
//! All functionality is provided in two ways.
//! 1. The generic standalone function that takes a generic parameter. e.g. [relu()].
//! 2. The struct method for tensor structs. e.g. [crate::tensor::Tensor::relu()].
//!
//! The functions are all just pass throughs to the tensor methods.
//!
//! # Fallibility
//!
//! All tensor methods also have a `try_*` variant, like [crate::tensor::Tensor::relu()] and [crate::tensor::Tensor::try_relu()].
//!
//! These methods return a [Result], where the error in most cases indicates an allocation
//! error.
//!
//! # Axes/Dimensions for broadcasting/reductions/selecting
//!
//! For the following sections, some traits/functions utilizing const `isize` to determine
//! the axis to apply the transformation to.
//!
//! Here are the valid axes for each tensor:
//! 1. 0d tensor: `Axis<0>`
//! 2. 1d tensor: `Axis<0>`
//! 3. 2d tensor: `Axis<0>`, `Axis<1>`
//! 4. 3d tensor: `Axis<0>`, `Axis<1>`, `Axis<2>`,
//! 5. 4d tensor: `Axis<0>`, `Axis<1>`, `Axis<2>`, `Axis<3>`
//! 6. etc.
//!
//! To specify multiple axes you can use `Axes2`, `Axes3`, and `Axes4`
//!
//! # Reductions
//!
//! There are a number of methods that reduce 1 or more axes.Anything that can be reduced can also
//! be broadcasted back to the original shape using [BroadcastTo].
//!
//! Each axis reducing function has two generic parameters:
//! 1. The target shape
//! 2. The axes to reduce along
//! **You only need to specify one of these!** Generally it is better practice to specify the
//! target shape, unless it is ambiguous in which case you should specify the axes.
//!
//! For example:
//! ```rust
//! # use dfdx_core::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t: Tensor<Rank3<2, 4, 6>, f32, _> = dev.zeros();
//! // shape version
//! let _ = t.clone().sum::<Rank1<4>, _>();
//! // axes version
//! let _ = t.clone().sum::<_, Axes2<0, 2>>();
//! // typed version
//! let _: Tensor<Rank1<4>, _, _> = t.clone().sum();
//! ```
//!
//! Complete list of reductions:
//!
//! - [MaxTo]
//! - [MeanTo]
//! - [MinTo]
//! - [SumTo]
//! - [VarTo]
//! - [StddevTo]
//! - [LogSumExpTo]
//!
//! # Broadcasts
//!
//! Broadcasting tensors is provided through the [BroadcastTo] trait. Similar to reductions
//! there are two generic parameters to broadcast:
//! 1. (Required) The target shape
//! 2. (usually optional) The axes *of the result type* to broadcast
//! You'll only need to specify axes if the shape makes the broadcasts ambiguous.
//!
//! For example:
//! ```rust
//! # use dfdx_core::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t: Tensor<Rank1<4>, f32, _> = dev.zeros();
//! // shape version
//! let _ = t.clone().broadcast::<Rank3<2, 4, 6>, _>();
//! // typed version
//! let _: Tensor<Rank3<2, 4, 6>, _, _> = t.clone().broadcast();
//! ```
//!
//! Rust can also infer the output type if you use it in another operation:
//! ```rust
//! # use dfdx_core::prelude::*;
//! # let dev: Cpu = Default::default();
//! let big: Tensor<Rank2<2, 5>, f32, _> = dev.zeros();
//! let small: Tensor<Rank1<5>, f32, _> = dev.zeros();
//! let _ = big + small.broadcast();
//! ```
//!
//! # Permutes
//!
//! Permuting has an identical interface to broadcasts/reductions:
//! ```rust
//! # use dfdx_core::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t: Tensor<Rank3<2, 3, 4>, f32, _> = dev.zeros();
//! // shape version
//! let _ = t.clone().permute::<Rank3<3, 4, 2>, _>();
//! // axes version
//! let _ = t.clone().permute::<_, Axes3<1, 2, 0>>();
//! ```
//!
//! # Indexing using select and gather
//!
//! Two traits provide indexing capability [SelectTo] and [GatherTo]. The difference is:
//! 1. [SelectTo::select] allows you to select a single value
//! 2. [GatherTo::gather] allows you select multiple values from the same axis.
//!
//! For example you can select from the 0th axis like so:
//! ```rust
//! # use dfdx_core::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let r: Tensor<Rank1<3>, f32, _> = t.select(dev.tensor(1));
//! assert_eq!(r.array(), [4.0, 5.0, 6.0]);
//! ```
//!
//! Or you can gather from the 0th axis to select multiple entries:
//! ```rust
//! # use dfdx_core::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let r: Tensor<Rank2<3, 3>, f32, _> = t.gather(dev.tensor([1, 1, 0]));
//! assert_eq!(r.array(), [
//!     [4.0, 5.0, 6.0],
//!     [4.0, 5.0, 6.0],
//!     [1.0, 2.0, 3.0],
//! ]);
//! ```
//!
//! To select from anything after the 0th axis, you need a multi-dimensional
//! axis. See [GatherTo] and [SelectTo] docstrings for examples of this.
//!
//! But you can use [BroadcastTo] to make this easy! In this example we select
//! the same index from the 1st axis of a tensor:
//! ```rust
//! # use dfdx_core::prelude::*;
//! # let dev: Cpu = Default::default();
//! let t = dev.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let r = t.select::<Rank1<2>, _>(dev.tensor(1).broadcast());
//! assert_eq!(r.array(), [2.0, 5.0]);
//! ```

mod utilities;
pub use utilities::*;

mod abs;
mod accurate_gelu;
mod adam;
mod add;
mod attention_reshape;
pub(crate) mod axpy;
mod bce;
mod boolean;
mod broadcast_to;
mod choose;
mod clamp;
mod cmp;
mod concat;
mod concat_along;
mod concat_shape_along;
mod concat_tensor_along;
mod cos;
mod div;
mod dropout;
mod exp;
mod fast_gelu;
mod huber_error;
mod ln;
mod log_softmax;
mod logsumexp_to;
mod matmul;
mod max_to;
mod maximum;
mod mean_to;
mod min_to;
mod minimum;
mod mul;
mod nans_to;
mod negate;
mod normalize;
pub(super) mod optim;
mod permute_to;
mod pow;
mod prelu;
mod realize_to;
mod recip;
mod relu;
mod reshape_to;
mod rmsprop;
mod roll;
mod select_and_gather;
mod sgd;
mod sigmoid;
mod sin;
mod slice;
mod softmax;
mod sqrt;
mod square;
mod stack;
mod stddev_to;
mod sub;
mod sum_to;
mod tanh;
mod to_dtype;
mod tri;
mod upscale2d;
mod var_to;

pub use abs::abs;
pub use accurate_gelu::accurate_gelu;
pub use adam::AdamConfig;
pub use add::{add, TryAdd};
pub use attention_reshape::TryAttentionReshape;
pub use axpy::axpy;
pub use bce::bce_with_logits;
pub use boolean::{bool_and, bool_not, bool_or, bool_xor};
pub use broadcast_to::BroadcastTo;
pub use choose::ChooseFrom;
pub use clamp::clamp;
pub use cmp::{eq, ge, gt, le, lt, ne, TryEq, TryGe, TryGt, TryLe, TryLt, TryNe};
#[allow(deprecated)]
pub use concat::TryConcat;
#[allow(deprecated)]
pub use concat_along::TryConcatAlong;
pub use concat_shape_along::TryConcatShapeAlong;
pub use concat_tensor_along::TryConcatTensorAlong;
pub use cos::cos;
pub use div::{div, TryDiv};
pub use dropout::dropout;
pub use exp::exp;
pub use fast_gelu::fast_gelu;
#[allow(deprecated)]
pub use fast_gelu::gelu;
pub use huber_error::huber_error;
pub use ln::ln;
pub use log_softmax::log_softmax;
pub use logsumexp_to::LogSumExpTo;
pub use matmul::{matmul, TryMatMul};
pub use max_to::MaxTo;
pub use maximum::maximum;
pub use mean_to::MeanTo;
pub use min_to::MinTo;
pub use minimum::minimum;
pub use mul::{mul, TryMul};
pub use nans_to::nans_to;
pub use negate::negate;
pub use normalize::normalize;
pub use optim::*;
pub use permute_to::PermuteTo;
pub use pow::{powf, powi};
pub use prelu::{leakyrelu, prelu, TryPReLU};
pub use realize_to::RealizeTo;
pub use recip::recip;
pub use relu::relu;
pub use reshape_to::ReshapeTo;
pub use rmsprop::RMSpropConfig;
pub use roll::Roll;
pub use select_and_gather::{GatherTo, SelectTo};
pub use sgd::SgdConfig;
pub use sigmoid::sigmoid;
pub use sin::sin;
pub use slice::slice;
pub use softmax::softmax;
pub use sqrt::sqrt;
pub use square::square;
pub use stack::{AddDim, TryStack};
pub use stddev_to::StddevTo;
pub use sub::{sub, TrySub};
pub use sum_to::SumTo;
pub use tanh::tanh;
pub use to_dtype::{to_dtype, ToDtypeKernel};
pub use tri::{lower_tri, upper_tri};
pub use upscale2d::{
    Bilinear, GenericUpscale2D, NearestNeighbor, TryUpscale2D, Upscale2DKernel, UpscaleMethod,
};
pub use var_to::VarTo;

mod conv1d;
pub use conv1d::TryConv1D;

mod conv2d;
pub use conv2d::TryConv2D;

#[cfg(feature = "nightly")]
mod convtrans2d;
#[cfg(feature = "nightly")]
pub use convtrans2d::TryConvTrans2D;

mod pool2d;
pub use pool2d::{Pool2DKind, TryPool2D};
