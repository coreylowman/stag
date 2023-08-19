#![feature(generic_const_exprs)]

mod abs;
mod batch_norm2d;
mod bias1d;
mod bias2d;
mod conv2d;
mod cos;
mod dropout;
mod exp;
mod flatten2d;
mod gelu;
mod generalized_add;
mod layer_norm1d;
mod leaky_relu;
mod linear;
mod ln;
mod log_softmax;
mod matmul;
mod multi_head_attention;
mod optim;
mod pool_2d_avg;
mod pool_2d_max;
mod pool_2d_min;
mod pool_global_avg;
mod pool_global_max;
mod pool_global_min;
mod prelu;
mod prelu1d;
mod relu;
mod reshape;
mod residual_add;
mod sigmoid;
mod sin;
mod softmax;
mod sqrt;
mod square;
mod tanh;
mod transformer;

pub use dfdx_nn_core::*;
pub use dfdx_nn_derives::*;

pub use optim::adam::Adam;
pub use optim::rmsprop::RMSprop;
pub use optim::sgd::Sgd;

pub use abs::Abs;
pub use batch_norm2d::{BatchNorm2D, BatchNorm2DConfig, BatchNorm2DConstConfig};
pub use bias1d::{Bias1D, Bias1DConfig, Bias1DConstConfig};
pub use bias2d::{Bias2D, Bias2DConfig, Bias2DConstConfig};
pub use conv2d::{Conv2D, Conv2DConfig, Conv2DConstConfig};
pub use cos::Cos;
pub use dropout::{Dropout, DropoutOneIn};
pub use exp::Exp;
pub use flatten2d::Flatten2D;
pub use gelu::{AccurateGeLU, FastGeLU};
pub use generalized_add::GeneralizedAdd;
pub use layer_norm1d::{LayerNorm1D, LayerNorm1DConfig, LayerNorm1DConstConfig};
pub use leaky_relu::LeakyReLU;
pub use linear::{Linear, LinearConfig, LinearConstConfig};
pub use ln::Ln;
pub use log_softmax::LogSoftmax;
pub use matmul::{MatMul, MatMulConfig, MatMulConstConfig};
pub use multi_head_attention::{MultiHeadAttention, MultiHeadAttentionConfig};
pub use pool_2d_avg::{AvgPool2D, AvgPool2DConst};
pub use pool_2d_max::{MaxPool2D, MaxPool2DConst};
pub use pool_2d_min::{MinPool2D, MinPool2DConst};
pub use pool_global_avg::AvgPoolGlobal;
pub use pool_global_max::MaxPoolGlobal;
pub use pool_global_min::MinPoolGlobal;
pub use prelu::{PReLU, PReLUConfig};
pub use prelu1d::{PReLU1D, PReLU1DConfig};
pub use relu::ReLU;
pub use reshape::Reshape;
pub use residual_add::ResidualAdd;
pub use sigmoid::Sigmoid;
pub use sin::Sin;
pub use softmax::Softmax;
pub use sqrt::Sqrt;
pub use square::Square;
pub use tanh::Tanh;
pub use transformer::{
    DecoderBlock, DecoderBlockConfig, EncoderBlock, EncoderBlockConfig, Transformer,
    TransformerConfig,
};
