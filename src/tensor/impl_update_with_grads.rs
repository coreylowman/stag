use std::ops::SubAssign;

use crate::{
    arrays::{Dtype, Shape},
    devices::Device,
    gradients::{CanUpdateWithGradients, GradientProvider, UnusedTensors},
};

use super::Tensor;

impl<S: Shape, E: Dtype, D: Device, T> CanUpdateWithGradients<D> for Tensor<S, E, D, T>
where
    D::Storage<S, E>: std::ops::SubAssign,
{
    /// Subtracts the gradient for the tensor from [HasArrayData::mut_data].
    fn update<G: GradientProvider<D>>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        match grads.gradient(self) {
            Some(grad) => self.storage.sub_assign(grad),
            None => unused.add(self),
        }
    }
}
