use crate::prelude::*;

/// Implements RMS layer normalization as described in [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).
///
/// This calls [normalize_rms()] on the last axis of the input to normalize to unit std dev, and then does an element-wise
/// affine transform using learnable parameters.
///
/// Epsilon is passed to [normalize_rms()] and added to the variance to ensure big enough numbers. It defaults to `1e-5`.
///
/// Generics:
/// - `M` The size of the affine transform tensors.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// # use dfdx::*;
/// # let dev: Cpu = Default::default();
/// type Model = LayerRMSNorm1DConstConfig<5>;
/// let model = dev.build_module::<f32>(Model::default());
/// let _: Tensor<Rank1<5>, f32, _> = model.forward(dev.zeros::<Rank1<5>>());
/// ```
#[derive(Default, Clone, Copy, Debug)]
#[repr(transparent)]
pub struct LayerRMSNorm1DConfig<M: Dim>(pub M);

/// Compile time sugar alias around [LayerRMSNorm1DConfig]
pub type LayerRMSNorm1DConstConfig<const M: usize> = LayerRMSNorm1DConfig<Const<M>>;

impl<M: Dim, E: Dtype, D: Device<E>> BuildOnDevice<E, D> for LayerRMSNorm1DConfig<M> {
    type Built = LayerRMSNorm1D<M, E, D>;
    fn try_build_on_device(&self, device: &D) -> Result<Self::Built, crate::tensor::Error> {
        Ok(LayerRMSNorm1D {
            gamma: device.try_ones_like(&(self.0,))?,
            beta: device.try_zeros_like(&(self.0,))?,
            epsilon: 1e-5,
        })
    }
}

/// See [LayerRMSNorm1DConfig]
#[derive(Clone, Debug, UpdateParams, ZeroGrads)]
#[cfg_attr(feature = "safetensors", derive(SaveSafeTensors, LoadSafeTensors))]
pub struct LayerRMSNorm1D<M: Dim, Elem: Dtype, Dev: Device<Elem>> {
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub gamma: Tensor<(M,), Elem, Dev>,
    #[param]
    #[cfg_attr(feature = "safetensors", serialize)]
    pub beta: Tensor<(M,), Elem, Dev>,
    #[cfg_attr(feature = "safetensors", serialize)]
    pub epsilon: f64,
}

impl<M: Dim, E: Dtype, D: Device<E>> ResetParams<E, D> for LayerRMSNorm1D<M, E, D> {
    fn try_reset_params(&mut self) -> Result<(), crate::tensor::Error> {
        self.gamma.try_fill_with_ones()?;
        self.beta.try_fill_with_zeros()?;
        Ok(())
    }
}

impl<M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(M,), E, D, T>>
    for LayerRMSNorm1D<M, E, D>
{
    type Output = Tensor<(M,), E, D, T>;
    fn try_forward(&self, x: Tensor<(M,), E, D, T>) -> Result<Self::Output, Error> {
        let x = x.try_normalize_rms::<Axis<0>>(self.epsilon)?;
        let x = self.gamma.retaped::<T>().try_mul(x)?;
        self.beta.retaped::<T>().try_add(x)
    }
}

impl<Batch: Dim, M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>> Module<Tensor<(Batch, M), E, D, T>>
    for LayerRMSNorm1D<M, E, D>
{
    type Output = Tensor<(Batch, M), E, D, T>;
    fn try_forward(&self, x: Tensor<(Batch, M), E, D, T>) -> Result<Self::Output, Error> {
        let x = x.try_normalize_rms::<Axis<1>>(self.epsilon)?;
        let x = self.gamma.retaped::<T>().broadcast_like(&x).try_mul(x)?;
        self.beta.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}

impl<Batch: Dim, Seq: Dim, M: Dim, E: Dtype, D: Device<E>, T: Tape<E, D>>
    Module<Tensor<(Batch, Seq, M), E, D, T>> for LayerRMSNorm1D<M, E, D>
{
    type Output = Tensor<(Batch, Seq, M), E, D, T>;
    fn try_forward(&self, x: Tensor<(Batch, Seq, M), E, D, T>) -> Result<Self::Output, Error> {
        let x = x.try_normalize_rms::<Axis<2>>(self.epsilon)?;
        let x = self.gamma.retaped::<T>().broadcast_like(&x).try_mul(x)?;
        self.beta.retaped::<T>().broadcast_like(&x).try_add(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    #[test]
    fn test_layer_rms_norm_reset() {
        let dev: TestDevice = Default::default();

        let mut m = dev.build_module::<TestDtype>(<LayerRMSNorm1DConstConfig<5>>::default());
        assert_close_to_literal!(m.gamma, [1.0; 5]);
        assert_close_to_literal!(m.beta, [0.0; 5]);

        m.gamma = dev.sample_normal();
        m.beta = dev.sample_normal();

        assert_ne!(m.gamma.array(), [TestDtype::ONE; 5]);
        assert_ne!(m.beta.array(), [TestDtype::default(); 5]);

        m.reset_params();

        assert_close_to_literal!(m.gamma, [1.0; 5]);
        assert_close_to_literal!(m.beta, [0.0; 5]);
    }

    #[test]
    fn test_layer_rms_norm_1d_forward() {
        let dev: TestDevice = Default::default();
        let mut m = dev.build_module::<TestDtype>(<LayerRMSNorm1DConstConfig<5>>::default());
        let x = dev.sample_normal::<Rank1<5>>();
        let r = m.forward_mut(x.leaky_trace());
        assert_close_to_literal!(
            r,
            [0.53631353, 0.6458002, -1.8330059, 0.12289862, -0.9593052]
        );
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&m.gamma),
            [0.10726271, 0.12916003, -0.3666012, 0.024579724, -0.19186105]
        );
        assert_close_to_literal!(g.get(&m.beta), [0.2; 5]);
    }

    #[test]
    fn test_layer_rms_norm_2d_forward() {
        let dev: TestDevice = Default::default();
        let m = dev.build_module::<TestDtype>(<LayerRMSNorm1DConstConfig<5>>::default());
        let x = dev.sample_normal::<Rank2<3, 5>>();
        let r = m.forward(x.leaky_trace());
        assert_close_to_literal!(
            r,
            [
                [0.53631353, 0.6458002, -1.8330059, 0.12289862, -0.9593052],
                [1.0418473, -1.199064, 0.49583954, 0.5000605, 1.4074267],
                [0.90727454, -1.6644237, -0.5176145, 1.0127299, -0.33612955]
            ]
        );
        let g = r.mean().backward();
        assert_close_to_literal!(
            g.get(&m.gamma),
            [
                0.16569571,
                -0.14784585,
                -0.123652056,
                0.10904594,
                0.0074661337
            ]
        );
        assert_close_to_literal!(g.get(&m.beta), [0.2; 5]);
    }
}

// Implementation references:
// - https://github.com/johnma2006/mamba-minimal/blob/03de542a36d873f6e6c4057ad687278cc6ae944d/model.py#L328
// - https://github.com/kroggen/mamba.c/blob/7387f49e352f86a0c22041c0f66fd2a40b58a207/mamba.c#L222
