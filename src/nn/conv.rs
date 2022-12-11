use crate::{
    gradients::Tape,
    optim::*,
    shapes::*,
    tensor::{Cpu, Tensor},
    tensor_ops::{BroadcastTo, Device, TryConv2DTo},
};

use super::{Module, ModuleMut, ResetParams};

/// **Requires Nightly** Performs 2d convolutions on 3d and 4d images.
///
/// **Pytorch Equivalent**: `torch.nn.Conv2d`
///
/// Generics:
/// - `IN_CHAN`: The number of input channels in an image.
/// - `OUT_CHAN`: The number of channels in the output of the layer.
/// - `KERNEL_SIZE`: The size of the kernel applied to both width and height of the images.
/// - `STRIDE`: How far to move the kernel each step. Defaults to `1`
/// - `PADDING`: How much zero padding to add around the images. Defaults to `0`.
#[derive(Debug, Clone)]
pub struct Conv2D<
    const IN_CHAN: usize,
    const OUT_CHAN: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
    D: Device<f32> = Cpu,
> {
    pub weight: Tensor<Rank4<OUT_CHAN, IN_CHAN, KERNEL_SIZE, KERNEL_SIZE>, f32, D>,
    pub bias: Tensor<Rank1<OUT_CHAN>, f32, D>,
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize, D>
    GradientUpdate<D, f32> for Conv2D<I, O, K, S, P, D>
where
    D: Device<f32>,
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), <D>::Err>
    where
        U: ParamUpdater<D, f32>,
    {
        self.weight.update(updater, unused)?;
        self.bias.update(updater, unused)?;
        Ok(())
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize, D>
    ResetParams<D, f32> for Conv2D<I, O, K, S, P, D>
where
    D: Device<f32>,
{
    fn try_new(device: &D) -> Result<Self, <D>::Err> {
        let k = (I * K * K) as f32;
        let bound = 1.0 / k.sqrt();
        Ok(Self {
            weight: device.try_uniform(-bound, bound)?,
            bias: device.try_uniform(-bound, bound)?,
        })
    }
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        let k = (I * K * K) as f32;
        let bound = 1.0 / k.sqrt();
        self.weight.try_fill_with_uniform(-bound, bound)?;
        self.bias.try_fill_with_uniform(-bound, bound)?;
        Ok(())
    }
}

impl<const C: usize, const O: usize, const K: usize, const S: usize, const P: usize, D, Img>
    Module<Img> for Conv2D<C, O, K, S, P, D>
where
    D: Device<f32>,
    Img: TryConv2DTo<Tensor<Rank4<O, C, K, K>, f32, D>, S, P>,
    for<'a> Bias2D<'a, O, D>: Module<Img::Output, Output = Img::Output>,
{
    type Output = Img::Output;
    fn forward(&self, x: Img) -> Self::Output {
        Bias2D { beta: &self.bias }.forward(x.conv2d_to(self.weight.clone()))
    }
}

impl<const I: usize, const O: usize, const K: usize, const S: usize, const P: usize, D, Img>
    ModuleMut<Img> for Conv2D<I, O, K, S, P, D>
where
    D: Device<f32>,
    Self: Module<Img>,
{
    type Output = <Self as Module<Img>>::Output;
    fn forward_mut(&mut self, input: Img) -> Self::Output {
        self.forward(input)
    }
}

#[derive(Clone, Debug)]
struct Bias2D<'a, const C: usize, D: Device<f32> = Cpu> {
    beta: &'a Tensor<Rank1<C>, f32, D>,
}

impl<'a, const C: usize, H: Dim, W: Dim, D: Device<f32>, T: Tape<D>>
    Module<Tensor<(Const<C>, H, W), f32, D, T>> for Bias2D<'a, C, D>
{
    type Output = Tensor<(Const<C>, H, W), f32, D, T>;
    fn forward(&self, input: Tensor<(Const<C>, H, W), f32, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_like(input.shape()) + input
    }
}

impl<'a, B: Dim, const C: usize, H: Dim, W: Dim, D: Device<f32>, T: Tape<D>>
    Module<Tensor<(B, Const<C>, H, W), f32, D, T>> for Bias2D<'a, C, D>
{
    type Output = Tensor<(B, Const<C>, H, W), f32, D, T>;
    fn forward(&self, input: Tensor<(B, Const<C>, H, W), f32, D, T>) -> Self::Output {
        self.beta.retaped::<T>().broadcast_like(input.shape()) + input
    }
}

#[cfg(feature = "nightly")]
#[cfg(test)]
mod tests {
    use crate::{
        nn::ModuleBuilder,
        tensor::{AsArray, RandnTensor, ZerosTensor},
        tensor_ops::*,
        tests::build_test_device,
    };

    use super::*;

    #[rustfmt::skip]
    #[test]
    fn test_forward_3d_sizes() {
        let d = build_test_device!();
        let x = d.zeros::<Rank3<3, 10, 10>>();
        let _: Tensor<Rank3<2, 8, 8>, _, _, _> = Conv2D::<3, 2, 3, 1, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank3<4, 8, 8>, _, _, _> = Conv2D::<3, 4, 3, 1, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank3<4, 9, 9>, _, _, _> = Conv2D::<3, 4, 2, 1, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank3<4, 7, 7>, _, _, _> = Conv2D::<3, 4, 4, 1, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank3<2, 4, 4>, _, _, _> = Conv2D::<3, 2, 3, 2, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank3<2, 3, 3>, _, _, _> = Conv2D::<3, 2, 3, 3, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank3<2, 10, 10>, _, _, _> = Conv2D::<3, 2, 3, 1, 1, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank3<2, 12, 12>, _, _, _> = Conv2D::<3, 2, 3, 1, 2, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank3<2, 6, 6>, _, _, _> = Conv2D::<3, 2, 3, 2, 2, _>::new(&d).forward(x.clone());
    }

    #[rustfmt::skip]
    #[test]
    fn test_forward_4d_sizes() {
        let d = build_test_device!();
        let x = d.zeros::<Rank4<5, 3, 10, 10>>();
        let _: Tensor<Rank4<5, 2, 8, 8>, _, _, _> = Conv2D::<3, 2, 3, 1, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank4<5, 4, 8, 8>, _, _, _> = Conv2D::<3, 4, 3, 1, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank4<5, 4, 9, 9>, _, _, _> = Conv2D::<3, 4, 2, 1, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank4<5, 4, 7, 7>, _, _, _> = Conv2D::<3, 4, 4, 1, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 4, 4>, _, _, _> = Conv2D::<3, 2, 3, 2, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 3, 3>, _, _, _> = Conv2D::<3, 2, 3, 3, 0, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 10, 10>, _, _, _> = Conv2D::<3, 2, 3, 1, 1, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 12, 12>, _, _, _> = Conv2D::<3, 2, 3, 1, 2, _>::new(&d).forward(x.clone());
        let _: Tensor<Rank4<5, 2, 6, 6>, _, _, _> = Conv2D::<3, 2, 3, 2, 2, _>::new(&d).forward(x.clone());
    }

    #[test]
    fn test_2_conv_sizes() {
        let dev = Cpu::default();
        type A = Conv2D<1, 2, 3>;
        type B = Conv2D<2, 4, 3>;
        let _: Tensor<Rank3<4, 6, 6>, _, _> =
            <(A, B)>::new(&dev).forward(dev.zeros::<Rank3<1, 10, 10>>());
    }

    #[test]
    fn test_3_conv_sizes() {
        type A = Conv2D<1, 2, 3>;
        type B = Conv2D<2, 4, 3>;
        type C = Conv2D<4, 1, 1, 1, 1>;

        let dev = Cpu::default();
        let _: Tensor<Rank3<1, 8, 8>, _, _> =
            <(A, B, C)>::new(&dev).forward_mut(dev.zeros::<Rank3<1, 10, 10>>());
    }

    #[test]
    fn test_conv_with_optimizer() {
        let dev = build_test_device!();

        let mut m: Conv2D<2, 4, 3, 1, 0, _> = dev.build();

        let weight_init = m.weight.clone();
        let bias_init = m.bias.clone();

        let mut opt: Sgd<_, _> = Default::default();
        let out = m.forward(dev.randn::<Rank4<8, 2, 28, 28>>().trace());
        let g = out.square().mean().backward();

        assert_ne!(g.get(&m.weight).array(), [[[[0.0; 3]; 3]; 2]; 4]);
        assert_ne!(g.get(&m.bias).array(), [0.0; 4]);

        opt.update(&mut m, g).expect("unused params");

        assert_ne!(weight_init.array(), m.weight.array());
        assert_ne!(bias_init.array(), m.bias.array());
    }
}
