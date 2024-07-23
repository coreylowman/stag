mod cpu_kernel;

#[cfg(feature = "cuda")]
mod cuda_kernel;

use crate::{shapes::*, tensor::*};

use super::ReshapeTo;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum Pool2DKind {
    Avg,
    Min,
    Max,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Pool2DOp {
    pub kind: Pool2DKind,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub batch: usize,
    pub chan: usize,
    pub h_in: usize,
    pub h_out: usize,
    pub w_in: usize,
    pub w_out: usize,
}

pub(super) trait Pool2DKernel<E: Dtype>: Storage<E> {
    fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self>, Error>;

    fn forward<I: Shape, O: Shape>(
        &self,
        op: Pool2DOp,
        inp: &Tensor<I, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Error>;

    #[allow(clippy::too_many_arguments)]
    fn backward<I: Shape, O: Shape>(
        &self,
        op: Pool2DOp,
        inp: &Tensor<I, E, Self>,
        grad_inp: &mut Self::Vec,
        out: &Tensor<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Error>;
}

pub trait TryPool2D<Kernel, Stride, Padding, Dilation>: Sized {
    type Pooled;

    fn pool2d(
        self,
        kind: Pool2DKind,
        kernel: Kernel,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
    ) -> Self::Pooled {
        self.try_pool2d(kind, kernel, stride, padding, dilation)
            .unwrap()
    }

    fn try_pool2d(
        self,
        kind: Pool2DKind,
        kernel: Kernel,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
    ) -> Result<Self::Pooled, Error>;
}

#[cfg(feature = "nightly")]
impl<
        const KERNEL: usize,
        const STRIDE: usize,
        const PADDING: usize,
        const DILATION: usize,
        const DIM: usize,
    > TryPool2D<Const<KERNEL>, Const<STRIDE>, Const<PADDING>, Const<DILATION>> for Const<DIM>
where
    Const<{ (DIM + 2 * PADDING - DILATION * (KERNEL - 1) - 1) / STRIDE + 1 }>: Sized,
{
    type Pooled = Const<{ (DIM + 2 * PADDING - DILATION * (KERNEL - 1) - 1) / STRIDE + 1 }>;
    fn try_pool2d(
        self,
        _: Pool2DKind,
        _: Const<KERNEL>,
        _: Const<STRIDE>,
        _: Const<PADDING>,
        _: Const<DILATION>,
    ) -> Result<Self::Pooled, Error> {
        Ok(Const)
    }
}

macro_rules! const_try_pool {
    ($Dim:expr, $Kernel:expr, $Stride:expr, $Padding:expr, $Dilation:expr, out=$Out_dim:expr) => {
        #[cfg(not(feature = "nightly"))]
        impl TryPool2D<Const<$Kernel>, Const<$Stride>, Const<$Padding>, Const<$Dilation>>
            for Const<$Dim>
        {
            // ($Dim + 2 * $Padding - $Dilation * ($Kernel - 1) - 1) / $Stride + 1
            //   def compute_output_size(dim, kernel_size, stride, padding, dilation):
            //    output_size = int(int(dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            //    return output_size
            type Pooled = Const<$Out_dim>;

            fn try_pool2d(
                self,
                _: Pool2DKind,
                _: Const<$Kernel>,
                _: Const<$Stride>,
                _: Const<$Padding>,
                _: Const<$Dilation>,
            ) -> Result<Self::Pooled, Error> {
                Ok(Const)
            }
        }
    };
}

const_try_pool!(1, 2, 1, 0, 1, out = 0);
const_try_pool!(2, 2, 1, 0, 1, out = 1);
const_try_pool!(3, 2, 1, 0, 1, out = 2);
const_try_pool!(4, 2, 1, 0, 1, out = 3);

const_try_pool!(1, 2, 2, 0, 1, out = 0);
const_try_pool!(2, 2, 2, 0, 1, out = 1);
const_try_pool!(3, 2, 2, 0, 1, out = 1);
const_try_pool!(4, 2, 2, 0, 1, out = 2);

const_try_pool!(1, 1, 2, 0, 1, out = 1);
const_try_pool!(2, 1, 2, 0, 1, out = 1);
const_try_pool!(3, 1, 2, 0, 1, out = 2);
const_try_pool!(4, 1, 2, 0, 1, out = 2);

const_try_pool!(4, 2, 1, 0, 2, out = 2);
const_try_pool!(5, 2, 1, 0, 2, out = 3);

impl<Kernel: Dim, Stride: Dim, Padding: Dim, Dilation: Dim>
    TryPool2D<Kernel, Stride, Padding, Dilation> for usize
{
    type Pooled = usize;
    fn try_pool2d(
        self,
        _: Pool2DKind,
        kernel: Kernel,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
    ) -> Result<Self::Pooled, Error> {
        Ok((self + 2 * padding.size() - 1)
            .checked_sub(dilation.size() * (kernel.size() - 1))
            .unwrap()
            / stride.size()
            + 1)
    }
}

impl<Chan, Kernel, Stride, Padding, Dilation, H, W, E, D, T>
    TryPool2D<Kernel, Stride, Padding, Dilation> for Tensor<(Chan, H, W), E, D, T>
where
    Chan: Dim,
    Kernel: Dim,
    Stride: Dim,
    Padding: Dim,
    Dilation: Dim,
    H: Dim + TryPool2D<Kernel, Stride, Padding, Dilation>,
    H::Pooled: Dim,
    W: Dim + TryPool2D<Kernel, Stride, Padding, Dilation>,
    W::Pooled: Dim,
    E: Dtype,
    D: Pool2DKernel<E> + crate::tensor_ops::reshape_to::ReshapeKernel<E>,
    T: Tape<E, D>,
{
    type Pooled = Tensor<(Chan, H::Pooled, W::Pooled), E, D, T>;

    fn try_pool2d(
        self,
        kind: Pool2DKind,
        kernel: Kernel,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
    ) -> Result<Self::Pooled, Error> {
        let (chan, h, w) = self.shape;
        let img = self.try_reshape_like(&(Const::<1>, chan, h, w))?;
        let out = img.try_pool2d(kind, kernel, stride, padding, dilation)?;
        let (_, _, out_h, out_w) = out.shape;
        out.try_reshape_like(&(chan, out_h, out_w))
    }
}

impl<Chan, Kernel, Stride, Padding, Dilation, Batch, H, W, E, D, T>
    TryPool2D<Kernel, Stride, Padding, Dilation> for Tensor<(Batch, Chan, H, W), E, D, T>
where
    Chan: Dim,
    Kernel: Dim,
    Stride: Dim,
    Padding: Dim,
    Dilation: Dim,
    Batch: Dim,
    H: Dim + TryPool2D<Kernel, Stride, Padding, Dilation>,
    H::Pooled: Dim,
    W: Dim + TryPool2D<Kernel, Stride, Padding, Dilation>,
    W::Pooled: Dim,
    E: Dtype,
    D: Pool2DKernel<E>,
    T: Tape<E, D>,
{
    type Pooled = Tensor<(Batch, Chan, H::Pooled, W::Pooled), E, D, T>;

    fn try_pool2d(
        self,
        kind: Pool2DKind,
        kernel: Kernel,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
    ) -> Result<Self::Pooled, Error> {
        let (batch, chan, h, w) = self.shape;
        if self.strides != self.shape.strides() {
            panic!("Image input to pool2d must be contiguous");
        }
        let h_out = h.pool2d(kind, kernel, stride, padding, dilation);
        let w_out = w.pool2d(kind, kernel, stride, padding, dilation);
        let op = Pool2DOp {
            kind,
            stride: stride.size(),
            padding: padding.size(),
            kernel: kernel.size(),
            dilation: dilation.size(),
            batch: batch.size(),
            chan: chan.size(),
            h_in: h.size(),
            h_out: h_out.size(),
            w_in: w.size(),
            w_out: w_out.size(),
        };
        let (img, mut tape) = self.split_tape();
        let mut out = img.device.alloc((batch, chan, h_out, w_out))?;
        img.device.forward(op, &img, &mut out)?;
        let img_ghost = img.ghost();
        let out_ghost = out.ghost();
        let out_clone = out.clone();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&img_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_img, grad_out) = grads.mut_and_ref(&img_ghost, &out_ghost);
            img.device
                .backward(op, &img, grad_img, &out_clone, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{tensor_ops::*, tests::*};

    #[test]
    fn test_pool2d_3d_max2d_eq_grads() {
        let dev: TestDevice = Default::default();
        let x = dev
            .tensor([[[1.0, 1., 0.5, 0.2], [0.2, 0.2, 0.5, 1.2]]])
            .to_dtype::<TestDtype>();
        let r = x.leaky_trace().pool2d(
            Pool2DKind::Max,
            Const::<2>,
            Const::<1>,
            Const::<0>,
            Const::<1>,
        );
        assert_close_to_literal!(r, [[[1., 1., 1.2]]]);
        let g = r.sum().backward();
        assert_close_to_literal!(g.get(&x), [[[1., 2., 0., 0.], [0., 0., 0., 1.]]]);
    }

    #[test]
    fn test_pool2d_3d_min2d_eq_grads() {
        let dev: TestDevice = Default::default();
        let x = dev
            .tensor([[[1., 1., 0.5, 0.2], [0.2, 0.2, 0.5, 1.2]]])
            .to_dtype::<TestDtype>();
        let r = x.leaky_trace().pool2d(
            Pool2DKind::Min,
            Const::<2>,
            Const::<1>,
            Const::<0>,
            Const::<1>,
        );
        assert_close_to_literal!(r, [[[0.2, 0.2, 0.2]]]);
        let g = r.sum().backward();
        assert_close_to_literal!(g.get(&x), [[[0., 0., 0., 1.], [1., 2., 0., 0.]]]);
    }

    #[test]
    fn test_pool2d_3d_max2d() {
        let dev = TestDevice::seed_from_u64(234);
        let x: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let r = x.leaky_trace().pool2d(
            Pool2DKind::Max,
            Const::<2>,
            Const::<2>,
            Const::<0>,
            Const::<1>,
        );
        assert_close_to_literal!(r, [[[1.79155397, 1.10126066]], [[1.14464748, 2.26301837]]]);
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&x),
            [
                [[1.49969184, 0., 0., 0.75198889],[0., 0., 0., 0.],[0., 0., 0., 0.]],
                [[0., 0., 2.40301466, 0.],[0.78533345, 0., 0., 0.],[0., 0., 0., 0.]]
            ]
        );
    }

    #[test]
    fn test_pool2d_3d_min2d() {
        let dev = TestDevice::seed_from_u64(234);
        let x: Tensor<Rank3<2, 3, 4>, TestDtype, _> = dev.sample_normal();
        let r = x.leaky_trace().pool2d(
            Pool2DKind::Min,
            Const::<2>,
            Const::<2>,
            Const::<0>,
            Const::<1>,
        );
        assert_close_to_literal!(
            r,
            [[[-1.09635627, -1.07717276]], [[-0.01996479, -1.82562149]]]
        );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&x),
            [
                [[0., 0., 0., 0.],[0.083521545, 0., 0., 0.08513925],[0., 0., 0., 0.]],
                [[0., 0.2450583, 0., 0.04027937],[0., 0., 0., 0.],[0., 0., 0., 0.]],
            ]
        );
    }

    #[test]
    fn test_pool2d_4d_avg2d() {
        let dev = TestDevice::seed_from_u64(234);
        let x: Tensor<Rank4<2, 4, 2, 2>, TestDtype, _> = dev.sample_normal();
        let r = x.leaky_trace().pool2d(
            Pool2DKind::Avg,
            Const::<1>,
            Const::<2>,
            Const::<0>,
            Const::<1>,
        );
        assert_close_to_literal!(
            r,
            [
                [[[1.791554]], [[-1.0963563]], [[0.86268073]], [[0.28538525]]],
                [[[1.1446475]], [[0.2833436]], [[-1.2026008]], [[0.21327473]]],
            ]
        );
        let g = r.exp().mean().backward();
        #[rustfmt::skip]
        assert_close_to_literal!(
            g.get(&x),
            [
                [[[0.7498459, 0.0], [0.0, 0.0]],[[0.041760772, 0.0], [0.0, 0.0]],[[0.29618803, 0.0], [0.0, 0.0]],[[0.16628431, 0.0], [0.0, 0.0]]],
                [[[0.39266673, 0.0], [0.0, 0.0]],[[0.16594516, 0.0], [0.0, 0.0]],[[0.037551485, 0.0], [0.0, 0.0]],[[0.15471558, 0.0], [0.0, 0.0]]]
            ]
        );
    }

    #[test]
    fn test_pool2d_dilated() {
        let dev: TestDevice = Default::default();
        let x = dev
            .tensor([[
                [0., 1., 2., 4., 5.],
                [6., 7., 8., 9., 10.],
                [11., 12., 13., 14., 15.],
                [16., 17., 18., 19., 20.],
            ]])
            .to_dtype::<TestDtype>();
        let y_max = x.leaky_trace().pool2d(
            Pool2DKind::Max,
            Const::<2>,
            Const::<1>,
            Const::<0>,
            Const::<2>,
        );
        assert_close_to_literal!(y_max, [[[13., 14., 15.], [18., 19., 20.]]]);
        let y_min = x.clone().pool2d(
            Pool2DKind::Min,
            Const::<2>,
            Const::<1>,
            Const::<0>,
            Const::<2>,
        );
        assert_close_to_literal!(y_min, [[[0., 1., 2.], [6., 7., 8.]]]);

        let grads = y_max.mean().backward();
        let v = 1.0 / 6.0;
        assert_close_to_literal!(
            grads.get(&x),
            [[
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [0., 0., v, v, v],
                [0., 0., v, v, v]
            ]]
        );
    }
}
