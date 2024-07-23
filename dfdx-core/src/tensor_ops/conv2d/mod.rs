use crate::{shapes::*, tensor::*, tensor_ops::ReshapeTo};

mod cpu_kernel;

#[cfg(all(not(feature = "cudnn"), feature = "cuda"))]
mod cuda_kernel;

#[cfg(feature = "cudnn")]
mod cudnn_kernel;

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(super) struct Conv2DOp {
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub batch: usize,
    pub chan_in: usize,
    pub chan_out: usize,
    pub h_in: usize,
    pub h_out: usize,
    pub w_in: usize,
    pub w_out: usize,
}

pub(super) trait Conv2DKernel<E: Dtype>: Storage<E> {
    fn alloc<S: Shape>(&self, s: S) -> Result<Tensor<S, E, Self>, Error>;

    fn forward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: Conv2DOp,
        lhs: &Tensor<L, E, Self>,
        rhs: &Tensor<R, E, Self>,
        out: &mut Tensor<O, E, Self>,
    ) -> Result<(), Error>;

    #[allow(clippy::too_many_arguments)]
    fn backward<L: Shape, R: Shape, O: Shape>(
        &self,
        op: Conv2DOp,
        lhs: &Tensor<L, E, Self>,
        grad_lhs: &mut Self::Vec,
        rhs: &Tensor<R, E, Self>,
        grad_rhs: &mut Self::Vec,
        out: &impl Tensorlike<O, E, Self>,
        grad_out: &Self::Vec,
    ) -> Result<(), Error>;
}

/// Apply the 2d convolution to a tensor.
///
/// [Const] dims **require nightly**:
/// ```ignore
/// #![feature(generic_const_exprs)]
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let x: Tensor<Rank4<2, 3, 32, 32>, f32, _> = dev.sample_normal();
/// let w: Tensor<Rank4<6, 3, 3, 3>, f32, _> = dev.sample_normal();
/// let y = (x, w).conv2d(
///     Const::<1>, // stride
///     Const::<0>, // padding
///     Const::<1>, // dilation
///     Const::<1>, // groups
/// );
/// ```
///
/// [usize] dims can be used on stable:
/// ```rust
/// # use dfdx_core::prelude::*;
/// # let dev: Cpu = Default::default();
/// let x: Tensor<_, f32, _> = dev.sample_normal_like(&(
///     2,  // batch size
///     3,  // input channels
///     32, // height
///     32, // width
/// ));
/// let w: Tensor<_, f32, _> = dev.sample_normal_like(&(
///     6, // output channels
///     3, // input channels
///     3, // kernel size
///     3, // kernel size
/// ));
/// let y = (x, w).conv2d(
///     1, // stride
///     0, // padding
///     1, // dilation
///     1, // groups
/// );
/// ```
pub trait TryConv2D<Stride, Padding, Dilation, Groups>: Sized {
    type Convolved;

    /// Applies a 2D convolution to the input tensor.
    fn conv2d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
    ) -> Self::Convolved {
        self.try_conv2d(stride, padding, dilation, groups).unwrap()
    }

    /// Fallibly applies a 2D convolution to the input tensor.
    fn try_conv2d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
    ) -> Result<Self::Convolved, Error>;
}

#[cfg(feature = "nightly")]
impl<
        const KERNEL: usize,
        const STRIDE: usize,
        const PADDING: usize,
        const DILATION: usize,
        Groups: Dim,
        const DIM: usize,
    > TryConv2D<Const<STRIDE>, Const<PADDING>, Const<DILATION>, Groups>
    for (Const<DIM>, Const<KERNEL>)
where
    Const<{ (DIM + 2 * PADDING - DILATION * (KERNEL - 1) - 1) / STRIDE + 1 }>: Sized,
{
    type Convolved = Const<{ (DIM + 2 * PADDING - DILATION * (KERNEL - 1) - 1) / STRIDE + 1 }>;
    fn try_conv2d(
        self,
        _: Const<STRIDE>,
        _: Const<PADDING>,
        _: Const<DILATION>,
        _: Groups,
    ) -> Result<Self::Convolved, Error> {
        Ok(Const)
    }
}

macro_rules! const_try_conv {
    ($Dim:expr, $Kernel:expr, $Stride:expr, $Padding:expr, $Dilation:expr, out=$Out_dim:expr) => {
        #[cfg(not(feature = "nightly"))]
        impl<Groups: Dim> TryConv2D<Const<$Stride>, Const<$Padding>, Const<$Dilation>, Groups>
            for (Const<$Dim>, Const<$Kernel>)
        {
            // ($Dim + 2 * $Padding - $Dilation * ($Kernel - 1) - 1) / $Stride + 1
            //   def compute_output_size(dim, kernel_size, stride, padding, dilation):
            //    output_size = int(int(dim + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
            //    return output_size
            type Convolved = Const<$Out_dim>;

            fn try_conv2d(
                self,
                _: Const<$Stride>,
                _: Const<$Padding>,
                _: Const<$Dilation>,
                _: Groups,
            ) -> Result<Self::Convolved, Error> {
                Ok(Const)
            }
        }
    };
}

const_try_conv!(1, 2, 1, 0, 1, out = 0);
const_try_conv!(2, 2, 1, 0, 1, out = 1);
const_try_conv!(3, 2, 1, 0, 1, out = 2);

const_try_conv!(1, 2, 1, 2, 1, out = 4);

const_try_conv!(1, 1, 1, 1, 1, out = 3);
const_try_conv!(1, 2, 1, 1, 1, out = 2);
const_try_conv!(2, 2, 1, 1, 1, out = 3);
const_try_conv!(1, 3, 1, 1, 1, out = 1);
const_try_conv!(2, 3, 1, 1, 1, out = 2);
const_try_conv!(3, 2, 1, 1, 1, out = 4);

const_try_conv!(5, 3, 1, 0, 1, out = 3);

const_try_conv!(2, 2, 2, 0, 1, out = 1);
const_try_conv!(3, 2, 2, 0, 1, out = 1);
const_try_conv!(4, 2, 2, 0, 1, out = 2);

const_try_conv!(4, 2, 1, 0, 2, out = 2);
const_try_conv!(5, 2, 1, 0, 2, out = 3);

const_try_conv!(2, 3, 3, 4, 1, out = 3);
const_try_conv!(4, 3, 3, 4, 1, out = 4);

const_try_conv!(6, 2, 4, 3, 1, out = 3);
const_try_conv!(7, 2, 4, 3, 1, out = 3);

const_try_conv!(14, 3, 1, 0, 1, out = 12);
const_try_conv!(28, 6, 3, 2, 1, out = 9);

const_try_conv!(3, 3, 1, 0, 1, out = 1);
const_try_conv!(3, 3, 1, 1, 1, out = 3);
const_try_conv!(5, 2, 2, 1, 2, out = 3);

impl<Kernel: Dim, Stride: Dim, Padding: Dim, Dilation: Dim, Groups: Dim>
    TryConv2D<Stride, Padding, Dilation, Groups> for (usize, Kernel)
{
    type Convolved = usize;
    fn try_conv2d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        _: Groups,
    ) -> Result<Self::Convolved, Error> {
        let (dim, kernel) = self;
        Ok((dim + 2 * padding.size() - 1)
            .checked_sub(dilation.size() * (kernel.size() - 1))
            .unwrap()
            / stride.size()
            + 1)
    }
}

impl<InpChan, OutChan, Kernel, Stride, Padding, Dilation, Groups, H, W, E, D, T>
    TryConv2D<Stride, Padding, Dilation, Groups>
    for (
        Tensor<(InpChan, H, W), E, D, T>,
        Tensor<
            (
                OutChan,
                <InpChan as std::ops::Div<Groups>>::Output,
                Kernel,
                Kernel,
            ),
            E,
            D,
        >,
    )
where
    InpChan: Dim,
    OutChan: Dim,
    Kernel: Dim,
    Stride: Dim,
    Padding: Dim,
    Dilation: Dim,
    Groups: Dim,
    H: Dim,
    W: Dim,
    E: Dtype,
    D: Conv2DKernel<E> + crate::tensor_ops::reshape_to::ReshapeKernel<E>,
    T: Tape<E, D>,
    InpChan: std::ops::Div<Groups>,
    <InpChan as std::ops::Div<Groups>>::Output: Dim,
    (H, Kernel): TryConv2D<Stride, Padding, Dilation, Groups>,
    (W, Kernel): TryConv2D<Stride, Padding, Dilation, Groups>,
    <(H, Kernel) as TryConv2D<Stride, Padding, Dilation, Groups>>::Convolved: Dim,
    <(W, Kernel) as TryConv2D<Stride, Padding, Dilation, Groups>>::Convolved: Dim,
{
    type Convolved = Tensor<
        (
            OutChan,
            <(H, Kernel) as TryConv2D<Stride, Padding, Dilation, Groups>>::Convolved,
            <(W, Kernel) as TryConv2D<Stride, Padding, Dilation, Groups>>::Convolved,
        ),
        E,
        D,
        T,
    >;

    fn try_conv2d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
    ) -> Result<Self::Convolved, Error> {
        let (img, filters) = self;
        let (inp_chan, h, w) = img.shape;
        let img = img.try_reshape_like(&(Const::<1>, inp_chan, h, w))?;
        let out = (img, filters).try_conv2d(stride, padding, dilation, groups)?;
        let (_, out_chan, out_h, out_w) = out.shape;
        out.try_reshape_like(&(out_chan, out_h, out_w))
    }
}

impl<InpChan, OutChan, Kernel, Stride, Padding, Dilation, Groups, Batch, H, W, E, D, T>
    TryConv2D<Stride, Padding, Dilation, Groups>
    for (
        Tensor<(Batch, InpChan, H, W), E, D, T>,
        Tensor<
            (
                OutChan,
                <InpChan as std::ops::Div<Groups>>::Output,
                Kernel,
                Kernel,
            ),
            E,
            D,
        >,
    )
where
    InpChan: Dim,
    OutChan: Dim,
    Kernel: Dim,
    Stride: Dim,
    Padding: Dim,
    Dilation: Dim,
    Groups: Dim,
    Batch: Dim,
    H: Dim,
    W: Dim,
    E: Dtype,
    D: Conv2DKernel<E>,
    T: Tape<E, D>,
    InpChan: std::ops::Div<Groups>,
    <InpChan as std::ops::Div<Groups>>::Output: Dim,
    (H, Kernel): TryConv2D<Stride, Padding, Dilation, Groups>,
    (W, Kernel): TryConv2D<Stride, Padding, Dilation, Groups>,
    <(H, Kernel) as TryConv2D<Stride, Padding, Dilation, Groups>>::Convolved: Dim,
    <(W, Kernel) as TryConv2D<Stride, Padding, Dilation, Groups>>::Convolved: Dim,
{
    type Convolved = Tensor<
        (
            Batch,
            OutChan,
            <(H, Kernel) as TryConv2D<Stride, Padding, Dilation, Groups>>::Convolved,
            <(W, Kernel) as TryConv2D<Stride, Padding, Dilation, Groups>>::Convolved,
        ),
        E,
        D,
        T,
    >;

    fn try_conv2d(
        self,
        stride: Stride,
        padding: Padding,
        dilation: Dilation,
        groups: Groups,
    ) -> Result<Self::Convolved, Error> {
        let (img, filters) = self;
        assert_eq!(img.shape.1.size(), filters.shape.1.size() * groups.size());
        assert_eq!(filters.shape.2, filters.shape.3);
        let (batch, inp_chan, h, w) = img.shape;
        let (out_chan, inp_chan_over_groups, kernel, _) = filters.shape;
        assert_eq!(inp_chan / groups, inp_chan_over_groups);
        assert!(out_chan.size() % groups.size() == 0);
        if img.strides != img.shape.strides() || filters.strides != filters.shape.strides() {
            panic!("Image & filter inputs to conv2d must be contiguous");
        }
        let h_out = (h, kernel).conv2d(stride, padding, dilation, groups);
        let w_out = (w, kernel).conv2d(stride, padding, dilation, groups);
        let op = Conv2DOp {
            stride: stride.size(),
            padding: padding.size(),
            kernel: kernel.size(),
            dilation: dilation.size(),
            groups: groups.size(),
            batch: batch.size(),
            chan_in: inp_chan.size(),
            chan_out: out_chan.size(),
            h_in: h.size(),
            h_out: h_out.size(),
            w_in: w.size(),
            w_out: w_out.size(),
        };
        let (lhs, ltape) = img.split_tape();
        let (rhs, rtape) = filters.split_tape();
        let mut out = lhs.device.alloc((batch, out_chan, h_out, w_out))?;
        let mut tape = ltape.merge(rtape);
        lhs.device.forward(op, &lhs, &rhs, &mut out)?;
        let lhs_ghost = lhs.ghost();
        let rhs_ghost = rhs.ghost();
        let out_ghost = out.ghost();
        tape.add_backward_op(move |grads| {
            grads.try_alloc_for(&rhs_ghost)?;
            grads.try_alloc_for(&lhs_ghost)?;
            grads.try_alloc_for(&out_ghost)?;
            let (grad_lhs, grad_rhs, grad_out) =
                grads.muts_and_ref(&lhs_ghost, &rhs_ghost, &out_ghost);
            lhs.device
                .backward(op, &lhs, grad_lhs, &rhs, grad_rhs, &out_ghost, grad_out)
        });
        Ok(out.put_tape(tape))
    }
}
