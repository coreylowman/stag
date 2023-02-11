use crate::tensor_ops::cuda_kernels::UnaryOpCudaKernel;

unsafe impl cudarc::driver::AsKernelParam for super::SquareKernelOp {}

const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/square.ptx"));

impl UnaryOpCudaKernel<f32> for super::SquareKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "square_f32";
    const FWD_FN_NAME: &'static str = "square_forward_f32";
    const BWD_FN_NAME: &'static str = "square_backward_f32";
}

impl UnaryOpCudaKernel<f64> for super::SquareKernelOp {
    const PTX_SRC: &'static str = PTX_SRC;
    const MODULE_NAME: &'static str = "square_f64";
    const FWD_FN_NAME: &'static str = "square_forward_f64";
    const BWD_FN_NAME: &'static str = "square_backward_f64";
}
