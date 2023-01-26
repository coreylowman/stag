use crate::{
    shapes::{Shape, Unit},
    tensor::cuda::Cuda,
    tensor::cuda::CudaArray,
};
use cudarc::driver::{CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

use super::CmpKernel;

pub(crate) trait CmpOpCudaKernel<E: Unit> {
    /// Compiled by build.rs
    const PTX_SRC: &'static str;

    /// Unique name for the kernel
    const MODULE_NAME: &'static str;

    /// Name of function in the .cu file
    const FWD_FN_NAME: &'static str;
}

impl<Op: CmpOpCudaKernel<f32>> CmpKernel<Op, f32> for Cuda {
    fn forward<S: Shape>(
        &self,
        lhs: &Self::Storage<S, f32>,
        rhs: &Self::Storage<S, f32>,
    ) -> Result<Self::Storage<S, bool>, Self::Err> {
        if !self.dev.has_func(Op::MODULE_NAME, Op::FWD_FN_NAME) {
            self.dev
                .load_ptx(Op::PTX_SRC.into(), Op::MODULE_NAME, &[Op::FWD_FN_NAME])?;
        }

        let shape = lhs.shape;
        let strides = lhs.shape.strides();
        let numel = shape.num_elements();

        let mut storage = self.dev.alloc_zeros_async::<bool>(numel)?;

        let dims: CudaSlice<usize> = self.dev.take_async(shape.concrete().into())?;
        let lhs_strides: CudaSlice<usize> = self.dev.take_async(lhs.strides.into())?;
        let rhs_strides: CudaSlice<usize> = self.dev.take_async(rhs.strides.into())?;
        let out_strides: CudaSlice<usize> = self.dev.take_async(strides.into())?;

        let fwd_fn = self.dev.get_func(Op::MODULE_NAME, Op::FWD_FN_NAME).unwrap();
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let params = (
            numel,             // const size_t numel,
            S::NUM_DIMS,       // const size_t num_dims,
            &dims,             // const size_t *dims,
            lhs.data.as_ref(), // const float *lhs,
            &lhs_strides,      // const size_t *lhs_strides,
            rhs.data.as_ref(), // const float *rhs,
            &rhs_strides,      // const size_t *rhs_strides,
            &mut storage,      // bool *out,
            &out_strides,      // const size_t *out_strides
        );
        unsafe { fwd_fn.launch_async(cfg, params) }?;
        Ok(CudaArray {
            data: Arc::new(storage),
            shape,
            strides,
        })
    }
}

use super::EqKernelOp;

impl CmpOpCudaKernel<f32> for EqKernelOp {
    const PTX_SRC: &'static str = include_str!(concat!(env!("OUT_DIR"), "/cmp.ptx"));
    const MODULE_NAME: &'static str = "eq";
    const FWD_FN_NAME: &'static str = "eq_forward";
}
