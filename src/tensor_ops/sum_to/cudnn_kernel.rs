use crate::{
    shapes::*,
    tensor::{launch_cfg, Cuda, GhostTensor, Tensor},
    tensor_ops::reduction_utils::*,
};

use cudarc::{
    cudnn::{CudnnDataType, ReduceTensor},
    driver::{DeviceRepr, LaunchAsync, ValidAsZeroBits},
};

const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/sum_to.ptx"));

trait HasCudaKernel<E> {
    const MOD: &'static str;
    const FNS: &'static [&'static str];
}

impl HasCudaKernel<f32> for Cuda {
    const MOD: &'static str = "sum_f32";
    const FNS: &'static [&'static str] = &["sum_to_fwd_f32", "sum_to_bwd_f32"];
}

impl HasCudaKernel<f64> for Cuda {
    const MOD: &'static str = "sum_f64";
    const FNS: &'static [&'static str] = &["sum_to_fwd_f64", "sum_to_bwd_f64"];
}

fn make_at_least_4d<S: Shape>(shape: S, strides: S::Concrete) -> (Vec<i32>, Vec<i32>) {
    if S::NUM_DIMS >= 4 {
        (
            shape.concrete().into_iter().map(|x| x as i32).collect(),
            strides.into_iter().map(|x| x as i32).collect(),
        )
    } else {
        let src = shape.concrete();
        let mut shape = [1usize; 4];
        let mut strides = [0usize; 4];
        for i in 0..S::NUM_DIMS {
            shape[4 - S::NUM_DIMS + i] = src[i];
            strides[4 - S::NUM_DIMS + i] = strides[i];
        }
        (
            shape.into_iter().map(|x| x as i32).collect(),
            strides.into_iter().map(|x| x as i32).collect(),
        )
    }
}

impl<E: Dtype + ValidAsZeroBits + DeviceRepr + CudnnDataType> super::SumKernel<E> for Cuda
where
    Self: HasCudaKernel<E>,
{
    fn forward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &Tensor<Src, E, Self>,
    ) -> Result<Tensor<Dst, E, Self>, Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        assert!(Src::NUM_DIMS <= 8);
        assert!(Dst::NUM_DIMS <= 8);

        if !self.dev.has_func(Self::MOD, Self::FNS[0]) {
            self.dev.load_ptx(PTX_SRC.into(), Self::MOD, Self::FNS)?;
        }

        let mut storage = unsafe { self.dev.alloc::<E>(dst.num_elements()) }?;

        let reduce = self.cudnn.create_reduction_no_indices::<E>(
            cudarc::cudnn::sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
            cudarc::cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
        )?;
        let a = {
            let (a_dims, a_strides) = make_at_least_4d(inp.shape, inp.strides);
            self.cudnn.create_nd_tensor(&a_dims, &a_strides)
        }?;
        let c = {
            let (c_dims, c_strides) = make_at_least_4d(dst, dst.strides());
            self.cudnn.create_nd_tensor(&c_dims, &c_strides)
        }?;
        let op = ReduceTensor {
            reduce: &reduce,
            a: &a,
            c: &c,
        };

        let workspace_size = op.get_workspace_size().unwrap();
        let mut workspace = unsafe { self.get_workspace::<u8>(workspace_size) }?;
        let mut workspace = unsafe { workspace.transmute_mut::<u8>(workspace_size).unwrap() };

        unsafe {
            op.launch(
                &mut workspace,
                (E::ONE, E::default()),
                inp.data.as_ref(),
                &mut storage,
            )
        }?;
        Ok(self.build_tensor(dst, dst.strides(), storage))
    }

    fn backward<Src: Shape, Dst: Shape, Ax: Axes>(
        &self,
        dst: Dst,
        inp: &GhostTensor<Src, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err>
    where
        Src: ReduceShapeTo<Dst, Ax>,
    {
        let bwd_fn = self.dev.get_func(Self::MOD, Self::FNS[1]).unwrap();

        let out_strides: Src::Concrete =
            BroadcastStridesTo::<Src, Ax>::broadcast_strides(&dst, dst.strides());
        let physical_numel = inp.len;
        let elems_per_thread = E::from_usize(reduction_elems_per_thread::<_, Src>(
            inp.shape.concrete(),
            inp.strides,
            Ax::as_array(),
        ))
        .unwrap();

        let cfg = launch_cfg(physical_numel as u32);

        let mut info: Vec<usize> = Vec::with_capacity(3 * Src::NUM_DIMS);
        info.extend(inp.shape.concrete());
        info.extend(inp.strides);
        info.extend(out_strides);
        let info = self.dev.htod_copy(info)?;

        let params = (
            physical_numel,   // const size_t numel,
            Src::NUM_DIMS,    // const size_t num_dims,
            elems_per_thread, // const float elems_per_thread,
            &info,            // const size_t *info,
            grad_inp,         // float *grad_inp,
            grad_out,         // const float *grad_out,
        );
        unsafe { bwd_fn.launch(cfg, params) }?;
        Ok(())
    }
}
