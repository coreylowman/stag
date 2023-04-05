use super::ops::{BinaryKernel, UnaryKernel};
use crate::{
    shapes::{Dtype, Shape},
    tensor::{
        cpu::{Cpu, LendingIterator, NdIndex},
        unique_id, GhostTensor, Tensor, ZerosTensor,
    },
};

pub trait UnaryDerivative<E> {
    /// Whether the [UnaryDerivative::df] function can re-use the output
    /// from [UnaryDerivative::f].
    const DF_USES_FX: bool;
    /// Whether the derivative of this op can be computed without
    /// any data.
    const HAS_CONST_DF: bool;

    fn f(&self, x: &E) -> E;

    /// Receives `f(x)` if [UnaryDerivative::DF_USES_FX] is true,
    /// otherwise `x`.
    fn df(&self, x: &E) -> E;

    fn const_df(&self) -> E {
        unimplemented!()
    }
}

pub trait BinaryDerivative<E>: std::fmt::Debug {
    /// Whether the derivative of this op can be computed without
    /// any data.
    const HAS_CONST_DF: bool;
    fn f(&self, x: &E, y: &E) -> E;
    fn dfdx(&self, x: &E, y: &E) -> E;
    fn dfdy(&self, x: &E, y: &E) -> E;
    fn const_dfdx(&self) -> E {
        unimplemented!()
    }
    fn const_dfdy(&self) -> E {
        unimplemented!()
    }
}

impl<E: Dtype, Op: UnaryDerivative<E>> UnaryKernel<Op, E> for Cpu {
    const BACKWARD_WITHOUT_DATA: bool = Op::HAS_CONST_DF;
    fn forward<S: Shape>(
        &self,
        op: Op,
        inp: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let mut out = Tensor {
            id: unique_id(),
            data: inp.data.clone(),
            shape: inp.shape,
            strides: inp.strides,
            device: self.clone(),
            tape: Default::default(),
        };
        // NOTE: we can iterate over buf here because we know inp & out
        // have exact same strides due to clone.
        for x in out.buf_iter_mut() {
            *x = op.f(x);
        }
        Ok(out)
    }

    fn backward<S: Shape>(
        &self,
        op: Op,
        inp: &Tensor<S, E, Self>,
        grad_inp: &mut Self::Vec<E>,
        out: &Tensor<S, E, Self>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        debug_assert_eq!(grad_inp.len(), grad_out.len());
        debug_assert_eq!(inp.data.len(), grad_out.len());
        if Op::DF_USES_FX {
            for (i, x) in grad_inp.iter_mut().enumerate() {
                *x += op.df(&out.data[i]) * grad_out[i];
            }
        } else {
            for (i, x) in grad_inp.iter_mut().enumerate() {
                *x += op.df(&inp.data[i]) * grad_out[i];
            }
        }
        Ok(())
    }

    fn forward_reuse<S: Shape>(
        &self,
        op: Op,
        mut out: Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        out.id = unique_id();
        // NOTE: we can iterate over buf here because we know inp & out
        // have exact same strides due to clone.
        for x in out.buf_iter_mut() {
            *x = op.f(x);
        }
        Ok(out)
    }

    fn backward_without_data(
        &self,
        op: Op,
        grad_inp: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        for (i, x) in grad_inp.iter_mut().enumerate() {
            *x += op.const_df() * grad_out[i];
        }
        Ok(())
    }
}

impl<E: Dtype, Op: BinaryDerivative<E>> BinaryKernel<Op, E> for Cpu {
    const BACKWARD_WITHOUT_DATA: bool = Op::HAS_CONST_DF;
    fn forward<S: Shape>(
        &self,
        op: Op,
        lhs: &Tensor<S, E, Self>,
        rhs: &Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let mut out = self.try_zeros_like(&lhs.shape)?;

        let mut lhs_iter = lhs.iter();
        let mut rhs_iter = rhs.iter();
        // NOTE: we can use buf_iter_mut() here because StridedArray::new makes a contiguous array
        for o in out.buf_iter_mut() {
            let l = lhs_iter.next().unwrap();
            let r = rhs_iter.next().unwrap();
            *o = op.f(l, r);
        }
        Ok(out)
    }
    fn forward_reuse<S: Shape>(
        &self,
        op: Op,
        mut lhs: Tensor<S, E, Self>,
        mut rhs: Tensor<S, E, Self>,
    ) -> Result<Tensor<S, E, Self>, Self::Err> {
        let lhs_valid = lhs.strides == lhs.shape.strides();
        let rhs_valid = rhs.strides == rhs.shape.strides();
        if lhs_valid || rhs_valid {
            let lhs_count = std::sync::Arc::strong_count(&lhs.data);
            let rhs_count = std::sync::Arc::strong_count(&rhs.data);
            if lhs_valid && (lhs_count == 1 || !rhs_valid || rhs_count != 1) {
                lhs.id = unique_id();
                let mut rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
                for l in lhs.buf_iter_mut() {
                    *l = op.f(l, &rhs.data[rhs_idx.next().unwrap()]);
                }
                Ok(lhs)
            } else {
                rhs.id = unique_id();
                let mut lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
                for r in rhs.buf_iter_mut() {
                    *r = op.f(&lhs.data[lhs_idx.next().unwrap()], r);
                }
                Ok(rhs)
            }
        } else {
            <Self as BinaryKernel<Op, E>>::forward(self, op, &lhs, &rhs)
        }
    }
    fn backward<S: Shape>(
        &self,
        op: Op,
        lhs: &Tensor<S, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &Tensor<S, E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        let mut lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
        let mut rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
        let lhs_buf = lhs.data.as_ref();
        let rhs_buf = rhs.data.as_ref();
        // NOTE: we can use .buf_iter() here because we know the outcome of this op is
        // contiguous from forward
        for &go in grad_out.iter() {
            let lhs_i = lhs_idx.next().unwrap();
            let rhs_i = rhs_idx.next().unwrap();
            let l = &lhs_buf[lhs_i];
            let r = &rhs_buf[rhs_i];
            grad_lhs[lhs_i] += op.dfdx(l, r) * go;
            grad_rhs[rhs_i] += op.dfdy(l, r) * go;
        }
        Ok(())
    }
    fn backward_without_data<S: Shape>(
        &self,
        op: Op,
        lhs: &GhostTensor<S, E, Self>,
        grad_lhs: &mut Self::Vec<E>,
        rhs: &GhostTensor<S, E, Self>,
        grad_rhs: &mut Self::Vec<E>,
        grad_out: &Self::Vec<E>,
    ) -> Result<(), Self::Err> {
        assert!(Op::HAS_CONST_DF);
        let mut lhs_idx = NdIndex::new(lhs.shape, lhs.strides);
        let mut rhs_idx = NdIndex::new(rhs.shape, rhs.strides);
        for &go in grad_out.iter() {
            let lhs_i = lhs_idx.next().unwrap();
            let rhs_i = rhs_idx.next().unwrap();
            grad_lhs[lhs_i] += op.const_dfdx() * go;
            grad_rhs[rhs_i] += op.const_dfdy() * go;
        }
        Ok(())
    }
}
