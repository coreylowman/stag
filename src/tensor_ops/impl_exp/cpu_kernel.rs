use super::ExpKernelOp;
use crate::tensor_ops::utils::cpu::UnaryDerivative;

impl UnaryDerivative<f32> for ExpKernelOp {
    #[inline(always)]
    fn f(&self, x: &f32) -> f32 {
        x.exp()
    }
    #[inline(always)]
    fn df(&self, x: &f32) -> f32 {
        x.exp()
    }
}
