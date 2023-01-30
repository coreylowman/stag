#include "unary_op_macros.cuh"

struct SqrtKernelOp {};

UNARY_OP(float, sqrt_forward_f32, sqrt_backward_f32, SqrtKernelOp,
        sqrtf(x),
        0.5 / sqrtf(x))

UNARY_OP(double, sqrt_forward_f64, sqrt_backward_f64, SqrtKernelOp,
        sqrt(x),
        0.5 / sqrt(x))
        