#include "unary_op_macros.cuh"

struct ReLUKernelOp {};

UNARY_OP(float, relu_forward_f32, relu_backward_f32, ReLUKernelOp,
        fmaxf(x, 0.0),
        x > 0.0 ? 1.0 : 0.0)

UNARY_OP(double, relu_forward_f64, relu_backward_f64, ReLUKernelOp,
        fmax(x, 0.0),
        x > 0.0 ? 1.0 : 0.0)
        