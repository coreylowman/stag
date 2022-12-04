//! Demonstrates broadcasting tensors to different sizes, and axis reductions
//! with BroadcastTo and ReduceTo

use dfdx::{
    arrays::{Axis, Rank2, Rank4},
    tensor::{AsArray, Cpu, Tensor, TensorFromArray},
    tensor_ops::{Broadcast, BroadcastAlong, MeanTo},
};

fn main() {
    let dev: Cpu = Default::default();
    let a = dev.tensor([1.0, 2.0, 3.0]);

    // to broadcast, use `Broadcast::broadcast()` and specify
    // the output type. the axes that are broadcast are inferred for you!
    let b = a.broadcast::<Rank2<5, 3>>();
    assert_eq!(b.array(), [[1.0, 2.0, 3.0]; 5]);

    // we can really broadcast any axes on either side
    // here a (5,3) tensor is broadcast to (7,5,3,2).
    // so 7 is added in front, and 2 is added last
    let c = b.broadcast::<Rank4<7, 5, 3, 2>>();
    assert_eq!(c.array(), [[[[1.0; 2], [2.0; 2], [3.0; 2]]; 5]; 7]);

    // the opposite of broadcast is reducing
    // we've already introduced one reduction which is mean
    let d = c.mean_to::<Rank2<5, 3>>();
    assert_eq!(d.array(), [[1.0, 2.0, 3.0]; 5]);

    // Sometimes it's ambiguous which axes you mean to broadcast or reduce.
    // Here rust doesn't know if the new axis is the first or second.
    // We can use BroadcastAlong here to specify the new axis, and we
    // have to tell it the new shape as well.
    let e = dev.tensor([1.0]);
    let f: Tensor<Rank2<1, 1>, f32, _> = e.broadcast_along::<Axis<1>>();
    // NOTE: will fail with "Multiple impls satisfying...":
    // let f = e.broadcast::<Rank2<1, 1>>();

    // reductions have the same problem when it's ambiguous,
    // so we can also use mean_along with an axis
    let _ = f.mean_along::<Axis<0>>();
}
