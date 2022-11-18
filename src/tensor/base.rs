use crate::arrays::{
    Dtype, HasDtype, HasShape, Rank0, Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Shape,
};
use crate::devices::Device;
use crate::unique_id::HasUniqueId;
use crate::{
    gradients::{NoneTape, OwnedTape, Tape},
    unique_id::UniqueId,
};

#[derive(Debug, Clone)]
pub struct Tensor<S: Shape, E: Dtype, D: Device, T = NoneTape> {
    pub(crate) id: UniqueId,
    pub(crate) storage: D::Storage<S, E>,
    pub(crate) device: D,
    pub(crate) tape: T,
}

impl<S: Shape, E: Dtype, D: Device, T: Tape<D>> Tensor<S, E, D, T> {
    pub fn split_tape(self) -> (Tensor<S, E, D, NoneTape>, T) {
        (
            Tensor {
                id: self.id,
                storage: self.storage,
                device: self.device,
                tape: NoneTape,
            },
            self.tape,
        )
    }

    pub fn with_empty_tape(&self) -> Self {
        Tensor {
            id: self.id,
            storage: self.storage.clone(),
            device: self.device.clone(),
            tape: T::empty(self.device.clone()),
        }
    }

    pub fn with_diff_tape<New: Tape<D>>(&self) -> Tensor<S, E, D, New> {
        Tensor {
            id: self.id,
            storage: self.storage.clone(),
            device: self.device.clone(),
            tape: New::empty(self.device.clone()),
        }
    }

    pub fn with_none_tape(&self) -> Tensor<S, E, D, NoneTape> {
        Tensor {
            id: self.id,
            storage: self.storage.clone(),
            device: self.device.clone(),
            tape: NoneTape,
        }
    }

    pub fn put_tape<New: Tape<D>>(self, tape: New) -> Tensor<S, E, D, New> {
        Tensor {
            id: self.id,
            storage: self.storage,
            device: self.device,
            tape,
        }
    }
}

impl<S: Shape, E: Dtype, D: Device, T> HasShape for Tensor<S, E, D, T> {
    type Shape = S;
    fn shape(&self) -> &Self::Shape {
        self.storage.shape()
    }
}

impl<S: Shape, E: Dtype, D: Device, T> HasDtype for Tensor<S, E, D, T> {
    type Dtype = E;
}

impl<S: Shape, E: Dtype, D: Device, T> HasUniqueId for Tensor<S, E, D, T> {
    fn id(&self) -> &UniqueId {
        &self.id
    }
}

impl<S: Shape, E: Dtype, D: Device> Tensor<S, E, D, NoneTape> {
    pub fn trace(&self) -> Tensor<S, E, D, OwnedTape<D>> {
        self.clone().traced()
    }

    pub fn traced(self) -> Tensor<S, E, D, OwnedTape<D>> {
        let tape = OwnedTape::empty(self.device.clone());
        self.put_tape(tape)
    }
}

pub type Tensor0D<D, Tape = NoneTape, Elem = f32> = Tensor<Rank0, Elem, D, Tape>;
pub type Tensor1D<const M: usize, D, Tape = NoneTape, Elem = f32> = Tensor<Rank1<M>, Elem, D, Tape>;
pub type Tensor2D<const M: usize, const N: usize, D, Tape = NoneTape, Elem = f32> =
    Tensor<Rank2<M, N>, Elem, D, Tape>;
pub type Tensor3D<const M: usize, const N: usize, const O: usize, D, Tape = NoneTape, Elem = f32> =
    Tensor<Rank3<M, N, O>, Elem, D, Tape>;
pub type Tensor4D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    D,
    Tape = NoneTape,
    Elem = f32,
> = Tensor<Rank4<M, N, O, P>, Elem, D, Tape>;
pub type Tensor5D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
    D,
    Tape = NoneTape,
    Elem = f32,
> = Tensor<Rank5<M, N, O, P, Q>, Elem, D, Tape>;
pub type Tensor6D<
    const M: usize,
    const N: usize,
    const O: usize,
    const P: usize,
    const Q: usize,
    const R: usize,
    D,
    Tape = NoneTape,
    Elem = f32,
> = Tensor<Rank6<M, N, O, P, Q, R>, Elem, D, Tape>;