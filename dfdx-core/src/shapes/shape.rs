use super::{axes::*, ReduceShape, ReduceShapeTo};

/// Represents a single dimension of a multi dimensional [Shape]
pub trait Dim: 'static + Copy + Clone + std::fmt::Debug + Send + Sync + Eq + PartialEq {
    fn size(&self) -> usize;
    fn from_size(size: usize) -> Option<Self>;
}

/// Represents a single dimension where all
/// instances are guaranteed to be the same size at compile time.
pub trait ConstDim: Default + Dim {
    const SIZE: usize;
}

impl Dim for usize {
    #[inline(always)]
    fn size(&self) -> usize {
        *self
    }
    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        Some(size)
    }
}

/// Represents a [Dim] with size known at compile time
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Const<const M: usize>;
impl<const M: usize> Dim for Const<M> {
    #[inline(always)]
    fn size(&self) -> usize {
        M
    }
    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        if size == M {
            Some(Const)
        } else {
            None
        }
    }
}

impl<const M: usize> ConstDim for Const<M> {
    const SIZE: usize = M;
}

impl<const N: usize> core::ops::Add<Const<N>> for usize {
    type Output = usize;
    fn add(self, _: Const<N>) -> Self::Output {
        self.size() + N
    }
}
impl<const N: usize> core::ops::Add<usize> for Const<N> {
    type Output = usize;
    fn add(self, rhs: usize) -> Self::Output {
        N + rhs.size()
    }
}

#[cfg(feature = "nightly")]
impl<const N: usize, const M: usize> core::ops::Add<Const<N>> for Const<M>
where
    Const<{ M + N }>: Sized,
{
    type Output = Const<{ M + N }>;
    fn add(self, _: Const<N>) -> Self::Output {
        Const
    }
}

impl<const N: usize> core::ops::Mul<Const<N>> for usize {
    type Output = usize;
    fn mul(self, _: Const<N>) -> Self::Output {
        self.size() * N
    }
}
impl<const N: usize> core::ops::Mul<usize> for Const<N> {
    type Output = usize;
    fn mul(self, rhs: usize) -> Self::Output {
        N * rhs.size()
    }
}

#[cfg(feature = "nightly")]
impl<const N: usize, const M: usize> core::ops::Mul<Const<N>> for Const<M>
where
    Const<{ M * N }>: Sized,
{
    type Output = Const<{ M * N }>;
    fn mul(self, _: Const<N>) -> Self::Output {
        Const
    }
}

impl<const N: usize> core::ops::Div<Const<N>> for usize {
    type Output = usize;
    fn div(self, _: Const<N>) -> Self::Output {
        self.size() / N
    }
}
impl<const N: usize> core::ops::Div<usize> for Const<N> {
    type Output = usize;
    fn div(self, rhs: usize) -> Self::Output {
        N / rhs.size()
    }
}

#[cfg(feature = "nightly")]
impl<const N: usize, const M: usize> core::ops::Div<Const<N>> for Const<M>
where
    Const<{ M / N }>: Sized,
{
    type Output = Const<{ M / N }>;
    fn div(self, _: Const<N>) -> Self::Output {
        Const
    }
}

macro_rules! add_impl {
    ($X1:expr, $X2:expr, $Y:expr) => {
        #[cfg(not(feature = "nightly"))]
        impl core::ops::Add<Const<$X2>> for Const<$X1> {
            type Output = Const<$Y>;
            fn add(self, _: Const<$X2>) -> Self::Output {
                Const
            }
        }
    };
}

macro_rules! add_impl_2 {
    ($X1:expr, $X2:expr, $Y:expr) => {
        add_impl!($X1, $X2, $Y);
        add_impl!($X2, $X1, $Y);
    };
}

add_impl!(1, 1, 2);
add_impl!(2, 2, 4);
add_impl!(3, 3, 6);
add_impl_2!(1, 2, 3);
add_impl_2!(1, 3, 4);
add_impl_2!(1, 4, 5);
add_impl_2!(1, 5, 6);
add_impl_2!(1, 6, 7);
add_impl_2!(2, 3, 5);
add_impl_2!(2, 4, 6);
add_impl_2!(2, 5, 7);
add_impl_2!(2, 6, 8);

macro_rules! mul_div_impl {
    ($X1:expr, $X2:expr, $Y:expr) => {
        #[cfg(not(feature = "nightly"))]
        impl core::ops::Mul<Const<$X2>> for Const<$X1> {
            type Output = Const<$Y>;
            fn mul(self, _: Const<$X2>) -> Self::Output {
                Const
            }
        }

        #[cfg(not(feature = "nightly"))]
        impl core::ops::Div<Const<$X2>> for Const<$Y> {
            type Output = Const<$X1>;
            fn div(self, _: Const<$X2>) -> Self::Output {
                Const
            }
        }
    };
}

macro_rules! mul_div_impl_2 {
    ($X1:expr, $X2:expr, $Y:expr) => {
        mul_div_impl!($X1, $X2, $Y);
        mul_div_impl!($X2, $X1, $Y);
    };
}

mul_div_impl!(1, 1, 1);
mul_div_impl!(2, 2, 4);
mul_div_impl!(3, 3, 9);
mul_div_impl!(4, 4, 16);
mul_div_impl!(5, 5, 25);
mul_div_impl_2!(1, 2, 2);
mul_div_impl_2!(1, 3, 3);
mul_div_impl_2!(2, 3, 6);
mul_div_impl_2!(1, 4, 4);
mul_div_impl_2!(2, 4, 8);
mul_div_impl_2!(3, 4, 12);
mul_div_impl_2!(1, 5, 5);
mul_div_impl_2!(2, 5, 10);
mul_div_impl_2!(3, 5, 15);
mul_div_impl_2!(4, 5, 20);

/// Represents either `[T; N]` or `Vec<T>`
pub trait Array<T>: IntoIterator<Item = T> {
    type Dim: Dim;
    fn dim(&self) -> Self::Dim;
}
impl<T, const N: usize> Array<T> for [T; N] {
    type Dim = Const<N>;
    fn dim(&self) -> Self::Dim {
        Const
    }
}
impl<T> Array<T> for std::vec::Vec<T> {
    type Dim = usize;
    fn dim(&self) -> Self::Dim {
        self.len()
    }
}

/// A collection of dimensions ([Dim]) that change how a multi-dimensional
/// array is interacted with.
pub trait Shape:
    'static
    + std::fmt::Debug
    + Clone
    + Copy
    + Send
    + Sync
    + Eq
    + PartialEq
    + HasAxes<Self::AllAxes>
    + HasAxes<Self::LastAxis>
    + ReduceShapeTo<(), Self::AllAxes>
    + ReduceShape<Self::LastAxis>
{
    /// The number of dimensions the shape has
    const NUM_DIMS: usize;

    /// Is `[usize; Self::NUM_DIMS]`, but that is not usable yet.
    type Concrete: std::fmt::Debug
        + Clone
        + Copy
        + Default
        + Eq
        + PartialEq
        + std::ops::Index<usize, Output = usize>
        + std::ops::IndexMut<usize>
        + Send
        + Sync
        + IntoIterator<Item = usize>
        + Into<std::vec::Vec<usize>>
        + AsRef<[usize]>;

    /// All the axes of this shape
    type AllAxes: Axes;

    /// The last axis of this shape
    type LastAxis: Axes;

    fn concrete(&self) -> Self::Concrete;
    fn from_concrete(concrete: &Self::Concrete) -> Option<Self>;

    /// The number of elements in this shape; the product of all dimensions.
    #[inline(always)]
    fn num_elements(&self) -> usize {
        self.concrete().into_iter().product()
    }

    /// The strides of how this shape is layed out in memory.
    #[inline(always)]
    fn strides(&self) -> Self::Concrete {
        let sizes = self.concrete();
        let mut strides: Self::Concrete = Default::default();
        strides[Self::NUM_DIMS - 1] = 1;
        for i in (0..(Self::NUM_DIMS - 1)).rev() {
            strides[i] = strides[i + 1] * sizes[i + 1];
        }
        strides
    }
}

/// Represents a [Shape] that has all [ConstDim]s
pub trait ConstShape: Default + Shape {
    const NUMEL: usize;
}

/// Represents something that has a [Shape].
pub trait HasShape {
    type WithShape<New: Shape>: HasShape<Shape = New>;
    type Shape: Shape;
    fn shape(&self) -> &Self::Shape;
}

impl<S: Shape> HasShape for S {
    type WithShape<New: Shape> = New;
    type Shape = Self;
    fn shape(&self) -> &Self::Shape {
        self
    }
}

/// Compile time known shape with 0 dimensions
pub type Rank0 = ();
/// Compile time known shape with 1 dimensions
pub type Rank1<const M: usize> = (Const<M>,);
/// Compile time known shape with 2 dimensions
pub type Rank2<const M: usize, const N: usize> = (Const<M>, Const<N>);
/// Compile time known shape with 3 dimensions
pub type Rank3<const M: usize, const N: usize, const O: usize> = (Const<M>, Const<N>, Const<O>);
/// Compile time known shape with 4 dimensions
pub type Rank4<const M: usize, const N: usize, const O: usize, const P: usize> =
    (Const<M>, Const<N>, Const<O>, Const<P>);
/// Compile time known shape with 5 dimensions
pub type Rank5<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize> =
    (Const<M>, Const<N>, Const<O>, Const<P>, Const<Q>);
#[rustfmt::skip]
/// Compile time known shape with 6 dimensions
pub type Rank6<const M: usize, const N: usize, const O: usize, const P: usize, const Q: usize, const R: usize> =
    (Const<M>, Const<N>, Const<O>, Const<P>, Const<Q>, Const<R>);

macro_rules! shape {
    (($($D:tt $Idx:tt),*), rank=$Num:expr, all=$All:tt) => {
        impl<$($D: Dim, )*> Shape for ($($D, )*) {
            const NUM_DIMS: usize = $Num;
            type Concrete = [usize; $Num];
            type AllAxes = $All<$($Idx,)*>;
            type LastAxis = Axis<{$Num - 1}>;
            #[inline(always)]
            fn concrete(&self) -> Self::Concrete {
                [$(self.$Idx.size(), )*]
            }
            #[inline(always)]
            fn from_concrete(concrete: &Self::Concrete) -> Option<Self> {
                Some(($(Dim::from_size(concrete[$Idx])?, )*))
            }
        }
        impl<$($D: ConstDim, )*> ConstShape for ($($D, )*) {
            const NUMEL: usize = $($D::SIZE * )* 1;
         }

        impl Shape for [usize; $Num] {
            const NUM_DIMS: usize = $Num;
            type Concrete = Self;
            type AllAxes = $All<$($Idx,)*>;
            type LastAxis = Axis<{$Num - 1}>;

            fn concrete(&self) -> Self::Concrete {
                *self
            }

            fn from_concrete(concrete: &Self::Concrete) -> Option<Self> {
                Some(*concrete)
            }
        }
    };
}

impl Shape for () {
    const NUM_DIMS: usize = 0;
    type Concrete = [usize; 0];
    type AllAxes = Axis<0>;
    type LastAxis = Axis<0>;
    #[inline(always)]
    fn concrete(&self) -> Self::Concrete {
        []
    }
    #[inline(always)]
    fn strides(&self) -> Self::Concrete {
        []
    }
    #[inline(always)]
    fn from_concrete(_: &Self::Concrete) -> Option<Self> {
        Some(())
    }
}
impl ConstShape for () {
    const NUMEL: usize = 1;
}

shape!((D1 0), rank=1, all=Axis);
shape!((D1 0, D2 1), rank=2, all=Axes2);
shape!((D1 0, D2 1, D3 2), rank=3, all=Axes3);
shape!((D1 0, D2 1, D3 2, D4 3), rank=4, all=Axes4);
shape!((D1 0, D2 1, D3 2, D4 3, D5 4), rank=5, all=Axes5);
shape!((D1 0, D2 1, D3 2, D4 3, D5 4, D6 5), rank=6, all=Axes6);
