use crate::prelude::*;

/// `t.nans_to(value)`. Replaces any nans in `t` with `value`.
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// let t: Tensor1D<4> = Tensor1D::new([1.0, f32::NAN, f32::NAN, 4.0]);
/// let r = t.nans_to(0.0);
/// assert_eq!(r.data(), &[1.0, 0.0, 0.0, 4.0]);
/// ```
pub fn nans_to<T: Tensor<Dtype = f32>>(t: T, value: T::Dtype) -> T {
    map(
        t,
        move |x| x.is_nan().then_some(value).unwrap_or(*x),
        move |x| x.is_nan().then_some(0.0).unwrap_or(1.0),
    )
}

macro_rules! tensor_impl {
    ($typename:ident, [$($Vs:tt),*]) => {
impl<$(const $Vs: usize, )* H: Tape> $typename<$($Vs, )* H> {
    /// Calls [nans_to()] on `self`.
    pub fn nans_to(self, value: f32) -> Self {
        nans_to(self, value)
    }
}
    };
}

tensor_impl!(Tensor0D, []);
tensor_impl!(Tensor1D, [M]);
tensor_impl!(Tensor2D, [M, N]);
tensor_impl!(Tensor3D, [M, N, O]);
tensor_impl!(Tensor4D, [M, N, O, P]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nans_0d() {
        let t = Tensor0D::new(f32::NAN);
        let r = t.trace().nans_to(0.0);
        assert_eq!(r.data(), &0.0);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&t), &0.0);
    }

    #[test]
    fn test_nans_1d() {
        let t: Tensor1D<4> = Tensor1D::new([1.0, f32::NAN, f32::NAN, 4.0]);
        let r = t.trace().nans_to(0.0);
        assert_eq!(r.data(), &[1.0, 0.0, 0.0, 4.0]);
        // NOTE: .exp() so we cover case where nans_to() needs to use result grad
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[0.67957044, 0.0, 0.0, 13.649537]
        );
    }

    #[test]
    fn test_nans_2d() {
        let t: Tensor2D<2, 3> = Tensor2D::new([[1.0, f32::NAN, 3.0], [f32::NAN, 4.0, f32::NAN]]);
        let r = t.trace().nans_to(0.0);
        assert_eq!(r.data(), &[[1.0, 0.0, 3.0], [0.0, 4.0, 0.0]]);
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&t),
            &[[1.0 / 6.0, 0.0, 1.0 / 6.0], [0.0, 1.0 / 6.0, 0.0]]
        );
    }
}
