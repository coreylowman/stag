//! Macros for use with dfdx

// This `extern` is required for older `rustc` versions but newer `rustc`
// versions warn about the unused `extern crate`.
#[allow(unused_extern_crates)]
extern crate proc_macro;

use syn::{parse_macro_input, DeriveInput};

mod can_update_with_gradients;
mod reset_params;

/// Implements CanUpdateWithGradients for a Module
///
/// ```ignore
/// use dfdx::prelude::*;
/// use dfdx_macros::CanUpdateWithGradients;
///
/// #[derive(CanUpdateWithGradients)]
/// pub struct Linear<const I: usize, const O: usize> {
///     // Transposed weight matrix, shape (O, I)
///     pub weight: Tensor2D<O, I>,
///
///     // Bias vector, shape (O, )
///     pub bias: Tensor1D<O>,
/// }
/// ```
#[proc_macro_derive(CanUpdateWithGradients)]
pub fn derive_can_update_with_gradients(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let gen = can_update_with_gradients::gen(ast);
    gen.into()
}

/// Implements ResetParams for a Module
///
/// ```ignore
/// use dfdx::prelude::*;
/// use dfdx_macros::ResetParams;
///
/// #[derive(ResetParams)]
/// pub struct Linear<const I: usize, const O: usize> {
///     // Transposed weight matrix, shape (O, I)
///     pub weight: Tensor2D<O, I>,
///
///     // Bias vector, shape (O, )
///     pub bias: Tensor1D<O>,
/// }
/// ```
#[proc_macro_derive(ResetParams, attributes(reset_params))]
pub fn derive_reset_params(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);

    let gen = reset_params::gen(ast);
    gen.into()
}
