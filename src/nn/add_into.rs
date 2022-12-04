use crate::{
    arrays::Dtype,
    optim::{CanUpdateWithGradients, UnusedTensors, UpdateParams},
    tensor_ops::Device,
};

use super::{BuildModule, Module, ModuleMut};

/// Add inputs together into a single tensor. `T` should be a tuple
//// where every element of the tuple has the same output type
///
/// This provides a utility for networks where multiple inputs are needed
///
/// # Generics
/// - `T` the module to add the outputs together of
///
/// # Examples
/// ```rust
/// # use dfdx::prelude::*;
/// type Model = AddInto<(Linear<2, 5>, Linear<3, 5>)>;
/// let model: Model = Default::default();
/// let _: Tensor1D<5> = model.forward((Tensor1D::<2>::zeros(), Tensor1D::<3>::zeros()));
/// ```
#[derive(Debug, Default, Clone)]
pub struct AddInto<T>(pub T);

impl<T: CanUpdateWithGradients<D, E>, D: Device<E>, E: Dtype> CanUpdateWithGradients<D, E>
    for AddInto<T>
{
    fn update<U>(&mut self, updater: &mut U, unused: &mut UnusedTensors) -> Result<(), <D>::Err>
    where
        U: UpdateParams<D, E>,
    {
        self.0.update(updater, unused)
    }
}

impl<T: BuildModule<D, E>, D: Device<E>, E: Dtype> BuildModule<D, E> for AddInto<T> {
    fn try_build(device: &D) -> Result<Self, <D>::Err> {
        Ok(Self(BuildModule::try_build(device)?))
    }
    fn try_reset_params(&mut self) -> Result<(), <D>::Err> {
        self.0.try_reset_params()
    }
}

macro_rules! sum {
    ($H:tt) => { $H };
    ($H:tt, $($T:tt),+) => { $H + sum!($($T),+) };
}

macro_rules! add_into_impls {
    ($([$Mod:tt $ModVar:tt $Inp:tt $InpVar:tt]),+) => {
        impl<
            Out: std::ops::Add<Out, Output = Out>,
            $($Inp, )+
            $($Mod: Module<$Inp, Output = Out>, )+
        > Module<($($Inp, )+)> for AddInto<($($Mod, )+)> {
            type Output = Out;
            fn forward(&self, x: ($($Inp, )+)) -> Self::Output {
                let ($($ModVar, )+) = &self.0;
                let ($($InpVar, )+) = x;
                $(let $InpVar = $ModVar.forward($InpVar);)+
                sum!($($InpVar),*)
            }
        }
        impl<
            Out: std::ops::Add<Out, Output = Out>,
            $($Inp, )+
            $($Mod: ModuleMut<$Inp, Output = Out>, )+
        > ModuleMut<($($Inp, )+)> for AddInto<($($Mod, )+)> {
            type Output = Out;
            fn forward_mut(&mut self, x: ($($Inp, )+)) -> Self::Output {
                let ($($ModVar, )+) = &mut self.0;
                let ($($InpVar, )+) = x;
                $(let $InpVar = $ModVar.forward_mut($InpVar);)+
                sum!($($InpVar),*)
            }
        }
    };
}

add_into_impls!([A a Ai a_i], [B b Bi b_i]);
add_into_impls!([A a Ai a_i], [B b Bi b_i], [C c Ci c_i]);
add_into_impls!([A a Ai a_i], [B b Bi b_i], [C c Ci c_i], [D d Di d_i]);
add_into_impls!([A a Ai a_i], [B b Bi b_i], [C c Ci c_i], [D d Di d_i], [E e Ei e_i]);
add_into_impls!([A a Ai a_i], [B b Bi b_i], [C c Ci c_i], [D d Di d_i], [E e Ei e_i], [F f Fi f_i]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        arrays::*,
        gradients::OwnedTape,
        nn::{tests::SimpleUpdater, Linear, ReLU},
        tensor::*,
        tests::build_test_device,
        unique_id::HasUniqueId,
    };

    #[test]
    fn test_add_into_2() {
        let dev = build_test_device!();
        let m: AddInto<(Linear<2, 5, _>, Linear<3, 5, _>)> = BuildModule::build(&dev);
        let _: Tensor1D<5, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank1<2>>().traced(),
            dev.zeros::<Rank1<3>>().traced(),
        ));
        let _: Tensor2D<3, 5, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().traced(),
            dev.zeros::<Rank2<3, 3>>().traced(),
        ));
    }

    #[test]
    fn test_add_into_3() {
        let dev = build_test_device!();
        let m: AddInto<(Linear<2, 5>, Linear<3, 5>, Linear<4, 5>)> = BuildModule::build(&dev);
        let _: Tensor1D<5, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank1<2>>().traced(),
            dev.zeros::<Rank1<3>>().traced(),
            dev.zeros::<Rank1<4>>().traced(),
        ));
        let _: Tensor2D<3, 5, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().traced(),
            dev.zeros::<Rank2<3, 3>>().traced(),
            dev.zeros::<Rank2<3, 4>>().traced(),
        ));
    }

    #[test]
    fn test_add_into_4() {
        let dev = build_test_device!();
        type Model = AddInto<(Linear<2, 5>, Linear<3, 5>, Linear<4, 5>, Linear<5, 5>)>;
        let m: Model = BuildModule::build(&dev);
        let _: Tensor1D<5, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank1<2>>().traced(),
            dev.zeros::<Rank1<3>>().traced(),
            dev.zeros::<Rank1<4>>().traced(),
            dev.zeros::<Rank1<5>>().traced(),
        ));
        let _: Tensor2D<3, 5, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().traced(),
            dev.zeros::<Rank2<3, 3>>().traced(),
            dev.zeros::<Rank2<3, 4>>().traced(),
            dev.zeros::<Rank2<3, 5>>().traced(),
        ));
    }

    #[test]
    fn test_add_into_5() {
        let dev = build_test_device!();
        type Model = AddInto<(
            Linear<2, 5>,
            Linear<3, 5>,
            Linear<4, 5>,
            Linear<5, 5>,
            Linear<6, 5>,
        )>;
        let m: Model = BuildModule::build(&dev);
        let _: Tensor1D<5, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank1<2>>().traced(),
            dev.zeros::<Rank1<3>>().traced(),
            dev.zeros::<Rank1<4>>().traced(),
            dev.zeros::<Rank1<5>>().traced(),
            dev.zeros::<Rank1<6>>().traced(),
        ));
        let _: Tensor2D<3, 5, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().traced(),
            dev.zeros::<Rank2<3, 3>>().traced(),
            dev.zeros::<Rank2<3, 4>>().traced(),
            dev.zeros::<Rank2<3, 5>>().traced(),
            dev.zeros::<Rank2<3, 6>>().traced(),
        ));
    }

    #[test]
    fn test_add_into_6() {
        let dev = build_test_device!();
        type Model = AddInto<(
            Linear<2, 5>,
            Linear<3, 5>,
            Linear<4, 5>,
            Linear<5, 5>,
            Linear<6, 5>,
            Linear<7, 5>,
        )>;
        let m: Model = BuildModule::build(&dev);
        let _: Tensor1D<5, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank1<2>>().traced(),
            dev.zeros::<Rank1<3>>().traced(),
            dev.zeros::<Rank1<4>>().traced(),
            dev.zeros::<Rank1<5>>().traced(),
            dev.zeros::<Rank1<6>>().traced(),
            dev.zeros::<Rank1<7>>().traced(),
        ));
        let _: Tensor2D<3, 5, _, OwnedTape<_>> = m.forward((
            dev.zeros::<Rank2<3, 2>>().traced(),
            dev.zeros::<Rank2<3, 3>>().traced(),
            dev.zeros::<Rank2<3, 4>>().traced(),
            dev.zeros::<Rank2<3, 5>>().traced(),
            dev.zeros::<Rank2<3, 6>>().traced(),
            dev.zeros::<Rank2<3, 7>>().traced(),
        ));
    }

    #[test]
    fn test_missing_gradients() {
        let dev = build_test_device!();
        let mut model: AddInto<(Linear<5, 3, _>, Linear<5, 3, _>)> = BuildModule::build(&dev);
        let mut g: SimpleUpdater<_> = Default::default();

        // no gradients present
        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert_eq!(
            &unused.ids,
            &[
                *model.0 .0.weight.id(),
                *model.0 .0.bias.id(),
                *model.0 .1.weight.id(),
                *model.0 .1.bias.id()
            ]
        );

        // weight gradient is present
        g.0.get_mut(&model.0 .0.weight).unwrap();
        g.0.get_mut(&model.0 .0.bias).unwrap();
        g.0.get_mut(&model.0 .1.weight).unwrap();
        g.0.get_mut(&model.0 .1.bias).unwrap();

        let mut unused = Default::default();
        model.update(&mut g, &mut unused).unwrap();
        assert!(unused.is_empty());
    }

    #[test]
    fn longer_network() {
        let dev = build_test_device!();
        // check if it works in a longer neural net
        let mut model: (
            AddInto<(Linear<5, 3, _>, Linear<5, 3, _>)>,
            ReLU,
            Linear<3, 1, _>,
        ) = BuildModule::build(&dev);
        let _: Tensor<Rank1<1>, _, _, OwnedTape<_>> = model.forward((
            dev.zeros::<Rank1<5>>().traced(),
            dev.zeros::<Rank1<5>>().traced(),
        ));
        let _: Tensor<Rank2<5, 1>, _, _, OwnedTape<_>> = model.forward((
            dev.zeros::<Rank2<5, 5>>().traced(),
            dev.zeros::<Rank2<5, 5>>().traced(),
        ));
        let _: Tensor<Rank1<1>, _, _, OwnedTape<_>> = model.forward_mut((
            dev.zeros::<Rank1<5>>().traced(),
            dev.zeros::<Rank1<5>>().traced(),
        ));
        let _: Tensor<Rank2<5, 1>, _, _, OwnedTape<_>> = model.forward_mut((
            dev.zeros::<Rank2<5, 5>>().traced(),
            dev.zeros::<Rank2<5, 5>>().traced(),
        ));
    }
}
