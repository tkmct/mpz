//! This file defines arithmetic operations on field values.

use crate::arithmetic::{
    builder::ArithBuilderState,
    types::{CrtRepr, Fp},
};

/// Add two crt values.
pub fn add<const N: usize>(
    state: &mut ArithBuilderState,
    x: &CrtRepr<N>,
    y: &CrtRepr<N>,
) -> CrtRepr<N> {
    todo!()
}

/// Multiply two crt values.
pub fn mul<const N: usize>(
    state: &mut ArithBuilderState,
    x: &CrtRepr<N>,
    y: &CrtRepr<N>,
) -> CrtRepr<N> {
    todo!()
}

/// Multiply crt value by constant value.
pub fn cmul<const N: usize>(state: &mut ArithBuilderState, x: &CrtRepr<N>, c: Fp) -> CrtRepr<N> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arithmetic::{
        builder::ArithBuilderState,
        types::{ArithNode, CrtRepr, Fp},
    };

    #[test]
    fn test_add() {
        let state = ArithBuilderState {};
        let x = CrtRepr::<5>::new_from_id(0);
        let y = CrtRepr::<5>::new_from_id(5);
    }
}
