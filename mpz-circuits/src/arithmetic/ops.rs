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
    let z = x.nodes().iter().zip(y.nodes()).map(|(n_x, n_y)| {
        // Two CrtReprs having same const N have same modulus in each node.
        let n_z = state.add_add_gate(n_x, &n_y).unwrap();
    });

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

    use crate::{
        arithmetic::{
            builder::ArithBuilderState,
            types::{ArithNode, CrtRepr, Fp},
        },
        Feed,
    };

    #[test]
    fn test_add() {
        let mut state = ArithBuilderState::default();
        let x = CrtRepr::<5>::new_from_id(0);
        let y = CrtRepr::<5>::new_from_id(5);

        let z = add(&mut state, &x, &y);

        // check z and check builder state
        assert_eq!(
            z,
            CrtRepr::new([
                ArithNode::<Feed>::new(10, 2),
                ArithNode::<Feed>::new(11, 3),
                ArithNode::<Feed>::new(12, 5),
                ArithNode::<Feed>::new(13, 7),
                ArithNode::<Feed>::new(14, 11),
            ])
        );
    }
}
