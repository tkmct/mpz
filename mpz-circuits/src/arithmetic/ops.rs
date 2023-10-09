//! This file defines arithmetic operations on field values.
use crate::Feed;

use crate::arithmetic::{
    builder::ArithBuilderState,
    types::{ArithNode, CrtRepr, Fp},
};

/// Add two crt values.
pub fn add<const N: usize>(
    state: &mut ArithBuilderState,
    x: &CrtRepr<N>,
    y: &CrtRepr<N>,
) -> CrtRepr<N> {
    let nodes: [ArithNode<Feed>; N] =
        std::array::from_fn(|i| state.add_add_gate(&x.nodes()[i], &y.nodes()[i]).unwrap());

    CrtRepr::new(nodes)
}

/// Multiply two crt values.
pub fn mul<const N: usize>(
    state: &mut ArithBuilderState,
    x: &CrtRepr<N>,
    y: &CrtRepr<N>,
) -> CrtRepr<N> {
    let nodes: [ArithNode<Feed>; N] =
        std::array::from_fn(|i| state.add_mul_gate(&x.nodes()[i], &y.nodes()[i]).unwrap());

    CrtRepr::new(nodes)
}

/// Multiply crt value by constant value.
pub fn cmul<const N: usize>(state: &mut ArithBuilderState, x: &CrtRepr<N>, c: Fp) -> CrtRepr<N> {
    let nodes: [ArithNode<Feed>; N] =
        std::array::from_fn(|i| state.add_cmul_gate(&x.nodes()[i], c));

    CrtRepr::new(nodes)
}

// TODO: add a good way to add projection gate to circuit

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        arithmetic::{
            builder::ArithmeticCircuitBuilder,
            components::ArithGate,
            types::{ArithNode, CrtRepr, Fp},
            utils::PRIMES,
        },
        Feed, Sink,
    };

    #[test]
    fn test_add() {
        let mut builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_crt_input::<5>();
        let y = builder.add_crt_input::<5>();

        let z = add(&mut builder.state().borrow_mut(), &x, &y);

        let circ = builder.build().unwrap();

        // check z has correct CrtRepr
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

        assert_eq!(circ.feed_count(), 15);
        assert_eq!(circ.add_count(), 5);

        // check if appropriate gates are added to state.
        let gates = circ.gates();
        for (i, (gate, p)) in gates.iter().zip(PRIMES).enumerate() {
            assert_eq!(
                *gate,
                ArithGate::Add {
                    x: ArithNode::<Sink>::new(i, p),
                    y: ArithNode::<Sink>::new(i + 5, p),
                    z: ArithNode::<Feed>::new(i + 10, p),
                }
            );
        }
    }

    #[test]
    fn test_mul() {
        let mut builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_crt_input::<5>();
        let y = builder.add_crt_input::<5>();

        let z = mul(&mut builder.state().borrow_mut(), &x, &y);

        let circ = builder.build().unwrap();

        // check z has correct CrtRepr
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

        assert_eq!(circ.feed_count(), 15);
        assert_eq!(circ.mul_count(), 5);

        // check if appropriate gates are added to state.
        let gates = circ.gates();
        for (i, (gate, p)) in gates.iter().zip(PRIMES).enumerate() {
            assert_eq!(
                *gate,
                ArithGate::Mul {
                    x: ArithNode::<Sink>::new(i, p),
                    y: ArithNode::<Sink>::new(i + 5, p),
                    z: ArithNode::<Feed>::new(i + 10, p),
                }
            );
        }
    }

    #[test]
    fn test_cmul() {
        let mut builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_crt_input::<5>();
        let c = Fp(5);

        let z = cmul(&mut builder.state().borrow_mut(), &x, c);

        let circ = builder.build().unwrap();

        // check z has correct CrtRepr
        assert_eq!(
            z,
            CrtRepr::new([
                ArithNode::<Feed>::new(5, 2),
                ArithNode::<Feed>::new(6, 3),
                ArithNode::<Feed>::new(7, 5),
                ArithNode::<Feed>::new(8, 7),
                ArithNode::<Feed>::new(9, 11),
            ])
        );

        assert_eq!(circ.feed_count(), 10);
        assert_eq!(circ.cmul_count(), 5);

        // check if appropriate gates are added to state.
        let gates = circ.gates();
        for (i, (gate, p)) in gates.iter().zip(PRIMES).enumerate() {
            assert_eq!(
                *gate,
                ArithGate::Cmul {
                    x: ArithNode::<Sink>::new(i, p),
                    c,
                    z: ArithNode::<Feed>::new(i + 5, p),
                }
            );
        }
    }
}
