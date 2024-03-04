//! This file defines arithmetic operations on field values.
use crate::Feed;

use crate::arithmetic::{
    builder::ArithBuilderState,
    circuit::ArithCircuitError,
    types::{ArithNode, CrtRepr, TypeError},
    utils::is_same_crt_len,
};

use super::types::CrtValue;

/// Add two crt values.
/// Add gates for each ArithNode and returns new CrtRepr
pub fn add(
    state: &mut ArithBuilderState,
    x: &CrtRepr,
    y: &CrtRepr,
) -> Result<CrtRepr, ArithCircuitError> {
    if !is_same_crt_len(x, y) {
        return Err(TypeError::UnequalLength(x.len(), y.len()).into());
    }

    let repr = match (x, y) {
        (CrtRepr::Bool(xval), CrtRepr::Bool(yval)) => {
            let z = state.add_add_gate(&xval.nodes()[0], &yval.nodes()[0])?;
            CrtRepr::Bool(CrtValue::<1>::new([z]))
        }
        (CrtRepr::U32(xval), CrtRepr::U32(yval)) => {
            let zval: [ArithNode<Feed>; 10] = std::array::from_fn(|i| {
                state
                    .add_add_gate(&xval.nodes()[i], &yval.nodes()[i])
                    .unwrap() // TODO: handle error correctly!
            });
            CrtRepr::U32(CrtValue::<10>::new(zval))
        }
        _ => {
            // This should not happen
            return Err(TypeError::UnequalLength(x.len(), y.len()).into());
        }
    };

    Ok(repr)
}

/// Multiply two crt values.
/// Multiply gates for each ArithNode and returns new CrtRepr
pub fn mul(
    state: &mut ArithBuilderState,
    x: &CrtRepr,
    y: &CrtRepr,
) -> Result<CrtRepr, ArithCircuitError> {
    if !is_same_crt_len(x, y) {
        return Err(TypeError::UnequalLength(x.len(), y.len()).into());
    }

    let repr = match (x, y) {
        (CrtRepr::Bool(xval), CrtRepr::Bool(yval)) => {
            let z = state.add_mul_gate(&xval.nodes()[0], &yval.nodes()[0])?;
            CrtRepr::Bool(CrtValue::<1>::new([z]))
        }
        (CrtRepr::U32(xval), CrtRepr::U32(yval)) => {
            let zval: [ArithNode<Feed>; 10] = std::array::from_fn(|i| {
                state
                    .add_mul_gate(&xval.nodes()[i], &yval.nodes()[i])
                    .unwrap() // TODO: handle error correctly!
            });
            CrtRepr::U32(CrtValue::<10>::new(zval))
        }
        _ => {
            // This should not happen
            return Err(TypeError::UnequalLength(x.len(), y.len()).into());
        }
    };

    Ok(repr)
}

/// Multiply crt value by constant value.
pub fn cmul(state: &mut ArithBuilderState, x: &CrtRepr, c: u32) -> CrtRepr {
    match x {
        CrtRepr::Bool(xval) => {
            let z = state.add_cmul_gate(&xval.nodes()[0], c);
            CrtRepr::Bool(CrtValue::<1>::new([z]))
        }
        CrtRepr::U32(xval) => {
            let zval: [ArithNode<Feed>; 10] =
                std::array::from_fn(|i| state.add_cmul_gate(&xval.nodes()[i], c));
            CrtRepr::U32(CrtValue::<10>::new(zval))
        }
    }
}

// TODO: add a good way to add projection gate to circuit

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        arithmetic::{
            builder::ArithmeticCircuitBuilder,
            components::ArithGate,
            types::{ArithNode, CrtRepr},
            utils::PRIMES,
        },
        Feed, Sink,
    };

    #[test]
    fn test_add() {
        let builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_input::<u32>("x".into()).unwrap();
        let y = builder.add_input::<u32>("y".into()).unwrap();

        let z = add(&mut builder.state().borrow_mut(), &x.repr, &y.repr).unwrap();

        let circ = builder.build().unwrap();

        // check z has correct CrtRepr
        assert_eq!(
            z,
            CrtRepr::U32(CrtValue::new(std::array::from_fn(|i| {
                ArithNode::<Feed>::new(20 + i, PRIMES[i])
            })))
        );

        assert_eq!(circ.feed_count(), 30);
        assert_eq!(circ.add_count(), 10);

        // check if appropriate gates are added to state.
        let gates = circ.gates();
        for (i, (gate, p)) in gates.iter().zip(PRIMES).enumerate() {
            assert_eq!(
                *gate,
                ArithGate::Add {
                    x: ArithNode::<Sink>::new(i, p),
                    y: ArithNode::<Sink>::new(i + 10, p),
                    z: ArithNode::<Feed>::new(i + 20, p),
                }
            );
        }
    }

    #[test]
    fn test_mul() {
        let builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_input::<u32>("x".into()).unwrap();
        let y = builder.add_input::<u32>("y".into()).unwrap();

        let z = mul(&mut builder.state().borrow_mut(), &x.repr, &y.repr).unwrap();

        let circ = builder.build().unwrap();
        circ.print_gates();
        // dbg!(circ.clone());

        // check z has correct CrtRepr
        //
        assert_eq!(
            z,
            CrtRepr::U32(CrtValue::new(std::array::from_fn(|i| {
                ArithNode::<Feed>::new(20 + i, PRIMES[i])
            })))
        );

        assert_eq!(circ.feed_count(), 30);
        assert_eq!(circ.mul_count(), 10);

        // check if appropriate gates are added to state.
        let gates = circ.gates();
        for (i, (gate, p)) in gates.iter().zip(PRIMES).enumerate() {
            assert_eq!(
                *gate,
                ArithGate::Mul {
                    x: ArithNode::<Sink>::new(i, p),
                    y: ArithNode::<Sink>::new(i + 10, p),
                    z: ArithNode::<Feed>::new(i + 20, p),
                }
            );
        }
    }

    #[test]
    fn test_cmul() {
        let builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_input::<u32>("x".into()).unwrap();
        let c = 5;

        let z = cmul(&mut builder.state().borrow_mut(), &x.repr, c);

        let circ = builder.build().unwrap();

        // check z has correct CrtRepr
        assert_eq!(
            z,
            CrtRepr::U32(CrtValue::new(std::array::from_fn(|i| {
                ArithNode::<Feed>::new(10 + i, PRIMES[i])
            })))
        );

        assert_eq!(circ.feed_count(), 20);
        assert_eq!(circ.cmul_count(), 10);

        // check if appropriate gates are added to state.
        let gates = circ.gates();
        for (i, (gate, p)) in gates.iter().zip(PRIMES).enumerate() {
            assert_eq!(
                *gate,
                ArithGate::Cmul {
                    x: ArithNode::<Sink>::new(i, p),
                    c,
                    z: ArithNode::<Feed>::new(i + 10, p),
                }
            );
        }
    }

    #[test]
    fn test_conversion() {
        let builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_input::<u32>("x".into()).unwrap();
        let c = 5;

        let z = cmul(&mut builder.state().borrow_mut(), &x.repr, c);

        let circ = builder.build().unwrap();

        // check z has correct CrtRepr
        assert_eq!(
            z,
            CrtRepr::U32(CrtValue::new(std::array::from_fn(|i| {
                ArithNode::<Feed>::new(10 + i, PRIMES[i])
            })))
        );

        assert_eq!(circ.feed_count(), 20);
        assert_eq!(circ.cmul_count(), 10);

        // check if appropriate gates are added to state.
        let gates = circ.gates();
        for (i, (gate, p)) in gates.iter().zip(PRIMES).enumerate() {
            assert_eq!(
                *gate,
                ArithGate::Cmul {
                    x: ArithNode::<Sink>::new(i, p),
                    c,
                    z: ArithNode::<Feed>::new(i + 10, p),
                }
            );
        }
    }
}
