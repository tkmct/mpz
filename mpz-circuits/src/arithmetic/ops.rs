//! This file defines arithmetic operations on field values.
use crate::Feed;

use crate::arithmetic::{
    builder::ArithBuilderState,
    circuit::ArithCircuitError,
    types::{ArithNode, CrtRepr, TypeError},
    utils::is_same_crt_len,
};

use super::types::{CrtValue, MixedRadixValue};

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

/// Sum up every given ArithNode
pub fn add_many(
    state: &mut ArithBuilderState,
    xs: &[ArithNode<Feed>],
) -> Result<ArithNode<Feed>, ArithCircuitError> {
    if xs.len() < 2 {
        return Err(ArithCircuitError::InvalidLength(xs.len()));
    }
    let (head, tail) = xs.split_at(1);

    tail.iter()
        .try_fold(head[0], |acc, x| state.add_add_gate(&acc, &x))
}

/// Change modulus of an ArithNode to `to_modulus`
pub fn mod_change(
    state: &mut ArithBuilderState,
    x: &ArithNode<Feed>,
    to_modulus: u16,
) -> Result<ArithNode<Feed>, ArithCircuitError> {
    let from_modulus = x.modulus();
    if from_modulus == to_modulus {
        return Ok(x.clone());
    }
    let tab = (0..from_modulus)
        .map(|x| x % to_modulus)
        .collect::<Vec<_>>();
    state.add_proj_gate(x, to_modulus, tab)
}

/// Mixed radix addition returning MSB.
/// This method is used by crt_fractional_mixed_radix
pub fn mixed_radix_add_msb<const N: usize>(
    state: &mut ArithBuilderState,
    xs: &[MixedRadixValue<N>],
) -> Result<ArithNode<Feed>, ArithCircuitError> {
    // check moduli of all mixed radix value
    let len = xs.len();
    if len < 1 {
        return Err(ArithCircuitError::InvalidLength(len));
    }

    let wire_len = xs[0].len();

    let mut opt_carry = None;
    let mut max_carry = 0;

    for i in 0..wire_len - 1 {
        // all the ith digits, in one vec
        let ds = xs.iter().map(|x| x.wires()[i]).collect::<Vec<_>>();
        // compute the carry
        let q = xs[0].moduli()[i];
        // max_carry currently contains the max carry from the previous iteration
        let max_val = len as u16 * (q - 1) + max_carry;
        // now it is the max carry of this iteration
        max_carry = max_val / q;

        // mod change the digits to the max sum possible plus the max carry of the
        // previous iteration
        let modded_ds = ds
            .iter()
            .map(|d| mod_change(state, d, max_val + 1))
            .collect::<Result<Vec<ArithNode<Feed>>, ArithCircuitError>>()?;
        // add them up
        let sum = add_many(state, &modded_ds)?;
        // add in the carry
        let sum_with_carry = opt_carry
            .as_ref()
            .map_or(Ok(sum.clone()), |c| state.add_add_gate(&sum, &c))?;

        // carry now contains the carry information, we just have to project it to
        // the correct moduli for the next iteration. It will either be used to
        // compute the next carry, if i < n-2, or it will be used to compute the
        // output MSB, in which case it should be the modulus of the SB
        let next_mod = if i < wire_len - 2 {
            len as u16 * (xs[0].moduli()[i + 1] - 1) + max_carry + 1
        } else {
            xs[0].moduli()[i + 1] // we will be adding the carry to the MSB
        };

        let tt = (0..=max_val)
            .map(|i| (i / q) % next_mod)
            .collect::<Vec<_>>();
        opt_carry = Some(state.add_proj_gate(&sum_with_carry, next_mod, tt)?);
    }
    // compute the msb
    let ds = xs
        .iter()
        .map(|x| x.wires()[wire_len - 1].clone())
        .collect::<Vec<_>>();
    let digit_sum = add_many(state, &ds)?;
    opt_carry.as_ref().map_or(Ok(digit_sum.clone()), |d| {
        state.add_add_gate(&digit_sum, &d)
    })
}

/// Fractional mixed radix
pub fn crt_fractional_mixed_radix(
    state: &mut ArithBuilderState,
    x: &CrtRepr,
    ms: &[u16],
) -> Result<ArithNode<Feed>, ArithCircuitError> {
    let ndigits = ms.len();

    let q = x.moduli().iter().fold(1, |acc, &x| acc * x as u128);
    let M = ms.iter().fold(1, |acc, &x| acc * x as u128);

    let mut ds = Vec::new();

    todo!()
}

// pub fn crt_sign(state: &mut ArithBuilderState, x: &CrtRepr, accuracy: &str) -> Crt
//
// ///
// /// Return 0 if x is positive and 1 if x is negative
// pub fn crt_sgn(state: &mut ArithBuilderState, x: &CrtRepr, accuracy: &str) -> CrtRepr {
//     let sign = crt_sign(x, accuracy);
//     todo!()
// }
//
//
// /// Compare two crt representation wires.
// /// Returns 1 if x < y.
// pub fn crt_lt(state: &mut ArithBuilderState, x: &CrtRepr, y: &CrtRepr) -> CrtRepr {
//     let z = self.crt_sub(x,y)?;
//     self.crt_sign(&z, accuracy)
//     todo!()
// }

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
        let x = builder.add_input::<u32>().unwrap();
        let y = builder.add_input::<u32>().unwrap();

        let z = add(&mut builder.state().borrow_mut(), &x, &y).unwrap();

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
        let x = builder.add_input::<u32>().unwrap();
        let y = builder.add_input::<u32>().unwrap();

        let z = mul(&mut builder.state().borrow_mut(), &x, &y).unwrap();

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
        let x = builder.add_input::<u32>().unwrap();
        let c = 5;

        let z = cmul(&mut builder.state().borrow_mut(), &x, c);

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
    #[ignore]
    fn test_add_many() {}

    #[test]
    #[ignore]
    fn test_mod_change() {}
}
