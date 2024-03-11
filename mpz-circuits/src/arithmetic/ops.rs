//! This file defines arithmetic operations on field values.
use crate::Feed;

use crate::arithmetic::{
    builder::ArithBuilderState,
    circuit::ArithCircuitError,
    types::{ArithNode, CrtRepr, TypeError},
    utils::{as_mixed_radix, inv, is_same_crt_len},
};

use super::types::{CrtValue, MixedRadixValue};
use super::utils::PRIMES;

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

/// Subtraction
pub fn sub(
    state: &mut ArithBuilderState,
    x: &CrtRepr,
    y: &CrtRepr,
) -> Result<CrtRepr, ArithCircuitError> {
    if !is_same_crt_len(x, y) {
        return Err(TypeError::UnequalLength(x.len(), y.len()).into());
    }

    let repr = match (x, y) {
        (CrtRepr::Bool(xval), CrtRepr::Bool(yval)) => {
            let z = state.add_sub_gate(&xval.nodes()[0], &yval.nodes()[0])?;
            CrtRepr::Bool(CrtValue::<1>::new([z]))
        }
        (CrtRepr::U32(xval), CrtRepr::U32(yval)) => {
            let zval: [ArithNode<Feed>; 10] = std::array::from_fn(|i| {
                state
                    .add_sub_gate(&xval.nodes()[i], &yval.nodes()[i])
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
pub fn cmul(state: &mut ArithBuilderState, x: &CrtRepr, c: u64) -> CrtRepr {
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
pub fn mixed_radix_add_msb(
    state: &mut ArithBuilderState,
    xs: &[MixedRadixValue],
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
    crt: &CrtRepr,
    ms: &[u16],
) -> Result<ArithNode<Feed>, ArithCircuitError> {
    let ndigits = ms.len();

    let q = crt.moduli().iter().fold(1, |acc, &x| acc * x as u128);
    let m = ms.iter().fold(1, |acc, &x| acc * x as u128);

    let mut ds = Vec::new();

    for wire in crt.iter() {
        let p = wire.modulus();

        let mut tabs = vec![Vec::with_capacity(p as usize); ndigits];

        for x in 0..p {
            let crt_coef = inv(((q / p as u128) % p as u128) as i128, p as i128);
            let y = (m as f64 * x as f64 * crt_coef as f64 / p as f64).round() as u128 % m;
            let digits = as_mixed_radix(y, ms);
            for i in 0..ndigits {
                tabs[i].push(digits[i]);
            }
        }

        let new_ds = tabs
            .into_iter()
            .enumerate()
            .map(|(i, tt)| state.add_proj_gate(wire, ms[i], tt))
            .collect::<Result<Vec<ArithNode<Feed>>, ArithCircuitError>>()?;

        ds.push(MixedRadixValue::new(new_ds));
    }
    // println!("ds to be added: {:?}", ds);

    mixed_radix_add_msb(state, &ds)
}

/// Returns 0 if x is positive, 1 if x is negative
/// Returning ArithNode has mod 2
pub fn crt_sign_inner(
    state: &mut ArithBuilderState,
    x: &CrtRepr,
    accuracy: &str,
) -> Result<ArithNode<Feed>, ArithCircuitError> {
    let factors_of_m = &get_ms(x.moduli().len(), accuracy);
    let res = crt_fractional_mixed_radix(state, x, factors_of_m)?;
    let p = *factors_of_m.last().unwrap();
    let tt = (0..p).map(|x| (x >= p / 2) as u16).collect::<Vec<_>>();
    state.add_proj_gate(&res, 2, tt)
}

/// Returns 0 if x is positive, 1 if x is negative with given number of Wires
pub fn crt_sign<const N: usize>(
    state: &mut ArithBuilderState,
    x: &CrtRepr,
    accuracy: &str,
) -> Result<CrtRepr, ArithCircuitError> {
    let sign = crt_sign_inner(state, x, accuracy)?;
    match N {
        1 => Ok(CrtRepr::Bool(CrtValue::new([sign]))),
        10 => {
            let v: [ArithNode<Feed>; 10] = std::array::from_fn(|i| {
                let p = PRIMES[i];
                let tt = vec![0, 1];
                state.add_proj_gate(&sign, p, tt).unwrap()
            });

            Ok(CrtRepr::U32(CrtValue::new(v)))
        }
        _ => Err(ArithCircuitError::InvalidModuliLen(N)),
    }
}

/// Return 1 if x is positive and -1 if x is negative. -1 is represented as P-1.
/// N is a number of moduli (=wires) used to represent resulting value
pub fn crt_sgn<const N: usize>(
    state: &mut ArithBuilderState,
    x: &CrtRepr,
    accuracy: &str,
) -> Result<CrtRepr, ArithCircuitError> {
    let sign = crt_sign_inner(state, x, accuracy)?;
    match N {
        1 => {
            let v: [ArithNode<Feed>; 1] = std::array::from_fn(|i| {
                let p = PRIMES[i];
                let tt = vec![1, p - 1];
                state.add_proj_gate(&sign, p, tt).unwrap()
            });

            Ok(CrtRepr::Bool(CrtValue::new(v)))
        }
        10 => {
            let v: [ArithNode<Feed>; 10] = std::array::from_fn(|i| {
                let p = PRIMES[i];
                let tt = vec![1, p - 1];
                state.add_proj_gate(&sign, p, tt).unwrap()
            });

            Ok(CrtRepr::U32(CrtValue::new(v)))
        }
        _ => Err(ArithCircuitError::InvalidModuliLen(N)),
    }
}

/// Compute the `ms` needed for the number of CRT primes in `x`, with accuracy
/// `accuracy`.
///
/// Supported accuracy: ["100%", "99.9%", "99%"]
fn get_ms(len: usize, accuracy: &str) -> Vec<u16> {
    match accuracy {
        "100%" => match len {
            3 => vec![2; 5],
            4 => vec![3, 26],
            5 => vec![3, 4, 54],
            6 => vec![5, 5, 5, 60],
            7 => vec![5, 6, 6, 7, 86],
            8 => vec![5, 7, 8, 8, 9, 98],
            9 => vec![5, 5, 7, 7, 7, 7, 7, 76],
            10 => vec![5, 5, 6, 6, 6, 6, 11, 11, 202],
            11 => vec![5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 150],
            n => panic!("unknown exact Ms for {} primes!", n),
        },
        "99.999%" => match len {
            8 => vec![5, 5, 6, 7, 102],
            9 => vec![5, 5, 6, 7, 114],
            10 => vec![5, 6, 6, 7, 102],
            11 => vec![5, 5, 6, 7, 130],
            n => panic!("unknown 99.999% accurate Ms for {} primes!", n),
        },
        "99.99%" => match len {
            6 => vec![5, 5, 5, 42],
            7 => vec![4, 5, 6, 88],
            8 => vec![4, 5, 7, 78],
            9 => vec![5, 5, 6, 84],
            10 => vec![4, 5, 6, 112],
            11 => vec![7, 11, 174],
            n => panic!("unknown 99.99% accurate Ms for {} primes!", n),
        },
        "99.9%" => match len {
            5 => vec![3, 5, 30],
            6 => vec![4, 5, 48],
            7 => vec![4, 5, 60],
            8 => vec![3, 5, 78],
            9 => vec![9, 140],
            10 => vec![7, 190],
            n => panic!("unknown 99.9% accurate Ms for {} primes!", n),
        },
        "99%" => match len {
            4 => vec![3, 18],
            5 => vec![3, 36],
            6 => vec![3, 40],
            7 => vec![3, 40],
            8 => vec![126],
            9 => vec![138],
            10 => vec![140],
            n => panic!("unknown 99% accurate Ms for {} primes!", n),
        },
        _ => panic!("get_ms: unsupported accuracy {}", accuracy),
    }
}

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
