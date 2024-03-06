//! ArithmeticCircuit utils
//! most codes are from swanky
use std::mem::discriminant;

use crate::arithmetic::{
    circuit::ArithCircuitError,
    types::{CrtLen, CrtRepr, TypeError},
};

use super::types::ArithValue;

/// Number of primes supported by our library.
pub const NPRIMES: usize = 29;

/// Primes used in fancy garbling.
pub const PRIMES: [u16; 29] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109,
];

/// Returns the index of the given prime in ascending order.
/// ex. 2 => 0, 7 => 3, ...
pub fn get_index_of_prime(m: u16) -> Option<usize> {
    PRIMES.iter().position(|q| *q == m)
}

/// Check if crt type is equal
pub(crate) fn is_same_crt_len(a: &CrtRepr, b: &CrtRepr) -> bool {
    discriminant(a) == discriminant(b)
}

/// Convert `x` into mixed radix form using the provided `radii`.
pub(crate) fn as_mixed_radix(x: u128, radii: &[u16]) -> Vec<u16> {
    let mut x = x;
    radii
        .iter()
        .map(|&m| {
            if x >= m as u128 {
                let d = x % m as u128;
                x = (x - d) / m as u128;
                d as u16
            } else {
                let d = x as u16;
                x = 0;
                d
            }
        })
        .collect()
}

/// Invert inp_a mod inp_b.
pub(crate) fn inv(inp_a: i128, inp_b: i128) -> i128 {
    let mut a = inp_a;
    let mut b = inp_b;
    let mut q;
    let mut tmp;

    let (mut x0, mut x1) = (0, 1);

    if b == 1 {
        return 1;
    }

    while a > 1 {
        q = a / b;

        // a, b = b, a%b
        tmp = b;
        b = a % b;
        a = tmp;

        tmp = x0;
        x0 = x1 - q * x0;
        x1 = tmp;
    }

    if x1 < 0 {
        x1 += inp_b;
    }

    x1
}

/// Convert crt value to Fp
/// TODO: make this generic
pub fn convert_crt_to_value(len: usize, values: &[u16]) -> Result<ArithValue, TypeError> {
    if values.len() != len {
        return Err(TypeError::InvalidLength {
            expected: len,
            actual: values.len(),
        });
    };

    let n_acc = PRIMES.iter().take(len).fold(1, |acc, q| *q as i128 * acc);
    let mut ret = 0;

    for (p, a) in PRIMES.iter().take(len).zip(values) {
        let p = *p as i128;

        let q = n_acc / p;
        ret += *a as i128 * inv(q, p) * q;
        ret %= &n_acc;
    }
    Ok(ArithValue::U32(ret as u32))
}

/// Convert set of crt represented values to field points.
/// This methods uses ArithNodes in CrtRepr to reference values in feed.
/// TODO: make this generic over crt convertible type
pub fn convert_crts_to_values(
    crt_reprs: &[CrtRepr],
    feeds: &[Option<u16>],
) -> Result<Vec<u32>, ArithCircuitError> {
    // TODO: use convert_crt_to_value method inside to avoid duplication
    crt_reprs
        .iter()
        .map(|crt| {
            let n_acc = crt
                .iter()
                .fold(1, |acc, &node| node.modulus() as i128 * acc);

            let mut ret = 0;

            for node in crt.iter() {
                let p = node.modulus() as i128;
                let a = feeds[node.id()].expect("feed should be initialized."); // TODO: check return error if no feeds set.

                let q = n_acc / p;
                ret += a as i128 * inv(q, p) * q;
                ret %= &n_acc;
            }
            Ok(ret as u32)
        })
        .collect()
}

/// Convert set of values to CRT representation and put actual values in Feed.
pub fn convert_values_to_crts(
    crt_reprs: &[CrtRepr],
    values: &[u32],
) -> Result<Vec<Vec<u16>>, ArithCircuitError> {
    Ok(crt_reprs
        .iter()
        .zip(values.iter())
        .map(|(crt, val)| {
            crt.iter()
                .map(|n| (val % (n.modulus() as u32)) as u16)
                .collect::<Vec<u16>>()
        })
        .collect::<Vec<Vec<u16>>>())
}

/// Convert arithmetic value to crt representation.
pub fn convert_value_to_crt(value: ArithValue) -> Vec<u16> {
    match value {
        ArithValue::Bool(v) => vec![v as u16],
        ArithValue::U32(v) => PRIMES
            .iter()
            .take(u32::LEN)
            .map(|n| (v % (*n as u32)) as u16)
            .collect::<Vec<u16>>(),
    }
}

/// Returns `true` if `x` is a power of 2.
pub fn is_power_of_2(x: u16) -> bool {
    (x & (x - 1)) == 0
}

/// Determine how many `mod q` digits fit into a `u128` (includes the color
/// digit).
pub fn digits_per_u128(modulus: u16) -> usize {
    debug_assert_ne!(modulus, 0);
    debug_assert_ne!(modulus, 1);
    if modulus == 2 {
        128
    } else if modulus <= 4 {
        64
    } else if modulus <= 8 {
        42
    } else if modulus <= 16 {
        32
    } else if modulus <= 32 {
        25
    } else if modulus <= 64 {
        21
    } else if modulus <= 128 {
        18
    } else if modulus <= 256 {
        16
    } else if modulus <= 512 {
        14
    } else {
        (128.0 / (modulus as f64).log2().ceil()).floor() as usize
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        arithmetic::types::{ArithNode, CrtRepr, CrtValue},
        Feed,
    };

    use super::*;

    const CRT_124: [u16; 10] = [0, 1, 4, 5, 3, 7, 5, 10, 9, 8];

    fn generate_crt_repr() -> CrtRepr {
        let nodes: [ArithNode<Feed>; 10] =
            std::array::from_fn(|i| ArithNode::<Feed>::new(i, PRIMES[i]));
        CrtRepr::U32(CrtValue::new(nodes))
    }

    #[test]
    fn test_convert_crts_to_values() {
        let crt_repr = generate_crt_repr();

        let res = convert_values_to_crts(&[crt_repr], &[124]);
        assert!(res.is_ok());

        let crt_actual_vals = res.unwrap();
        assert_eq!(crt_actual_vals, vec![Vec::from(CRT_124)]);
    }

    #[test]
    fn test_convert_values_to_crts() {
        // prepre values and feeds
        let crt_repr = generate_crt_repr();
        // prepare 10 feed values using above
        let feeds: Vec<Option<u16>> = CRT_124.iter().map(|v| Some(*v)).collect();

        let res = convert_crts_to_values(&[crt_repr], &feeds);
        assert!(res.is_ok());

        let actual_vals = res.unwrap();
        assert_eq!(actual_vals, vec![124]);
    }
}
