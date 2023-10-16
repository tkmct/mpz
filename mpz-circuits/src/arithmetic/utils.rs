//! ArithmeticCircuit utils
use std::mem::discriminant;

use crate::arithmetic::{
    circuit::ArithCircuitError,
    types::{CrtRepr, Fp},
};

/// Number of primes supported by our library.
pub const NPRIMES: usize = 29;

/// Primes used in fancy garbling.
pub const PRIMES: [u16; 29] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109,
];

/// Check if crt type is equal
pub(crate) fn is_same_crt_len(a: &CrtRepr, b: &CrtRepr) -> bool {
    discriminant(a) == discriminant(b)
}

/// Invert inp_a mod inp_b.
fn inv(inp_a: i128, inp_b: i128) -> i128 {
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

/// Convert set of crt represented values to field points.
/// This methods uses ArithNodes in CrtRepr to reference values in feed.
pub fn convert_crts_to_values(
    crt_reprs: &[CrtRepr],
    feeds: &[Option<Fp>],
) -> Result<Vec<Fp>, ArithCircuitError> {
    crt_reprs
        .iter()
        .map(|crt| {
            let n_acc = crt
                .iter()
                .fold(1, |acc, &node| node.modulus() as i128 * acc);

            let mut ret = 0;

            for node in crt.iter() {
                let p = node.modulus() as i128;
                let a = feeds[node.id()].unwrap().0; // TODO: check return error if no feeds set.

                let q = n_acc / p;
                ret += a as i128 * inv(q, p) * q;
                ret %= &n_acc;
            }
            Ok(Fp(ret as u32))
        })
        .collect()
}

/// Convert set of values to CRT representation and put actual values in Feed.
pub fn convert_values_to_crts(
    crt_reprs: &[CrtRepr],
    values: &[Fp],
) -> Result<Vec<Vec<Fp>>, ArithCircuitError> {
    Ok(crt_reprs
        .iter()
        .zip(values.iter())
        .map(|(crt, val)| {
            crt.iter()
                .map(|n| Fp(val.0 % (n.modulus() as u32)))
                .collect::<Vec<Fp>>()
        })
        .collect::<Vec<Vec<Fp>>>())
}

#[cfg(test)]
mod tests {
    use crate::{
        arithmetic::types::{ArithNode, CrtRepr, CrtValue, Fp},
        Feed,
    };

    use super::*;

    const CRT_124: [Fp; 10] = [
        Fp(0),
        Fp(1),
        Fp(4),
        Fp(5),
        Fp(3),
        Fp(7),
        Fp(5),
        Fp(10),
        Fp(9),
        Fp(8),
    ];

    fn generate_crt_repr() -> CrtRepr {
        let nodes: [ArithNode<Feed>; 10] =
            std::array::from_fn(|i| ArithNode::<Feed>::new(i, PRIMES[i]));
        CrtRepr::U32(CrtValue::new(nodes))
    }

    #[test]
    fn test_convert_crts_to_values() {
        let crt_repr = generate_crt_repr();

        let res = convert_values_to_crts(&[crt_repr], &[Fp(124)]);
        assert!(res.is_ok());

        let crt_actual_vals = res.unwrap();
        assert_eq!(crt_actual_vals, vec![Vec::from(CRT_124)]);
    }

    #[test]
    fn test_convert_values_to_crts() {
        // prepre values and feeds
        let crt_repr = generate_crt_repr();
        // prepare 10 feed values using above
        let feeds: Vec<Option<Fp>> = CRT_124.iter().map(|v| Some(*v)).collect();

        let res = convert_crts_to_values(&[crt_repr], &feeds);
        assert!(res.is_ok());

        let actual_vals = res.unwrap();
        assert_eq!(actual_vals, vec![Fp(124)]);
    }
}
