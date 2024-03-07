//! BMR16 garble module

pub mod evaluator;
pub mod generator;

// Test generator and evaluator
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        bmr16::{evaluator::*, generator::*},
        encoding::{
            crt_encoding_state, ChaChaCrtEncoder, CrtDecoding, DecodeError, EncodedCrtValue,
        },
    };
    use mpz_circuits::{
        arithmetic::{
            ops::{add, crt_sign, mul, sub},
            types::{ArithValue, CrtLen, CrtRepr, CrtValue, ToCrtRepr},
            utils::{convert_values_to_crts, PRIMES},
        },
        ArithmeticCircuit, ArithmeticCircuitBuilder,
    };
    use mpz_core::aes::FIXED_KEY_AES;

    fn adder_circ<T: CrtLen + ToCrtRepr>() -> ArithmeticCircuit {
        let builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_input::<T>("x".into()).unwrap();
        let y = builder.add_input::<T>("y".into()).unwrap();

        let z = add(&mut builder.state().borrow_mut(), &x.repr, &y.repr).unwrap();
        builder.add_output(&z);
        builder.build().unwrap()
    }

    // circuit includes multiplication and projection gates
    fn mul_circ() -> ArithmeticCircuit {
        // TODO: write complex circuit
        let builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_input::<u32>("x".into()).unwrap();
        let y = builder.add_input::<u32>("y".into()).unwrap();

        let z = mul(&mut builder.state().borrow_mut(), &x.repr, &y.repr).unwrap();
        builder.add_output(&z);
        builder.build().unwrap()
    }

    fn sign_circ() -> ArithmeticCircuit {
        let builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_input::<u32>("x".into()).unwrap();
        let y = builder.add_input::<u32>("y".into()).unwrap();

        let z = sub(&mut builder.state().borrow_mut(), &x.repr, &y.repr).unwrap();
        let out = crt_sign::<10>(&mut builder.state().borrow_mut(), &z, "100%").unwrap();
        builder.add_output(&out);
        builder.build().unwrap()
    }

    #[test]
    fn test_garble_add_circ() {
        const BATCH_SIZE: usize = 1000;
        let encoder = ChaChaCrtEncoder::new([0; 32], &PRIMES[0..10]);
        let circ = adder_circ::<u32>();

        let a_val = 3;
        let b_val = 5;

        let expected = a_val + b_val;

        // TODO: construct CRT from actual value more easily.
        let a: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::U32(CrtValue::<10>::new_from_id(0))], &[a_val])
                .unwrap()[0]
                .clone();

        let b: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::U32(CrtValue::<10>::new_from_id(0))], &[b_val])
                .unwrap()[0]
                .clone();

        let full_inputs: Vec<EncodedCrtValue<crt_encoding_state::Full>> = circ
            .inputs()
            .iter()
            .map(|input| encoder.encode_by_len(0, input.repr.len()))
            .collect();

        let mut gen = BMR16Generator::new(
            Arc::new(circ.clone()),
            encoder.deltas().clone(),
            full_inputs.clone(),
        )
        .unwrap();

        let active_inputs: Vec<EncodedCrtValue<crt_encoding_state::Active>> = vec![
            full_inputs[0].select(encoder.deltas(), a),
            full_inputs[1].select(encoder.deltas(), b),
        ];

        let mut ev = BMR16Evaluator::new(Arc::new(circ.clone()), active_inputs).unwrap();

        while !(gen.is_complete() && ev.is_complete()) {
            let mut batch = Vec::with_capacity(BATCH_SIZE);
            for enc_gate in gen.by_ref() {
                batch.push(enc_gate);
                if batch.len() == BATCH_SIZE {
                    break;
                }
            }
            ev.evaluate(batch.iter());
        }

        let gen_digest = gen.hash().unwrap();
        let ev_digest = ev.hash().unwrap();

        assert_eq!(gen_digest, ev_digest);

        let deltas = encoder.deltas();

        let decodings = gen
            .outputs()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(idx, output)| output.decoding(idx, deltas, &FIXED_KEY_AES))
            .collect::<Vec<CrtDecoding>>();

        let outputs: Result<Vec<ArithValue>, DecodeError> = ev
            .outputs()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, encoded_output)| encoded_output.decode(i, &decodings[i]))
            .collect();

        assert!(outputs.is_ok());
        assert_eq!(outputs.unwrap()[0], ArithValue::from(expected));
    }

    #[test]
    fn test_garble_mul_circuit() {
        const BATCH_SIZE: usize = 1000;
        let encoder = ChaChaCrtEncoder::new([0; 32], &PRIMES[0..10]);
        let circ = mul_circ();

        let a_val = 20;
        let b_val = 4;

        let expected = a_val * b_val;

        // TODO: construct CRT from actual value more easily.
        let a: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::U32(CrtValue::<10>::new_from_id(0))], &[a_val])
                .unwrap()[0]
                .clone();

        let b: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::U32(CrtValue::<10>::new_from_id(0))], &[b_val])
                .unwrap()[0]
                .clone();

        let full_inputs: Vec<EncodedCrtValue<crt_encoding_state::Full>> = circ
            .inputs()
            .iter()
            .map(|input| encoder.encode_by_len(0, input.repr.len()))
            .collect();

        let mut gen = BMR16Generator::new(
            Arc::new(circ.clone()),
            encoder.deltas().clone(),
            full_inputs.clone(),
        )
        .unwrap();

        let active_inputs: Vec<EncodedCrtValue<crt_encoding_state::Active>> = vec![
            full_inputs[0].select(encoder.deltas(), a),
            full_inputs[1].select(encoder.deltas(), b),
        ];

        let mut ev = BMR16Evaluator::new(Arc::new(circ.clone()), active_inputs).unwrap();

        while !(gen.is_complete() && ev.is_complete()) {
            let mut batch = Vec::with_capacity(BATCH_SIZE);
            for enc_gate in gen.by_ref() {
                batch.push(enc_gate);
                if batch.len() == BATCH_SIZE {
                    break;
                }
            }
            ev.evaluate(batch.iter());
        }

        let gen_digest = gen.hash().unwrap();
        let ev_digest = ev.hash().unwrap();

        assert_eq!(gen_digest, ev_digest);

        let deltas = encoder.deltas();
        let decodings = gen
            .outputs()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(idx, output)| output.decoding(idx, deltas, &FIXED_KEY_AES))
            .collect::<Vec<CrtDecoding>>();

        let outputs: Result<Vec<ArithValue>, DecodeError> = ev
            .outputs()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, encoded_output)| encoded_output.decode(i, &decodings[i]))
            .collect();

        assert!(outputs.is_ok());
        assert_eq!(outputs.unwrap()[0], ArithValue::from(expected));
    }

    // TODO: test sign circuit
    #[test]
    fn test_garble_sign_circuit() {
        const BATCH_SIZE: usize = 1000;
        let circ = sign_circ();
        // TODO: get moduli used for the things
        let moduli = circ.moduli();

        let encoder = ChaChaCrtEncoder::new([0; 32], &moduli);

        let a_val = 4;
        let b_val = 20;

        let expected = 1;
        let a: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::U32(CrtValue::<10>::new_from_id(0))], &[a_val])
                .unwrap()[0]
                .clone();
        let b: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::U32(CrtValue::<10>::new_from_id(0))], &[b_val])
                .unwrap()[0]
                .clone();

        let full_inputs: Vec<EncodedCrtValue<crt_encoding_state::Full>> = circ
            .inputs()
            .iter()
            .map(|input| encoder.encode_by_len(0, input.repr.len()))
            .collect();

        let mut gen = BMR16Generator::new(
            Arc::new(circ.clone()),
            encoder.deltas().clone(),
            full_inputs.clone(),
        )
        .unwrap();

        let active_inputs: Vec<EncodedCrtValue<crt_encoding_state::Active>> = vec![
            full_inputs[0].select(encoder.deltas(), a),
            full_inputs[1].select(encoder.deltas(), b),
        ];

        let mut ev = BMR16Evaluator::new(Arc::new(circ.clone()), active_inputs).unwrap();

        while !(gen.is_complete() && ev.is_complete()) {
            let mut batch = Vec::with_capacity(BATCH_SIZE);
            for enc_gate in gen.by_ref() {
                batch.push(enc_gate);
                if batch.len() == BATCH_SIZE {
                    break;
                }
            }
            ev.evaluate(batch.iter());
        }

        let gen_digest = gen.hash().unwrap();
        let ev_digest = ev.hash().unwrap();

        assert_eq!(gen_digest, ev_digest);

        let deltas = encoder.deltas();
        let decodings = gen
            .outputs()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(idx, output)| output.decoding(idx, deltas, &FIXED_KEY_AES))
            .collect::<Vec<CrtDecoding>>();

        let outputs: Result<Vec<ArithValue>, DecodeError> = ev
            .outputs()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, encoded_output)| encoded_output.decode(i, &decodings[i]))
            .collect();

        assert!(outputs.is_ok());
        assert_eq!(outputs.unwrap()[0], ArithValue::from(expected));
    }
}
