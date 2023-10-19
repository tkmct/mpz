pub mod evaluator;
pub mod generator;

// Test generator and evaluator
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        bmr16::{evaluator::*, generator::*},
        encoding::{crt_encoding_state, ChaChaCrtEncoder, EncodedCrtValue},
    };
    use mpz_circuits::{
        arithmetic::{
            ops::add,
            types::{CrtLen, CrtRepr, CrtValue, Fp, ToCrtRepr},
            utils::convert_values_to_crts,
        },
        ArithmeticCircuit, ArithmeticCircuitBuilder,
    };

    fn adder_circ<T: CrtLen + ToCrtRepr>() -> ArithmeticCircuit {
        let builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_input::<T>().unwrap();
        let y = builder.add_input::<T>().unwrap();

        let z = add(&mut builder.state().borrow_mut(), &x, &y).unwrap();
        builder.add_output(&z);
        builder.build().unwrap()
    }

    #[test]
    fn test_garble_bmr16_bool_add() {
        const BATCH_SIZE: usize = 1000;
        let encoder = ChaChaCrtEncoder::<2>::new([0; 32]);
        let circ = adder_circ::<bool>();

        let a_val = Fp(0);
        let b_val = Fp(1);

        // TODO: implement Field operation on Fp or use arkworks Fp?
        // let expected = a_val + b_val;
        let expected = Fp(1);

        // TODO: construct CRT from actual value more easily.
        let a: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::Bool(CrtValue::<1>::new_from_id(0))], &[a_val])
                .unwrap()[0]
                .iter()
                .map(|fp| fp.0 as u16)
                .collect();
        let b: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::Bool(CrtValue::<1>::new_from_id(0))], &[b_val])
                .unwrap()[0]
                .iter()
                .map(|fp| fp.0 as u16)
                .collect();

        let full_inputs: Vec<EncodedCrtValue<crt_encoding_state::Full>> = circ
            .inputs()
            .iter()
            .map(|input| encoder.encode_by_len(0, input.len()))
            .collect();

        let active_inputs: Vec<EncodedCrtValue<crt_encoding_state::Active>> = vec![
            full_inputs[0].clone().select(encoder.deltas(), a),
            full_inputs[1].clone().select(encoder.deltas(), b),
        ];

        let mut gen = BMR16Generator::<1>::new(
            Arc::new(circ.clone()),
            encoder.deltas().clone(),
            full_inputs,
        )
        .unwrap();

        let mut ev = BMR16Evaluator::<1>::new(Arc::new(circ.clone()), active_inputs).unwrap();

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

        let decodings = gen.decodings().unwrap();
        let outputs = ev.decode_outputs(decodings);
        assert!(outputs.is_ok());
        assert_eq!(outputs.unwrap()[0], expected);
    }

    #[test]
    fn test_garble_bmr16() {
        const BATCH_SIZE: usize = 1000;
        let encoder = ChaChaCrtEncoder::<10>::new([0; 32]);
        let circ = adder_circ::<u32>();

        let a_val = Fp(3);
        let b_val = Fp(5);

        // TODO: implement Field operation on Fp or use arkworks Fp?
        // let expected = a_val + b_val;
        let expected = Fp(8);

        // TODO: construct CRT from actual value more easily.
        let a: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::U32(CrtValue::<10>::new_from_id(0))], &[a_val])
                .unwrap()[0]
                .iter()
                .map(|fp| fp.0 as u16)
                .collect();

        let b: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::U32(CrtValue::<10>::new_from_id(0))], &[b_val])
                .unwrap()[0]
                .iter()
                .map(|fp| fp.0 as u16)
                .collect();

        let full_inputs: Vec<EncodedCrtValue<crt_encoding_state::Full>> = circ
            .inputs()
            .iter()
            .map(|input| encoder.encode_by_len(0, input.len()))
            .collect();

        let active_inputs: Vec<EncodedCrtValue<crt_encoding_state::Active>> = vec![
            full_inputs[0].clone().select(encoder.deltas(), a),
            full_inputs[1].clone().select(encoder.deltas(), b),
        ];

        let mut gen = BMR16Generator::<10>::new(
            Arc::new(circ.clone()),
            encoder.deltas().clone(),
            full_inputs,
        )
        .unwrap();

        let mut ev = BMR16Evaluator::<10>::new(Arc::new(circ.clone()), active_inputs).unwrap();

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

        let decodings = gen.decodings().unwrap();
        let outputs = ev.decode_outputs(decodings);

        assert!(outputs.is_ok());
        assert_eq!(outputs.unwrap()[0], expected);
    }
}
