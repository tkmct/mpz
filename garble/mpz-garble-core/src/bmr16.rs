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
            types::{CrtRepr, CrtValue, Fp},
            utils::convert_values_to_crts,
        },
        ArithmeticCircuit, ArithmeticCircuitBuilder,
    };

    fn adder_circ() -> ArithmeticCircuit {
        let builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_input::<u32>().unwrap();
        let y = builder.add_input::<u32>().unwrap();

        let z = add(&mut builder.state().borrow_mut(), &x, &y).unwrap();
        builder.add_output(&z);

        builder.build().unwrap()
    }

    #[test]
    fn test_garble_bmr16() {
        const BATCH_SIZE: usize = 1000;
        let encoder = ChaChaCrtEncoder::<10>::new([0; 32]);
        let circ = adder_circ();

        let a: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::U32(CrtValue::<10>::new_from_id(0))], &[Fp(5)])
                .unwrap()[0]
                .iter()
                .map(|fp| fp.0 as u16)
                .collect();
        let b: Vec<u16> =
            convert_values_to_crts(&[CrtRepr::U32(CrtValue::<10>::new_from_id(0))], &[Fp(100)])
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

        let full_outputs = gen.outputs().unwrap();
        let active_outputs = ev.outputs().unwrap();

        let gen_digest = gen.hash().unwrap();
        let ev_digest = ev.hash().unwrap();

        assert_eq!(gen_digest, ev_digest);

        // In actual case, garbler send decoding value via channel.
        let outputs: Vec<Fp> = active_outputs
            .iter()
            .zip(full_outputs)
            .map(|(active_output, full_output)| {
                active_output.decode(&full_output.decoding()).unwrap()
            })
            .collect();

        let actual: [u8; 16] = outputs[0].clone().try_into().unwrap();

        assert_eq!(actual, expected);
    }
}
