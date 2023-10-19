use std::{collections::HashMap, sync::Arc};

use blake3::Hasher;
use mpz_circuits::arithmetic::{
    components::ArithGate, utils::PRIMES, ArithCircuitError, ArithmeticCircuit, TypeError,
};
use mpz_core::{
    aes::{FixedKeyAes, FIXED_KEY_AES},
    hash::Hash,
};

use crate::{
    circuit::EncryptedGate,
    encoding::{
        add_label, cmul_label, crt_encoding_state, output_tweak, CrtDecoding, CrtDelta,
        EncodedCrtValue, LabelModN,
    },
};

/// Errors that can occur during garbled circuit generation.
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum GeneratorError {
    #[error(transparent)]
    TypeError(#[from] TypeError),
    #[error(transparent)]
    CircuitError(#[from] ArithCircuitError),
    #[error("generator not finished")]
    NotFinished,
}

pub struct BMR16Generator<const N: usize> {
    cipher: &'static FixedKeyAes,
    /// Circuit to genrate a garbled circuit for
    circ: Arc<ArithmeticCircuit>,
    /// Delta values to use while generating the circuit
    deltas: HashMap<u16, CrtDelta>,
    /// The 0 value labels for the garbled circuit
    low_labels: Vec<Option<LabelModN>>,
    /// Current position in the circuit
    pos: usize,
    /// Current gate id. needed for streaming.
    _gid: usize,
    /// Hasher to use to hash the encrypted gates
    hasher: Option<Hasher>,
}

impl<const N: usize> BMR16Generator<N> {
    /// Create bmr16 generator struct.
    /// encoding of the input labels are done outside.
    pub fn new(
        circ: Arc<ArithmeticCircuit>,
        deltas: HashMap<u16, CrtDelta>,
        inputs: Vec<EncodedCrtValue<crt_encoding_state::Full>>,
    ) -> Result<Self, GeneratorError> {
        let hasher = Some(Hasher::new());

        // Set zero label for every input feed
        let mut low_labels = vec![None; circ.feed_count()];
        for (encoded, input) in inputs.iter().zip(circ.inputs()) {
            if encoded.len() != input.len() {
                return Err(TypeError::InvalidLength {
                    expected: encoded.len(),
                    actual: input.len(),
                }
                .into());
            }

            for (label, node) in encoded.iter().zip(input.iter()) {
                low_labels[node.id()] = Some(label.clone())
            }
        }

        Ok(Self {
            cipher: &(FIXED_KEY_AES),
            circ,
            deltas,
            low_labels,
            pos: 0,
            _gid: 1,
            hasher,
        })
    }

    /// Returns whether the generator has finished generating the circuit.
    pub fn is_complete(&self) -> bool {
        self.pos >= self.circ.gates().len()
    }

    /// Returns the encoded outputs of the circuit.
    pub fn outputs(
        &self,
    ) -> Result<Vec<EncodedCrtValue<crt_encoding_state::Full>>, GeneratorError> {
        if !self.is_complete() {
            return Err(GeneratorError::NotFinished);
        }

        Ok(self
            .circ
            .outputs()
            .iter()
            .map(|output| {
                let labels: Vec<LabelModN> = output
                    .iter()
                    .map(|node| {
                        self.low_labels
                            .get(node.id())
                            .expect("label index out of range")
                            .clone()
                            .expect("label should be initialized")
                    })
                    .collect();

                EncodedCrtValue::<crt_encoding_state::Full>::from(labels)
            })
            .collect())
    }

    /// Returns the hash of the encrypted gates.
    pub fn hash(&self) -> Option<Hash> {
        self.hasher.as_ref().map(|hasher| {
            let hash: [u8; 32] = hasher.finalize().into();
            Hash::from(hash)
        })
    }

    pub fn decodings(&self) -> Result<Vec<CrtDecoding>, GeneratorError> {
        let outputs = self.outputs()?;

        Ok(outputs
            .iter()
            .enumerate()
            .map(|(idx, output)| {
                let hashes = output
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let q = PRIMES[i];
                        let d = self.deltas.get(&q).unwrap();

                        (0..q)
                            .map(|k| {
                                let label = add_label(x, &cmul_label(d, k as u64));
                                let tweak = output_tweak(idx, k);
                                LabelModN::from_block(self.cipher.tccr(tweak, label.to_block()), q)
                            })
                            .collect::<Vec<LabelModN>>()
                    })
                    .collect();

                CrtDecoding::new(hashes)
            })
            .collect())
    }
}

impl<const N: usize> Iterator for BMR16Generator<N> {
    // FIXME: encryted gate for arithmetic gate depends on the number of wire
    // How to represent set of wires?
    type Item = EncryptedGate;

    // NOTE: calculate label for each gate. if the gate is free, just update label feeds(low_labels)
    // returns the encrypted truth table if the gate needs truth table. (MUL and PROJ)
    fn next(&mut self) -> Option<Self::Item> {
        let low_labels = &mut self.low_labels;

        while let Some(gate) = self.circ.gates().get(self.pos) {
            self.pos += 1;
            match gate {
                ArithGate::Add { x, y, z } => {
                    // zero labels for input x and y
                    let x_0 = low_labels
                        .get(x.id())
                        .expect("label index out of range")
                        .clone()
                        .expect("zero label should be set.");
                    let y_0 = low_labels
                        .get(y.id())
                        .expect("label index out of range")
                        .clone()
                        .expect("zero label should be set.");
                    // set zero label for z
                    low_labels[z.id()] = Some(add_label(&x_0, &y_0));
                }
                ArithGate::Cmul { x, c, z } => {
                    // zero labels for input x
                    let x_0 = low_labels
                        .get(x.id())
                        .expect("label index out of range")
                        .clone()
                        .expect("feed should be set.");
                    // set zero label for z
                    low_labels[z.id()] = Some(cmul_label(&x_0, c.0 as u64));
                }
                ArithGate::Mul { .. } => {
                    todo!()
                }
                ArithGate::Proj { .. } => {
                    todo!()
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use mpz_circuits::{arithmetic::ops::add, ArithmeticCircuit, ArithmeticCircuitBuilder};

    use super::BMR16Generator;
    use crate::{encoding::ChaChaCrtEncoder, EncryptedGate};

    fn adder_circ() -> ArithmeticCircuit {
        let builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_input::<u32>().unwrap();
        let y = builder.add_input::<u32>().unwrap();

        let z = add(&mut builder.state().borrow_mut(), &x, &y).unwrap();
        builder.add_output(&z);

        builder.build().unwrap()
    }

    #[test]
    fn test_bmr16_generator() {
        let encoder = ChaChaCrtEncoder::<10>::new([0; 32]);

        let circ = adder_circ();
        let deltas = HashMap::new();
        let mul_count = circ.mul_count();

        let encoded_inputs = circ
            .inputs()
            .iter()
            .map(|inp| encoder.encode_by_len(0, inp.len()))
            .collect();

        let mut gen = BMR16Generator::<10>::new(Arc::new(circ), deltas, encoded_inputs).unwrap();
        let enc_gates: Vec<EncryptedGate> = gen.by_ref().collect();

        assert!(gen.is_complete());
        assert_eq!(enc_gates.len(), mul_count);

        gen.outputs().unwrap();
        gen.hash().unwrap();
    }
}
