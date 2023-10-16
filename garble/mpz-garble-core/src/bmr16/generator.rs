use std::{collections::HashMap, sync::Arc};

use blake3::Hasher;

use crate::{
    circuit::EncryptedGate,
    encoding::{add_block, cmul_block, crt_encoding_state, Delta, EncodedCrtValue, Label},
};

use mpz_circuits::arithmetic::{
    components::ArithGate, ArithCircuitError, ArithmeticCircuit, TypeError,
};
use mpz_core::{
    aes::{FixedKeyAes, FIXED_KEY_AES},
    hash::Hash,
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
    deltas: HashMap<u16, Delta>,
    /// The 0 value labels for the garbled circuit
    low_labels: Vec<Option<Label>>,
    /// Current position in the circuit
    pos: usize,
    /// Current gate id
    gid: usize,
    /// Hasher to use to hash the encrypted gates
    hasher: Option<Hasher>,
}

impl<const N: usize> BMR16Generator<N> {
    /// Create bmr16 generator struct.
    /// encoding of the input labels are done outside.
    pub fn new(
        circ: Arc<ArithmeticCircuit>,
        deltas: HashMap<u16, Delta>,
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
                low_labels[node.id()] = Some(*label)
            }
        }

        Ok(Self {
            cipher: &(FIXED_KEY_AES),
            circ,
            deltas,
            low_labels,
            pos: 0,
            gid: 1,
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
                let labels: Vec<Label> = output
                    .iter()
                    .map(|node| self.low_labels[node.id()].expect("feed should be initialized"))
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
                    let x_0 = low_labels[x.id()].expect("feed should be set.");
                    let y_0 = low_labels[y.id()].expect("feed should be set.");
                    // set zero label for z
                    low_labels[z.id()] =
                        Some(Label::new(add_block(&x_0.to_inner(), &y_0.to_inner())));
                }
                ArithGate::Cmul { x, c, z } => {
                    // zero labels for input x
                    let x_0 = low_labels[x.id()].expect("feed should be set.");
                    // set zero label for z
                    low_labels[z.id()] = Some(Label::from(cmul_block(&x_0.to_inner(), *c)));
                }
                ArithGate::Mul { x, y, z } => {
                    todo!()
                }
                ArithGate::Proj { x, tt, z } => {
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
