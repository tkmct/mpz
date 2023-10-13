use std::{collections::HashMap, sync::Arc};

use blake3::Hasher;

use crate::{
    circuit::EncryptedGate,
    encoding::{crt_encoded_state, CrtLabels, Delta, EncodedCrtValue, Label},
};

use mpz_circuits::arithmetic::{
    components::ArithGate, ArithCircuitError, ArithmeticCircuit, TypeError,
};
use mpz_core::{
    aes::{FixedKeyAes, FIXED_KEY_AES},
    Block,
};

pub struct BMR16Generator<const N: usize> {
    // TODO: make this generic?
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
        inputs: Vec<EncodedCrtValue<crt_encoded_state::Full>>,
    ) -> Result<Self, ArithCircuitError> {
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

        Ok(BMR16Generator {
            cipher: &(FIXED_KEY_AES),
            circ,
            deltas,
            low_labels,
            pos: 0,
            gid: 1,
            hasher,
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
                    // zero labels for input x
                    let x_0s = low_labels[x.id()];
                }
                ArithGate::Cmul { x, c, z } => {
                    // do something
                }
                ArithGate::Mul { x, y, z } => {
                    // do something
                }
                ArithGate::Proj { x, tt, z } => {
                    // do something
                }
            }
        }

        Some(EncryptedGate::new([
            Block::new([0; 16]),
            Block::new([0; 16]),
        ]))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore]
    fn test_generator() {}
}
