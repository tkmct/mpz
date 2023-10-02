use std::{collections::HashMap, sync::Arc};

use crate::{
    circuit::EncryptedGate,
    encoding::{Delta, Label, Labels},
};

use mpz_circuits::{arithmetic::components::ArithGate, ArithmeticCircuit};
use mpz_core::{
    aes::{FixedKeyAes, FIXED_KEY_AES},
    Block,
};

pub struct BMR16Generator {
    // TODO: make this generic?
    cipher: &'static FixedKeyAes,
    /// Circuit to genrate a garbled circuit for
    circ: Arc<ArithmeticCircuit>,
    /// Delta values to use while generating the circuit
    deltas: HashMap<u16, Delta>,
    /// The 0 value labels for the garbled circuit
    // FIXME: how to represent set of labels for each node?
    low_labels: Vec<Option<Label>>,
    /// Current position in the circuit
    pos: usize,
}

impl BMR16Generator {
    // NOTE: encoding of the labels are done outside.
    pub fn new(circ: Arc<ArithmeticCircuit>, deltas: HashMap<u16, Delta>) -> Self {
        // TODO: fix this
        let low_labels = vec![];

        BMR16Generator {
            cipher: &(FIXED_KEY_AES),
            circ,
            deltas,
            low_labels,
            pos: 0,
        }
    }
}

impl Iterator for BMR16Generator {
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
