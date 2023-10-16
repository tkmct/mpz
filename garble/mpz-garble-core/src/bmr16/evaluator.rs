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
pub enum EvaluatorError {
    #[error(transparent)]
    TypeError(#[from] TypeError),
    #[error(transparent)]
    CircuitError(#[from] ArithCircuitError),
    #[error("generator not finished")]
    NotFinished,
}

pub struct BMR16Evaluator<const N: usize> {
    cipher: &'static FixedKeyAes,
    /// Circuit to genrate a garbled circuit for
    circ: Arc<ArithmeticCircuit>,
    /// The 0 value labels for the garbled circuit
    active_labels: Vec<Option<Label>>,
    /// Current position in the circuit
    pos: usize,
    /// Current gate id
    gid: usize,
    /// Whether the evaluator is finished
    complete: bool,
    /// Hasher to use to hash the encrypted gates
    hasher: Option<Hasher>,
}

impl<const N: usize> BMR16Evaluator<N> {
    pub fn new(
        circ: Arc<ArithmeticCircuit>,
        inputs: Vec<EncodedCrtValue<crt_encoding_state::Active>>,
    ) -> Result<Self, EvaluatorError> {
        let hasher = Some(Hasher::new());
        let mut active_labels: Vec<Option<Label>> = vec![None; circ.feed_count()];
        for (encoded, input) in inputs.iter().zip(circ.inputs()) {
            if encoded.len() != input.len() {
                return Err(TypeError::InvalidLength {
                    expected: input.len(),
                    actual: encoded.len(),
                }
                .into());
            }

            for (label, node) in encoded.iter().zip(input.iter()) {
                active_labels[node.id()] = Some(*label);
            }
        }

        Ok(Self {
            cipher: &(FIXED_KEY_AES),
            circ,
            active_labels,
            pos: 0,
            gid: 1,
            complete: false,
            hasher,
        })
    }

    /// Returns whether the evaluator has finished evaluating the circuit.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Returns the encoded outputs of the circuit.
    pub fn outputs(
        &self,
    ) -> Result<Vec<EncodedCrtValue<crt_encoding_state::Active>>, EvaluatorError> {
        if !self.is_complete() {
            return Err(EvaluatorError::NotFinished);
        }

        Ok(self
            .circ
            .outputs()
            .iter()
            .map(|output| {
                let labels: Vec<Label> = output
                    .iter()
                    .map(|node| self.active_labels[node.id()].expect("feed should be initialized"))
                    .collect();

                EncodedCrtValue::<crt_encoding_state::Active>::from(labels)
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

    pub fn evaluate<'a>(&mut self, mut encrypted_gates: impl Iterator<Item = &'a EncryptedGate>) {
        let labels = &mut self.active_labels;

        while self.pos < self.circ.gates().len() {
            match &self.circ.gates()[self.pos] {
                ArithGate::Add {
                    x: node_x,
                    y: node_y,
                    z: node_z,
                } => {
                    let x = labels[node_x.id()].expect("feed should be initialized");
                    let y = labels[node_y.id()].expect("feed should be initialized");
                    labels[node_z.id()] = Some(Label::new(add_block(&x.to_inner(), &y.to_inner())));
                }
                ArithGate::Cmul {
                    x: node_x,
                    c,
                    z: node_z,
                } => {
                    let x = labels[node_x.id()].expect("feed should be initialized");
                    labels[node_z.id()] = Some(Label::new(cmul_block(&x.to_inner(), *c)));
                }
                ArithGate::Mul { x, y, z } => {
                    todo!()
                }
                ArithGate::Proj { x, tt, z } => {
                    todo!()
                }
            }

            self.pos += 1;
        }

        self.complete = true;
    }
}
