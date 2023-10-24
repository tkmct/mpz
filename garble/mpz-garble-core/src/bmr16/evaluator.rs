//! BMR16 evaluator implementation
use std::sync::Arc;

use blake3::Hasher;

use crate::{
    circuit::ArithEncryptedGate,
    encoding::{
        add_label, cmul_label, crt_encoding_state, output_tweak, tweak, tweak2, CrtDecoding,
        DecodeError, EncodedCrtValue, LabelModN,
    },
};

use mpz_circuits::arithmetic::{
    components::ArithGate,
    utils::{convert_crt_to_value, PRIMES},
    ArithCircuitError, ArithmeticCircuit, TypeError,
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
    #[error(transparent)]
    DecodeError(#[from] DecodeError),
}

/// BMR16 evaluator struct
pub struct BMR16Evaluator<const N: usize> {
    cipher: &'static FixedKeyAes,
    /// Circuit to genrate a garbled circuit for
    circ: Arc<ArithmeticCircuit>,
    /// The 0 value labels for the garbled circuit
    active_labels: Vec<Option<LabelModN>>,
    /// Current position in the circuit
    pos: usize,
    /// Current gate id. needed for streaming
    gid: usize,
    /// Whether the evaluator is finished
    complete: bool,
    /// Hasher to use to hash the encrypted gates
    hasher: Option<Hasher>,
}

impl<const N: usize> BMR16Evaluator<N> {
    /// Construct new evaluator instance from circuit and inputs.
    /// Inputs should be encoded as active encoding using EncodedCrtValue.
    pub fn new(
        circ: Arc<ArithmeticCircuit>,
        inputs: Vec<EncodedCrtValue<crt_encoding_state::Active>>,
    ) -> Result<Self, EvaluatorError> {
        let hasher = Some(Hasher::new());
        let mut active_labels: Vec<Option<LabelModN>> = vec![None; circ.feed_count()];
        for (encoded, input) in inputs.iter().zip(circ.inputs()) {
            if encoded.len() != input.len() {
                return Err(TypeError::InvalidLength {
                    expected: input.len(),
                    actual: encoded.len(),
                }
                .into());
            }

            for (label, node) in encoded.iter().zip(input.iter()) {
                active_labels[node.id()] = Some(label.clone());
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
                let labels: Vec<LabelModN> = output
                    .iter()
                    .map(|node| {
                        self.active_labels[node.id()]
                            .clone()
                            .expect("feed should be initialized")
                    })
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

    /// Evaluate garbled circuit.
    pub fn evaluate<'a>(
        &mut self,
        mut encrypted_gates: impl Iterator<Item = &'a ArithEncryptedGate>,
    ) {
        let labels = &mut self.active_labels;

        while self.pos < self.circ.gates().len() {
            match &self.circ.gates()[self.pos] {
                ArithGate::Add {
                    x: node_x,
                    y: node_y,
                    z: node_z,
                } => {
                    let x = labels
                        .get(node_x.id())
                        .expect("label index out of range")
                        .clone()
                        .expect("label should be initialized");
                    let y = labels
                        .get(node_y.id())
                        .expect("label index out of range")
                        .clone()
                        .expect("label should be initialized");

                    labels[node_z.id()] = Some(add_label(&x, &y));
                }
                ArithGate::Cmul {
                    x: node_x,
                    c,
                    z: node_z,
                } => {
                    let x = labels
                        .get(node_x.id())
                        .expect("label index out of range")
                        .clone()
                        .expect("label should be initialized");

                    labels[node_z.id()] = Some(cmul_label(&x, *c as u64));
                }
                ArithGate::Mul {
                    x: node_x,
                    y: node_y,
                    z: node_z,
                } => {
                    if let Some(gate) = encrypted_gates.next() {
                        let x = labels
                            .get(node_x.id())
                            .expect("label index out of range")
                            .clone()
                            .expect("zero label should be set.");

                        let y = labels
                            .get(node_y.id())
                            .expect("label index out of range")
                            .clone()
                            .expect("zero label should be set.");

                        debug_assert_eq!(node_x.modulus(), node_y.modulus());

                        // TODO: swap based on modulus size
                        // if node_x.modulus() < node_y.modulus() {
                        //     std::mem::swap(&mut x, &mut y);
                        // }

                        let q_x = node_x.modulus();
                        let q_y = node_y.modulus();

                        // let unequal = q_x != q_y;
                        // let ngates = q_x as usize + q_y as usize - 2 + unequal as usize;

                        let gate_num = self.gid;
                        self.gid += 1;
                        let g = tweak2(gate_num as u64, 0);

                        // let [hashA, hashB] = hash_wires([A, B], g);
                        let x_enc = self.cipher.tccr(g, x.to_block());
                        let y_enc = self.cipher.tccr(g, y.to_block());

                        // garbler's half gate
                        let l = if x.color() == 0 {
                            LabelModN::from_block(x_enc, q_x)
                            // Wire::hash_to_mod(x_enc, q_x)
                        } else {
                            let ct_left = gate
                                .get(x.color() as usize - 1)
                                .expect("gate should be received");
                            LabelModN::from_block(*ct_left ^ x_enc, q_x)
                        };

                        // evaluator's half gate
                        let r = if y.color() == 0 {
                            LabelModN::from_block(y_enc, q_x)
                        } else {
                            let ct_right = gate
                                .get((q_x + y.color()) as usize - 2)
                                .expect("gate should be received");
                            LabelModN::from_block(*ct_right ^ y_enc, q_x)
                        };

                        // // hack for unequal mods
                        // // TODO: Batch this with original hash if unequal.
                        // let new_b_color = if unequal {
                        //     let minitable = *gate.last().unwrap();
                        //     let ct = u128::from(minitable) >> (B.color() * 16);
                        //     let pt = u128::from(B.hash(tweak2(gate_num as u64, 1))) ^ ct;
                        //     pt as u16
                        // } else {
                        //     B.color()
                        // };
                        //

                        labels[node_z.id()] = Some(add_label(
                            &l,
                            &add_label(&r, &cmul_label(&x, y.color() as u64)),
                        ));
                    } else {
                        return;
                    }
                }
                ArithGate::Proj {
                    x: node_x,
                    tt,
                    z: node_z,
                } => {
                    let q = node_z.modulus();

                    if let Some(gate) = encrypted_gates.next() {
                        let gate_num = self.gid;
                        self.gid += 1;
                        let t = tweak(gate_num);

                        let x = labels
                            .get(node_x.id())
                            .expect("label index out of range")
                            .clone()
                            .expect("label should be initialized");

                        if x.color() == 0 {
                            let hash = self.cipher.tccr(t, x.to_block());
                            labels[node_z.id()] = Some(LabelModN::from_block(hash, q));
                        } else {
                            let ct = gate
                                .get(x.color() as usize - 1)
                                .expect("gate should be received.");
                            labels[node_z.id()] = Some(LabelModN::from_block(
                                *ct ^ self.cipher.tccr(t, x.to_block()),
                                q,
                            ));
                        }
                    } else {
                        return;
                    }
                }
            }
            self.pos += 1;
        }

        self.complete = true;
    }

    /// Decode outputs of garbled circuit using decoding info given by generator.
    pub fn decode_outputs(&self, decodings: Vec<CrtDecoding>) -> Result<Vec<u32>, EvaluatorError> {
        let outputs = self.outputs()?;

        let values: Result<Vec<u32>, DecodeError> = outputs
            .iter()
            .zip(decodings.iter())
            .enumerate()
            .map(|(idx, (output, decoding))| {
                let decoded_nodes: Vec<u16> = output
                    .iter()
                    .enumerate()
                    .map(|(i, label)| {
                        let q = PRIMES[i];
                        let mut decoded = None;

                        for k in 0..q {
                            let tweak = output_tweak(idx, k);
                            let actual =
                                LabelModN::from_block(self.cipher.tccr(tweak, label.to_block()), q);
                            let dec = decoding.get(i, k as usize).unwrap_or_else(|| {
                                panic!("Decoding should be set for.{}, {}, {}", idx, i, k)
                            });

                            if actual == *dec {
                                decoded = Some(k);
                                break;
                            }
                        }
                        decoded.ok_or(DecodeError::LackOfDecodingInfo(idx, q))
                    })
                    .collect::<Result<Vec<u16>, DecodeError>>()?;

                convert_crt_to_value(output.len(), &decoded_nodes).map_err(|e| e.into())
            })
            .collect();

        values.map_err(|e| e.into())
    }
}
