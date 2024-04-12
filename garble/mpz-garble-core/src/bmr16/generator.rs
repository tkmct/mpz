//! BMR16 generator implementation
use crate::encoding::negate_label;
use std::collections::HashMap;
use std::sync::Arc;

use blake3::Hasher;
use mpz_circuits::arithmetic::{
    components::ArithGate, ArithCircuitError, ArithmeticCircuit, TypeError,
};
use mpz_core::{
    aes::{FixedKeyAes, FIXED_KEY_AES},
    hash::Hash,
    Block,
};

use crate::{
    circuit::ArithEncryptedGate,
    encoding::{
        add_label, cmul_label, crt_encoding_state, tweak, tweak2, CrtDelta, EncodedCrtValue,
        LabelModN,
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

/// BMR16 generator
pub struct BMR16Generator {
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
    gid: usize,
    /// Hasher to use to hash the encrypted gates
    hasher: Option<Hasher>,
}

impl BMR16Generator {
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
            if encoded.len() != input.repr.len() {
                return Err(TypeError::InvalidLength {
                    expected: encoded.len(),
                    actual: input.repr.len(),
                }
                .into());
            }

            for (label, node) in encoded.iter().zip(input.repr.iter()) {
                low_labels[node.id()] = Some(label.clone())
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
}

impl Iterator for BMR16Generator {
    // FIXME: encryted gate for arithmetic gate depends on the number of wire
    // How to represent set of wires?
    type Item = ArithEncryptedGate;

    // NOTE: calculate label for each gate. if the gate is free, just update label feeds(low_labels)
    // returns the encrypted truth table if the gate needs truth table. (MUL and PROJ)
    fn next(&mut self) -> Option<Self::Item> {
        let low_labels = &mut self.low_labels;

        while let Some(gate) = self.circ.gates().get(self.pos) {
            self.pos += 1;
            match gate {
                ArithGate::Add { x, y, z } => {
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
                ArithGate::Sub { x, y, z } => {
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
                    let neg_y_0 = negate_label(&y_0);
                    low_labels[z.id()] = Some(add_label(&x_0, &neg_y_0));
                }

                ArithGate::Cmul { x, c, z } => {
                    // zero labels for input x
                    let x_0 = low_labels
                        .get(x.id())
                        .expect("label index out of range")
                        .clone()
                        .expect("feed should be set.");
                    // set zero label for z
                    low_labels[z.id()] = Some(cmul_label(&x_0, *c as u64));
                }
                ArithGate::Mul {
                    x: node_x,
                    y: node_y,
                    z: node_z,
                } => {
                    let gate_num = self.gid;
                    self.gid += 1;

                    let x = low_labels
                        .get(node_x.id())
                        .expect("label index out of range")
                        .clone()
                        .expect("zero label should be set.");

                    let y = low_labels
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

                    let d_x = self.deltas.get(&q_x).expect("delta should be set");
                    let d_y = self.deltas.get(&q_y).expect("deltas should be set.");

                    // prepare empty block array as gate encoding
                    let mut gate = vec![Block::new([0; 16]); q_x as usize + q_y as usize - 2];

                    // TODO: add color method to wire
                    let r = y.color();

                    // convert (u64, u64) to Block
                    let g = tweak2(gate_num as u64, 0);

                    // X = H(A+aD) + arD such that a + A.color == 0
                    let alpha = (q_x - x.color()) % q_x; // alpha = -A.color
                    let x1 = add_label(&x, &cmul_label(&d_x, alpha as u64));

                    //
                    // // Y = H(B + bD) + (b + r)A such that b + B.color == 0
                    let beta = (q_y - y.color()) % q_y;
                    let y1 = add_label(&y, &cmul_label(&d_y, beta as u64));

                    let x_enc = self.cipher.tccr(g, x1.to_block());
                    let y_enc = self.cipher.tccr(g, y1.to_block());

                    let x_enc_label = add_label(
                        &LabelModN::from_block(x_enc, q_x),
                        &cmul_label(&d_x, (alpha as u64 * r as u64) % q_x as u64),
                    );
                    let y_enc_label = add_label(
                        &LabelModN::from_block(y_enc, q_x),
                        &cmul_label(&x, (beta as u64 + r as u64) % q_x as u64),
                    );

                    // precompute a lookup table of X.minus(&D_cmul[(a * r % q)])
                    //                            = X.plus(&D_cmul[((q - (a * r % q)) % q)])
                    let mut precomp = Vec::with_capacity(q_x as usize);
                    let mut t = x_enc_label.clone();
                    precomp.push(t.to_block());
                    for _ in 1..q_x {
                        t = add_label(&t, &d_x);
                        precomp.push(t.to_block())
                    }

                    // We can vectorize the hashes here too, but then we need to precompute all `q` sums of A
                    // with delta [A, A + D, A + D + D, etc.]
                    // Would probably need another alloc which isn't great
                    let mut t = x.clone();
                    for a in 0..q_x {
                        if a > 0 {
                            t = add_label(&t, &d_x);
                        }

                        // garbler's half-gate: outputs X-arD
                        // G = H(A+aD) ^ X+a(-r)D = H(A+aD) ^ X-arD
                        if t.color() != 0 {
                            gate[t.color() as usize - 1] = self.cipher.tccr(g, t.to_block())
                                ^ precomp[((q_x - (a * r % q_x)) % q_x) as usize]
                        }
                    }
                    precomp.clear();

                    // precompute a lookup table of Y.minus(&A_cmul[((b+r) % q)])
                    //                            = Y.plus(&A_cmul[((q - ((b+r) % q)) % q)])
                    let mut t = y_enc_label.clone();
                    precomp.push(t.to_block());
                    for _ in 1..q_x {
                        t = add_label(&t, &x);
                        precomp.push(t.to_block())
                    }

                    // Same note about vectorization as A
                    let mut t = y.clone();
                    for b in 0..q_y {
                        if b > 0 {
                            t = add_label(&t, &d_y);
                        }
                        // evaluator's half-gate: outputs Y-(b+r)D
                        // G = H(B+bD) + Y-(b+r)A
                        if t.color() != 0 {
                            gate[q_x as usize - 1 + t.color() as usize - 1] =
                                self.cipher.tccr(g, t.to_block())
                                    ^ precomp[((q_x - ((b + r) % q_x)) % q_x) as usize];
                        }
                    }

                    self.low_labels[node_z.id()] = Some(add_label(&x_enc_label, &y_enc_label));

                    // return encrypted mul gate
                    return Some(ArithEncryptedGate::new(gate));
                }
                ArithGate::Proj {
                    x: node_x,
                    tt,
                    z: node_z,
                } => {
                    // let tt = tt.ok_or(GarblerError::TruthTableRequired)?;
                    let x_low = low_labels
                        .get(node_x.id())
                        .expect("label index out of range")
                        .clone()
                        .expect("zero label should be set.");

                    let q_in = node_x.modulus();
                    let q_out = node_z.modulus();

                    let mut gate = vec![Block::new([0; 16]); q_in as usize - 1];
                    let tao = x_low.color();

                    let gate_num = self.gid;
                    self.gid += 1;

                    let g = tweak(gate_num);
                    let d_in = self
                        .deltas
                        .get(&q_in)
                        .expect(&format!("delta not set for {}", q_in));
                    let d_out = self
                        .deltas
                        .get(&q_out)
                        .expect(&format!("delta not set for {}", q_out));

                    // output zero-wire
                    // W_g^0 <- -H(g, W_{a_1}^0 - \tao\Delta_m) - \phi(-\tao)\Delta_n
                    // TODO: more readable
                    let z_low = add_label(
                        &LabelModN::from_block(
                            self.cipher.tccr(
                                g,
                                add_label(
                                    &x_low,
                                    &cmul_label(&d_in, (q_in as u64 - tao as u64) % q_in as u64),
                                )
                                .to_block(),
                            ),
                            q_out,
                        ),
                        &cmul_label(
                            &d_out,
                            (q_out as u64 - tt[((q_in - tao) % q_in) as usize] as u64)
                                % q_out as u64,
                        ),
                    );

                    // precompute `let C_ = C.plus(&Dout.cmul(tt[x as usize]))`
                    let z_precomputed = {
                        let mut z_tmp = z_low.clone();
                        (0..q_out)
                            .map(|x| {
                                if x > 0 {
                                    z_tmp = add_label(&z_tmp, &d_out);
                                }
                                z_tmp.to_block()
                            })
                            .collect::<Vec<Block>>()
                    };

                    let mut x_tmp = x_low.clone();
                    for x in 0..q_in {
                        if x > 0 {
                            x_tmp = add_label(&x_tmp, &d_in); // avoiding expensive cmul for `A_ = A.plus(&Din.cmul(x))`
                        }

                        let ix = (tao as usize + x as usize) % q_in as usize;
                        if ix == 0 {
                            continue;
                        }

                        let ct = self.cipher.tccr(g, x_tmp.to_block())
                            ^ z_precomputed[tt[x as usize] as usize];
                        gate[ix - 1] = ct;
                    }

                    low_labels[node_z.id()] = Some(z_low);

                    return Some(ArithEncryptedGate::new(gate));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use mpz_circuits::{
        arithmetic::{ops::add, utils::PRIMES},
        ArithmeticCircuit, ArithmeticCircuitBuilder,
    };

    use super::BMR16Generator;
    use crate::{encoding::ChaChaCrtEncoder, ArithEncryptedGate};

    fn adder_circ() -> ArithmeticCircuit {
        let builder = ArithmeticCircuitBuilder::default();
        let x = builder.add_input::<u32>("x".into()).unwrap();
        let y = builder.add_input::<u32>("y".into()).unwrap();

        let z = add(&mut builder.state().borrow_mut(), &x.repr, &y.repr).unwrap();
        builder.add_output(&z);

        builder.build().unwrap()
    }

    #[test]
    fn test_bmr16_generator() {
        let encoder = ChaChaCrtEncoder::new([0; 32], &PRIMES[0..10]);

        let circ = adder_circ();
        let deltas = encoder.deltas();
        let mul_count = circ.mul_count();

        let encoded_inputs = circ
            .inputs()
            .iter()
            .map(|inp| encoder.encode_by_len(0, inp.repr.len()))
            .collect();

        let mut gen = BMR16Generator::new(Arc::new(circ), deltas.clone(), encoded_inputs).unwrap();
        let enc_gates: Vec<ArithEncryptedGate> = gen.by_ref().collect();

        assert!(gen.is_complete());
        assert_eq!(enc_gates.len(), mul_count);

        gen.outputs().unwrap();
        gen.hash().unwrap();
    }
}
