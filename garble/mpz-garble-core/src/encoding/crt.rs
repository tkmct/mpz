use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::encoding::{Block, Delta, Label};

use mpz_circuits::arithmetic::{types::Fp, utils::PRIMES};

const DELTA_STREAM_ID: u64 = u64::MAX;

#[derive(Debug, thiserror::Error)]
pub(crate) struct DecodeError {
    msg: String,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DecodeError: {}", self.msg)
    }
}

pub(crate) fn add_block(x: &Block, y: &Block) -> Block {
    let inner: [u8; 16] = x
        .to_bytes()
        .iter()
        .zip(y.to_bytes())
        .map(|(x_b, y_b)| x_b.wrapping_add(y_b))
        .collect::<Vec<u8>>()
        .try_into()
        .expect("block length should match");
    Block::new(inner)
}

pub(crate) fn cmul_block(x: &Block, c: Fp) -> Block {
    let inner: [u8; 16] = x
        .to_bytes()
        .iter()
        .map(|x_b| ((*x_b as u64).wrapping_mul(c.0 as u64) % (u8::MAX as u64)) as u8)
        .collect::<Vec<u8>>()
        .try_into()
        .expect("block length should match");
    Block::new(inner)
}

/// Encoding state for CRT representation
pub mod state {
    /// Label state trait
    pub trait LabelState: std::marker::Copy {}

    /// Full state
    #[derive(Debug, Clone, Copy)]
    pub struct Full {}

    impl LabelState for Full {}

    /// Active state
    #[derive(Debug, Clone, Copy)]
    pub struct Active {}

    impl LabelState for Active {}
}

use state::*;

/// Set of labels.
/// This struct corresponds to one CrtValue
#[derive(Debug, Clone)]
pub struct Labels<S: LabelState> {
    state: S,
    labels: Vec<Label>,
}

impl Labels<state::Full> {
    pub(crate) fn new(labels: Vec<Label>) -> Self {
        Self {
            // TODO: add deltas here?
            state: state::Full {},
            labels,
        }
    }

    /// Create active label from values.
    /// Each value corresponds to labels.
    pub(crate) fn select(
        &self,
        deltas: &HashMap<u16, Delta>,
        values: Vec<u16>,
    ) -> Labels<state::Active> {
        let labels: Vec<Label> = values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let q = PRIMES[i];
                let d = deltas.get(&q).expect("Delta should be set for given prime");

                // TODO: is this okay?
                Label::new(add_block(
                    &self.labels[0].to_inner(),
                    &cmul_block(&d.0, Fp(*v as u32)),
                ))
            })
            .collect();

        Labels::<state::Active>::new(labels)
    }
}

impl Labels<state::Active> {
    fn new(labels: Vec<Label>) -> Self {
        Self {
            state: state::Active {},
            labels,
        }
    }
}

/// Encoded CRT Value.
#[derive(Debug, Clone)]
pub struct EncodedCrtValue<S: LabelState>(Labels<S>);

impl<S: LabelState> EncodedCrtValue<S> {
    /// Create new instance.
    pub(crate) fn new(labels: Labels<S>) -> Self {
        EncodedCrtValue(labels)
    }
    /// returns iterator of Labels.
    pub(crate) fn iter(&self) -> Box<dyn Iterator<Item = &Label> + '_> {
        Box::new(self.0.labels.iter())
    }

    /// returns length of labels
    #[allow(clippy::len_without_is_empty)]
    pub(crate) fn len(&self) -> usize {
        self.0.labels.len()
    }
}

impl EncodedCrtValue<state::Full> {
    pub(crate) fn select(
        &self,
        deltas: &HashMap<u16, Delta>,
        values: Vec<u16>,
    ) -> EncodedCrtValue<state::Active> {
        EncodedCrtValue(self.0.select(deltas, values))
    }

    pub(crate) fn decoding(&self) -> CrtDecoding {
        // hash output
        //
        // fn output(&mut self, X: &Wire) -> Result<Option<u16>, GarblerError> {
        //
        // let q = X.modulus();
        // let i = self.current_output();
        // let D = self.delta(q);
        // for k in 0..q {
        //     let block = X.plus(&D.cmul(k)).hash(output_tweak(i, k));
        //     self.channel.write_block(&block)?;
        // }
        // Ok(None)
        self.0;

        todo!()
    }
}

impl EncodedCrtValue<state::Active> {
    pub(crate) fn decode(&self, decoding: &CrtDecoding) -> Result<Fp, DecodeError> {
        // fn output(&mut self, x: &Wire) -> Result<Option<u16>, EvaluatorError> {
        // let q = x.modulus();
        // let i = self.current_output();
        //
        // // Receive the output ciphertext from the garbler
        // let ct = self.channel.read_blocks(q as usize)?;
        //
        // // Attempt to brute force x using the output ciphertext
        // let mut decoded = None;
        // for k in 0..q {
        //     let hashed_wire = x.hash(output_tweak(i, k));
        //     if hashed_wire == ct[k as usize] {
        //         decoded = Some(k);
        //         break;
        //     }
        // }
        //
        // if let Some(output) = decoded {
        //     Ok(Some(output))
        // } else {
        //     Err(EvaluatorError::DecodingFailed)
        // }

        todo!()
    }
}

impl From<Vec<Label>> for EncodedCrtValue<state::Full> {
    fn from(labels: Vec<Label>) -> Self {
        Self(Labels::<state::Full>::new(labels))
    }
}

impl From<Vec<Label>> for EncodedCrtValue<state::Active> {
    fn from(labels: Vec<Label>) -> Self {
        Self(Labels::<state::Active>::new(labels))
    }
}

/// Chacha encoder for CRT representation
pub struct ChaChaCrtEncoder<const N: usize> {
    seed: [u8; 32],
    deltas: HashMap<u16, Delta>,
}

impl<const N: usize> ChaChaCrtEncoder<N> {
    /// Create new encoder for CRT labels with provided seed.
    pub fn new(seed: [u8; 32]) -> Self {
        let mut rng = ChaCha20Rng::from_seed(seed);

        // Stream id u64::MAX is reserved to generate delta.
        // This way there is only ever 1 delta per seed
        rng.set_stream(DELTA_STREAM_ID);
        let mut deltas = HashMap::new();
        PRIMES.iter().take(N).for_each(|p| {
            deltas.insert(*p, Delta::random(&mut rng));
        });

        Self { seed, deltas }
    }

    // copied from encoding/encoder.rs
    fn get_rng(&self, id: u64) -> ChaCha20Rng {
        if id == DELTA_STREAM_ID {
            panic!("stream id {} is reserved", DELTA_STREAM_ID);
        }

        let mut rng = ChaCha20Rng::from_seed(self.seed);
        rng.set_stream(id);
        rng.set_word_pos(0);

        rng
    }

    /// Returns random number generator seed value.
    pub fn seed(&self) -> Vec<u8> {
        self.seed.to_vec()
    }

    /// Returns list of deltas used in a circuit.
    pub fn deltas(&self) -> &HashMap<u16, Delta> {
        &self.deltas
    }

    /// create encoded labels
    pub fn encode_by_len(&self, id: u64, len: usize) -> EncodedCrtValue<state::Full> {
        let mut rng = self.get_rng(id);

        let labels = Block::random_vec(&mut rng, len)
            .into_iter()
            .map(Label::new)
            .collect::<Vec<_>>();

        EncodedCrtValue::<state::Full>::from(labels)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CrtDecoding {}
