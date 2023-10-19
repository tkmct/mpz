use std::collections::HashMap;

use rand::{CryptoRng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use mpz_circuits::arithmetic::{
    utils::{digits_per_u128, is_power_of_2, PRIMES},
    TypeError,
};

use crate::encoding::{utils::unrank, Block};

const DELTA_STREAM_ID: u64 = u64::MAX;

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("Not enough decoding info provided for output idx {0} mod {1}")]
    LackOfDecodingInfo(usize, u16),
    #[error(transparent)]
    TypeError(#[from] TypeError),
}

pub(crate) fn add_label(x: &LabelModN, y: &LabelModN) -> LabelModN {
    debug_assert_eq!(x.modulus, y.modulus);
    debug_assert_eq!(x.inner.len(), y.inner.len());

    let q = x.modulus;
    let z_inner = x
        .iter()
        .zip(y.iter())
        .map(|(x_n, y_n)| {
            let (zp, overflow) = (x_n + y_n).overflowing_sub(q);
            if overflow {
                x_n + y_n
            } else {
                zp
            }
        })
        .collect();

    LabelModN::new(z_inner, q)
}

pub(crate) fn cmul_label(x: &LabelModN, c: u64) -> LabelModN {
    let q = x.modulus;
    let z_inner = x
        .iter()
        .map(|d| (*d as u32 * c as u32 % q as u32) as u16)
        .collect();

    LabelModN::new(z_inner, q)
}

/// mod N label.
/// Label is represented by vector of elements in Z_n.
/// Length of the label is defined by security parameter λ
/// λ_m = floor(λ/log_m)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelModN {
    /// inner representation of the label
    inner: Vec<u16>,
    /// modulus of the wire label
    modulus: u16,
}

/// These codes are brought from swanky
impl LabelModN {
    /// create new LabelModN instance from inner vec and modulus
    pub fn new(inner: Vec<u16>, modulus: u16) -> Self {
        Self { inner, modulus }
    }

    /// convert label into block
    pub fn to_block(&self) -> Block {
        let mut x = 0u128;

        for &d in self.inner.iter().rev() {
            let (xp, overflow) = x.overflowing_mul(self.modulus.into());
            debug_assert!(!overflow, "overflow!!!! x={}", x);
            x = xp + d as u128;
        }

        Block::from(x.to_le_bytes())
    }

    /// convert block to label
    pub fn from_block(block: Block, modulus: u16) -> Self {
        let inner = if is_power_of_2(modulus) {
            // It's a power of 2, just split the digits.
            let ndigits = digits_per_u128(modulus);
            let width = 128 / ndigits;
            let mask = (1 << width) - 1;
            let x = u128::from_le_bytes(block.into());
            (0..ndigits)
                .map(|i| ((x >> (width * i)) & mask) as u16)
                .collect::<Vec<u16>>()
        } else {
            unrank(u128::from_le_bytes(block.into()), modulus)
        };

        Self { modulus, inner }
    }

    /// generate random label for given modulus
    pub fn random<R: Rng + CryptoRng + ?Sized>(rng: &mut R, modulus: u16) -> Self {
        let mut block = Block::random(rng);
        block.set_lsb();
        Self::from_block(block, modulus)
    }

    /// Iterate over inner vec
    pub fn iter(&self) -> Box<dyn Iterator<Item = &u16> + '_> {
        Box::new(self.inner.iter())
    }
}

/// Delta for generalized Free-XOR
pub type CrtDelta = LabelModN;

/// Encoding state for CRT representation
/// TODO: because we don't keep delta inside state, there might be better way to express
/// label state rather than using struct. we can just use enum?
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
    _state: S,
    labels: Vec<LabelModN>,
}

impl Labels<state::Full> {
    /// Create new Labels instance from vector of LabelModN
    pub fn new(labels: Vec<LabelModN>) -> Self {
        Self {
            // TODO: add deltas here?
            _state: state::Full {},
            labels,
        }
    }

    /// Create active label from values.
    /// Each value corresponds to labels.
    /// TODO: move this into generator?
    pub fn select(
        &self,
        deltas: &HashMap<u16, CrtDelta>,
        values: Vec<u16>,
    ) -> Labels<state::Active> {
        let labels: Vec<LabelModN> = values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let q = PRIMES[i];
                let d = deltas.get(&q).expect("Delta should be set for given prime");

                add_label(&self.labels[i], &cmul_label(d, *v as u64))
            })
            .collect();

        Labels::<state::Active>::new(labels)
    }
}

impl Labels<state::Active> {
    fn new(labels: Vec<LabelModN>) -> Self {
        Self {
            _state: state::Active {},
            labels,
        }
    }
}

/// Encoded CRT Value.
#[derive(Debug, Clone)]
pub struct EncodedCrtValue<S: LabelState>(Labels<S>);

impl<S: LabelState> EncodedCrtValue<S> {
    /// returns iterator of Labels.
    pub(crate) fn iter(&self) -> Box<dyn Iterator<Item = &LabelModN> + '_> {
        Box::new(self.0.labels.iter())
    }

    /// returns length of labels
    #[allow(clippy::len_without_is_empty)]
    pub(crate) fn len(&self) -> usize {
        self.0.labels.len()
    }
}

impl EncodedCrtValue<state::Full> {
    /// retrieve actual label value using zero label and delta
    pub fn select(
        &self,
        deltas: &HashMap<u16, CrtDelta>,
        values: Vec<u16>,
    ) -> EncodedCrtValue<state::Active> {
        EncodedCrtValue(self.0.select(deltas, values))
    }
}

impl From<Vec<LabelModN>> for EncodedCrtValue<state::Full> {
    fn from(labels: Vec<LabelModN>) -> Self {
        Self(Labels::<state::Full>::new(labels))
    }
}

impl From<Vec<LabelModN>> for EncodedCrtValue<state::Active> {
    fn from(labels: Vec<LabelModN>) -> Self {
        Self(Labels::<state::Active>::new(labels))
    }
}

/// Chacha encoder for CRT representation
pub struct ChaChaCrtEncoder<const N: usize> {
    seed: [u8; 32],
    deltas: HashMap<u16, CrtDelta>,
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
            deltas.insert(*p, CrtDelta::random(&mut rng, *p));
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
    pub fn deltas(&self) -> &HashMap<u16, CrtDelta> {
        &self.deltas
    }

    /// create encoded labels
    pub fn encode_by_len(&self, id: u64, len: usize) -> EncodedCrtValue<state::Full> {
        let mut rng = self.get_rng(id);

        let labels = Block::random_vec(&mut rng, len)
            .into_iter()
            .enumerate()
            .map(|(i, block)| LabelModN::from_block(block, PRIMES[i]))
            .collect::<Vec<_>>();

        EncodedCrtValue::<state::Full>::from(labels)
    }
}

#[derive(Debug, Clone)]
pub struct CrtDecoding(Vec<Vec<LabelModN>>);

impl CrtDecoding {
    pub fn new(inner: Vec<Vec<LabelModN>>) -> Self {
        Self(inner)
    }

    pub fn get(&self, i: usize, j: usize) -> Option<&LabelModN> {
        self.0.get(i)?.get(j)
    }
}

pub(crate) fn output_tweak(i: usize, k: u16) -> Block {
    let (left, _) = (i as u128).overflowing_shl(64);
    Block::from((left + k as u128).to_le_bytes())
}
