use bytemuck::cast;

use mpz_core::{
    aes::{FixedKeyAes, FIXED_KEY_AES},
    commit::HashCommit,
};
use rand::{CryptoRng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};

use mpz_circuits::arithmetic::{
    types::{ArithValue, CrtValueType},
    utils::{convert_crt_to_value, digits_per_u128, get_index_of_prime, is_power_of_2, PRIMES},
    TypeError,
};

use crate::encoding::{utils::unrank, Block};

const DELTA_STREAM_ID: u64 = u64::MAX;

#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum DecodeError {
    #[error("Not enough decoding info provided for output idx {0} mod {1}")]
    LackOfDecodingInfo(usize, u16),
    #[error(transparent)]
    TypeError(#[from] TypeError),
}

/// Add two LabelModN
pub fn add_label(x: &LabelModN, y: &LabelModN) -> LabelModN {
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

/// Negate label
pub fn negate_label(x: &LabelModN) -> LabelModN {
    let q = x.modulus;
    let z_inner = x
        .iter()
        .map(|x_n| if *x_n > 0 { q - *x_n } else { 0 })
        .collect();

    LabelModN::new(z_inner, q)
}

/// Cmul LabelModN with constant value
pub fn cmul_label(x: &LabelModN, c: u64) -> LabelModN {
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

    /// Create a zero LabelModN instance
    pub fn zero(modulus: u16) -> Self {
        let len = digits_per_u128(modulus);
        Self {
            inner: vec![0; len],
            modulus,
        }
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

    /// generate random delta label
    pub fn random_delta<R: Rng + CryptoRng + ?Sized>(rng: &mut R, modulus: u16) -> Self {
        let mut r = Self::random(rng, modulus);
        r.inner[0] = 1;
        r
    }

    /// Iterate over inner vec
    pub fn iter(&self) -> Box<dyn Iterator<Item = &u16> + '_> {
        Box::new(self.inner.iter())
    }

    /// Return color digit of the label
    pub fn color(&self) -> u16 {
        let color = self.inner[0];
        debug_assert!(color < self.modulus);
        color
    }

    /// Return the modulus of label
    pub fn modulus(&self) -> u16 {
        self.modulus
    }
}

impl Default for LabelModN {
    fn default() -> Self {
        LabelModN::new(vec![], 0)
    }
}

impl std::fmt::Display for LabelModN {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Label(q:{}, {:#?})",
            self.modulus,
            hex::encode(self.inner.clone().hash_commit().1.as_bytes())
        )
    }
}

/// Delta for generalized Free-XOR
pub type CrtDelta = LabelModN;

/// Encoding state for CRT representation
/// TODO: because we don't keep delta inside state, there might be better way to express
/// label state rather than using struct. we can just use enum?
pub mod state {
    use serde::{Deserialize, Serialize};

    /// Label state trait
    pub trait LabelState: std::marker::Copy {}

    /// Full state
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Full {}

    impl LabelState for Full {}

    /// Active state
    #[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
    pub struct Active {}

    impl LabelState for Active {}
}

use state::*;

/// Set of labels.
/// This struct corresponds to one CrtValue
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Labels<S: LabelState> {
    _state: S,
    labels: Vec<LabelModN>,
}

impl<S: LabelState> Labels<S> {
    /// Return reference to inner labels
    pub fn inner(&self) -> &Vec<LabelModN> {
        &self.labels
    }
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
}

impl Labels<state::Active> {
    /// create new active encoding label.
    pub fn new(labels: Vec<LabelModN>) -> Self {
        Self {
            _state: state::Active {},
            labels,
        }
    }
}

impl<T: LabelState> std::fmt::Display for Labels<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let res = self
            .labels
            .iter()
            .map(|label| format!("\n{}", label))
            .collect::<Vec<_>>()
            .concat();
        write!(f, "{}", res)
    }
}

/// Encoded CRT Value.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub struct EncodedCrtValue<S: LabelState>(Labels<S>);

impl<S: LabelState> EncodedCrtValue<S> {
    /// returns iterator of Labels.
    pub fn iter(&self) -> Box<dyn Iterator<Item = &LabelModN> + '_> {
        Box::new(self.0.labels.iter())
    }

    /// returns length of labels
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.0.labels.len()
    }

    /// Returns i-th inner label if exists
    pub(crate) fn get_label(&self, i: usize) -> Option<&LabelModN> {
        self.0.inner().get(i)
    }

    /// Returns decoding info of this value
    pub fn decoding(&self, idx: usize, deltas: &[CrtDelta], cipher: &FixedKeyAes) -> CrtDecoding {
        let hashes = self
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let q = PRIMES[i];
                let d =
                    get_delta_by_modulus(deltas, q).expect("delta should be set for given prime");

                (0..q)
                    .map(|k| {
                        let label = add_label(x, &cmul_label(&d, k as u64));
                        let tweak = output_tweak(idx, k);
                        LabelModN::from_block(cipher.tccr(tweak, label.to_block()), q)
                    })
                    .collect::<Vec<LabelModN>>()
            })
            .collect();

        CrtDecoding::new(hashes)
    }
}

impl EncodedCrtValue<state::Full> {
    /// Create active label from values.
    /// Each value corresponds to labels.
    ///
    /// # Arguments
    ///
    /// - `values` - The actual values for individual modulus from which encoded crt value is created.
    pub fn select(&self, deltas: &[CrtDelta], values: Vec<u16>) -> EncodedCrtValue<state::Active> {
        let active_labels: Vec<LabelModN> = values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let q = PRIMES[i];
                let d =
                    get_delta_by_modulus(deltas, q).expect("Delta should be set for given prime");

                add_label(
                    // TODO: not use unwrap here
                    self.get_label(i).unwrap(),
                    &cmul_label(&d, *v as u64),
                )
            })
            .collect();

        EncodedCrtValue::<state::Active>::from(active_labels)
    }
}

impl EncodedCrtValue<state::Active> {
    /// Decode active encoded value
    ///
    /// # Arguments
    ///
    /// - `id` - id of the output. used to tweak output.
    /// - `decoding` - Decoding info used to check decoded value integrity.
    pub fn decode(&self, id: usize, decoding: &CrtDecoding) -> Result<ArithValue, DecodeError> {
        let cipher = &FIXED_KEY_AES;

        let decoded_nodes: Vec<u16> = self
            .iter()
            .enumerate()
            .map(|(i, label)| {
                let q = PRIMES[i];
                let mut decoded = None;

                for k in 0..q {
                    let tweak = output_tweak(id, k);
                    let actual = LabelModN::from_block(cipher.tccr(tweak, label.to_block()), q);
                    let dec = decoding.get(i, k as usize).unwrap_or_else(|| {
                        panic!("Decoding should be set for.{}, {}, {}", id, i, k)
                    });

                    if actual == *dec {
                        decoded = Some(k);
                        break;
                    }
                }
                decoded.ok_or(DecodeError::LackOfDecodingInfo(id, q))
            })
            .collect::<Result<Vec<u16>, DecodeError>>()?;

        convert_crt_to_value(self.len(), &decoded_nodes).map_err(|e| e.into())
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

impl<T: LabelState> std::fmt::Display for EncodedCrtValue<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "EncodedCrtValue {}", self.0)
    }
}

/// Chacha encoder for CRT representation
pub struct ChaChaCrtEncoder<const N: usize> {
    seed: [u8; 32],
    deltas: [CrtDelta; N],
}

impl<const N: usize> ChaChaCrtEncoder<N> {
    /// Create new encoder for CRT labels with provided seed.
    /// * `seed` - seed value of encoder
    /// * `num_wire` - maximum number of wires used in circuit
    pub fn new(seed: [u8; 32]) -> Self {
        let mut rng = ChaCha20Rng::from_seed(seed);

        // Stream id u64::MAX is reserved to generate delta.
        // This way there is only ever 1 delta per seed
        rng.set_stream(DELTA_STREAM_ID);
        let deltas: [CrtDelta; N] =
            std::array::from_fn(|i| CrtDelta::random_delta(&mut rng, PRIMES[i]));

        Self { seed, deltas }
    }

    /// copied from encoding/encoder.rs
    pub fn get_rng(&self, id: u64) -> ChaCha20Rng {
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
    pub fn deltas(&self) -> &[CrtDelta; N] {
        &self.deltas
    }

    /// create encoded labels based on given type
    pub fn encode_by_type(&self, id: u64, ty: CrtValueType) -> EncodedCrtValue<state::Full> {
        match ty {
            CrtValueType::Bool => self.encode_by_len(id, 1),
            CrtValueType::U32 => self.encode_by_len(id, 10),
            _ => panic!("invalid type: {:?}", ty),
        }
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

/// Decoding info used to decode output values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrtDecoding(Vec<Vec<LabelModN>>);

impl CrtDecoding {
    /// Create a decoding isntance from vec of vec of label
    pub fn new(inner: Vec<Vec<LabelModN>>) -> Self {
        Self(inner)
    }

    /// get label by indcies
    pub fn get(&self, i: usize, j: usize) -> Option<&LabelModN> {
        self.0.get(i)?.get(j)
    }
}

impl std::fmt::Display for CrtDecoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut res = String::new();
        res.push_str(
            &self
                .0
                .iter()
                .map(|inner| {
                    inner
                        .iter()
                        .map(|label| format!("\t{}\n", label))
                        .collect::<Vec<_>>()
                        .concat()
                })
                .collect::<Vec<_>>()
                .concat(),
        );

        write!(f, "CrtDecoding\n{}", res)
    }
}

// Following codes are based on swanky
pub(crate) fn output_tweak(i: usize, k: u16) -> Block {
    let (left, _) = (i as u128).overflowing_shl(64);
    Block::from((left + k as u128).to_le_bytes())
}

pub(crate) fn tweak(i: usize) -> Block {
    Block::from(cast::<u64, [u8; 16]>(i as u64))
}

pub(crate) fn tweak2(i: u64, j: u64) -> Block {
    let a = [i, j];
    Block::new(cast::<[u64; 2], [u8; 16]>(a))
}

/// get delta corresponding to the modulus given.
/// returns None if given value is not prime
/// panic if the delta doesn't exist for given modulus value
///
/// * `deltas` - slice of delta values sorted by size of modulus
/// * `q` - modulus to get delta
pub fn get_delta_by_modulus(deltas: &[CrtDelta], q: u16) -> Option<CrtDelta> {
    let idx = get_index_of_prime(q)?;
    Some(deltas[idx].clone())
}
