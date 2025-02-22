use std::ops::Index;

use mpz_core::Block;
use serde::{Deserialize, Serialize};

use crate::EncodingCommitment;

/// Encrypted gate truth table
///
/// For the half-gate garbling scheme a truth table will typically have 2 rows, except for in
/// privacy-free garbling mode where it will be reduced to 1.
///
/// We do not yet support privacy-free garbling.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EncryptedGate(#[serde(with = "serde_arrays")] pub(crate) [Block; 2]);

impl EncryptedGate {
    pub(crate) fn new(inner: [Block; 2]) -> Self {
        Self(inner)
    }

    pub(crate) fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[..16].copy_from_slice(&self.0[0].to_bytes());
        bytes[16..].copy_from_slice(&self.0[1].to_bytes());
        bytes
    }
}

impl Index<usize> for EncryptedGate {
    type Output = Block;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// A garbled circuit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarbledCircuit {
    /// Encrypted gates of the circuit
    pub gates: Vec<EncryptedGate>,
    /// Encoding commitments of the circuit outputs
    pub commitments: Option<Vec<EncodingCommitment>>,
}
/// Encrypted gate truth table for arithmetic gate
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArithEncryptedGate(pub(crate) Vec<Block>);

impl ArithEncryptedGate {
    pub(crate) fn new(inner: Vec<Block>) -> Self {
        Self(inner)
    }

    pub(crate) fn get(&self, i: usize) -> Option<&Block> {
        self.0.get(i)
    }
}

impl std::fmt::Display for ArithEncryptedGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let res = self
            .0
            .iter()
            .map(|b| hex::encode(b.to_bytes()))
            .collect::<Vec<_>>()
            .join(",\n\t");
        write!(f, "{}", res)
    }
}
