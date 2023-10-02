use crate::{Feed, Sink};
use std::marker::PhantomData;

/// TODO: make these values generic over circuit struct or trait.
/// Default crt moduli
pub(crate) const CRT_MODULUI: [u16; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];

/// A node in an arithmetic circuit.
/// Node represent a single wire with specific modulus.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArithNode<T> {
    id: usize,
    modulus: u16,
    _pd: PhantomData<T>,
}

impl<T> ArithNode<T> {
    /// Create a new ArithNode instance with given modulus.
    pub fn new(id: usize, modulus: u16) -> Self {
        ArithNode {
            id,
            modulus,
            _pd: PhantomData,
        }
    }

    /// Returns the id of the node.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns he modulus of the node.
    pub fn modulus(&self) -> u16 {
        self.modulus
    }
}

impl From<ArithNode<Feed>> for ArithNode<Sink> {
    fn from(node: ArithNode<Feed>) -> ArithNode<Sink> {
        Self {
            id: node.id,
            modulus: node.modulus,
            _pd: PhantomData,
        }
    }
}

impl From<&ArithNode<Feed>> for ArithNode<Sink> {
    fn from(node: &ArithNode<Feed>) -> ArithNode<Sink> {
        Self {
            id: node.id,
            modulus: node.modulus,
            _pd: PhantomData,
        }
    }
}

impl From<ArithNode<Sink>> for ArithNode<Feed> {
    fn from(node: ArithNode<Sink>) -> ArithNode<Feed> {
        Self {
            id: node.id,
            modulus: node.modulus,
            _pd: PhantomData,
        }
    }
}

impl From<&ArithNode<Sink>> for ArithNode<Feed> {
    fn from(node: &ArithNode<Sink>) -> ArithNode<Feed> {
        Self {
            id: node.id,
            modulus: node.modulus,
            _pd: PhantomData,
        }
    }
}

/// Field value type.
/// T must implement field operations
/// Should we fix field size for now?
/// FIXME: make Fp generic for any field point?
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fp(pub u32);
