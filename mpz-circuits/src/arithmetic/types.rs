use crate::{Feed, Sink};
use std::marker::PhantomData;

/// TODO: make these values generic over circuit struct or trait.
/// Default crt moduli
pub(crate) const CRT_MODULUI: [u16; 10] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29];

/// A node in an arithmetic circuit.
/// Node represent a set of wires represented as CRT.
// TODO: in order to make efficient circuit, make this generic over number of moduli: currently set to 10
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ArithNode<T> {
    id: usize,
    moduli: [u16; 10],
    _pd: PhantomData<T>,
}

impl<T> ArithNode<T> {
    /// Create a new ArithNode instance with given moduli.
    pub fn new(id: usize, moduli: [u16; 10]) -> Self {
        ArithNode {
            id,
            moduli,
            _pd: PhantomData,
        }
    }

    /// Create a new ArithNode instance with default CRT moduli.
    /// This moduli can represent up to 6469693230 (> 2^32).
    pub fn new_u32(id: usize) -> Self {
        Self::new(id, CRT_MODULUI)
    }

    /// Returns the id of the node.
    pub fn id(&self) -> usize {
        self.id
    }
}

impl From<ArithNode<Feed>> for ArithNode<Sink> {
    fn from(node: ArithNode<Feed>) -> ArithNode<Sink> {
        Self {
            id: node.id,
            moduli: node.moduli,
            _pd: PhantomData,
        }
    }
}

impl From<&ArithNode<Feed>> for ArithNode<Sink> {
    fn from(node: &ArithNode<Feed>) -> ArithNode<Sink> {
        Self {
            id: node.id,
            moduli: node.moduli,
            _pd: PhantomData,
        }
    }
}

impl From<ArithNode<Sink>> for ArithNode<Feed> {
    fn from(node: ArithNode<Sink>) -> ArithNode<Feed> {
        Self {
            id: node.id,
            moduli: node.moduli,
            _pd: PhantomData,
        }
    }
}

impl From<&ArithNode<Sink>> for ArithNode<Feed> {
    fn from(node: &ArithNode<Sink>) -> ArithNode<Feed> {
        Self {
            id: node.id,
            moduli: node.moduli,
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
