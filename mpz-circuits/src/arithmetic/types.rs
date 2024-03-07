//! Types used for Arithmetic value representation in arithmetic circuit.
use std::marker::PhantomData;

use crate::{
    arithmetic::utils::{NPRIMES, PRIMES},
    {Feed, Sink},
};

/// An error related to binary type conversions.
#[derive(Debug, PartialEq, thiserror::Error)]
#[allow(missing_docs)]
pub enum TypeError {
    #[error("Invalid crt representation length: expected: {expected}, actual: {actual}")]
    InvalidLength { expected: usize, actual: usize },

    #[error("Crt representation length does not mach: got {0} and {1}")]
    UnequalLength(usize, usize),

    #[error("The conversion exceeds the size of destination")]
    ExceedSize,
}

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

/// Crt representation type.
/// TODO: we have to handle generic Crt<N> type as CrtRepr.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CrtRepr {
    /// Boolean type of CRT representation. represented with 1 wire which describe 0/1.
    Bool(CrtValue<1>),
    /// U32 type of CRT representation. u32 can be represented using at least first 10 primes with CRT.
    U32(CrtValue<10>),
}

impl CrtRepr {
    /// returns length of moduli.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            CrtRepr::Bool(_) => 1,
            CrtRepr::U32(_) => 10,
        }
    }

    /// Iterate through ArithNode.
    pub fn iter(&self) -> Box<dyn Iterator<Item = &ArithNode<Feed>> + '_> {
        match self {
            CrtRepr::Bool(v) => Box::new(v.0.iter()),
            CrtRepr::U32(v) => Box::new(v.0.iter()),
        }
    }

    /// Retuns a slice of moduli
    pub fn moduli(&self) -> &[u16] {
        match self {
            CrtRepr::Bool(v) => v.moduli(),
            CrtRepr::U32(v) => v.moduli(),
        }
    }
}

/// Trait for convertable to CrtRepr
pub trait ToCrtRepr {
    /// create new CrtRepr instance from slice of ArithNode
    fn new_crt_repr(nodes: &[ArithNode<Feed>]) -> Result<CrtRepr, TypeError>;
}

/// Crt length trait
#[allow(missing_docs)]
pub trait CrtLen {
    const LEN: usize;
}

impl ToCrtRepr for bool {
    fn new_crt_repr(nodes: &[ArithNode<Feed>]) -> Result<CrtRepr, TypeError> {
        Ok(CrtRepr::Bool(CrtValue::<1>::new(
            nodes.try_into().map_err(|_| TypeError::InvalidLength {
                actual: nodes.len(),
                expected: 1,
            })?,
        )))
    }
}

impl CrtLen for bool {
    const LEN: usize = 1;
}

impl ToCrtRepr for u32 {
    fn new_crt_repr(nodes: &[ArithNode<Feed>]) -> Result<CrtRepr, TypeError> {
        Ok(CrtRepr::U32(CrtValue::<10>::new(
            nodes.try_into().map_err(|_| TypeError::InvalidLength {
                actual: nodes.len(),
                expected: 10,
            })?,
        )))
    }
}

impl CrtLen for u32 {
    const LEN: usize = 10;
}

/// CRT representation of a field element in circuit.
/// This bundles crt wires. each wire has modulus and unique id in a circuit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CrtValue<const N: usize>([ArithNode<Feed>; N]);

impl<const N: usize> CrtValue<N> {
    pub(crate) fn new(nodes: [ArithNode<Feed>; N]) -> CrtValue<N> {
        // check if N is less than NPRIMES
        // There is unstable feature to do this compile time using generic_const_expr.
        assert!(N <= NPRIMES, "Const N should be less than NPRIMES");
        CrtValue(nodes)
    }

    /// generate CrtRepr which takes N ids starting from given id.
    /// eg. CrtRepr<5>::new_from_id(4) will generate CrtRepr with ArithNodes
    /// having the following id and moduli pairs
    /// [(2,4), (3,5), (5,6), (7,7), (11,8)]
    #[allow(dead_code)]
    pub fn new_from_id(id: usize) -> CrtValue<N> {
        let mut nodes = [ArithNode::<Feed>::new(0, 0); N];
        for (i, p) in (0..N).zip(PRIMES) {
            nodes[i] = ArithNode::<Feed>::new(id + i, p);
        }

        CrtValue(nodes)
    }

    pub(crate) fn nodes(&self) -> [ArithNode<Feed>; N] {
        self.0
    }

    /// Returns the moduli array
    pub(crate) fn moduli(&self) -> &[u16; N] {
        // Unwrapping is safe because N is always less than NPRIMES.
        TryFrom::try_from(&PRIMES[..N]).unwrap()
    }

    /// Returns the length of moduli array
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns value type of the CrtValue
    pub fn value_type(&self) -> CrtValueType {
        let len = self.len();
        match len {
            1 => CrtValueType::Bool,
            10 => CrtValueType::U32,
            _ => panic!("Not supported type."),
        }
    }
}

/// A crt value type that can be encoded into a binary representation.
/// This just defines number of wires used to represent a value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum CrtValueType {
    Bool,
    U32,
}

impl CrtValueType {
    /// returns the length of moduli array
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            CrtValueType::Bool => 1,
            CrtValueType::U32 => 10,
        }
    }
}

/// Actual arithmetic value which can be encoded into Crt representation
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(missing_docs)]
pub enum ArithValue {
    Bool(bool),
    P10(u64),
}

impl ArithValue {
    /// Returns CrtValueType of this arith value.
    pub fn value_type(&self) -> CrtValueType {
        match self {
            Self::Bool(_) => CrtValueType::Bool,
            Self::P10(_) => CrtValueType::U32,
        }
    }

    /// Returns length of the value represented in CRT
    pub fn num_wire(&self) -> usize {
        match self {
            Self::Bool(_) => 1,
            Self::P10(_) => 10,
        }
    }
}

impl From<u32> for ArithValue {
    fn from(value: u32) -> Self {
        ArithValue::P10(value as u64)
    }
}

impl TryFrom<u64> for ArithValue {
    type Error = TypeError;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        let p: u64 = PRIMES[0..10].iter().fold(1, |acc, x| acc * *x as u64);
        if value >= p {
            return Err(TypeError::ExceedSize);
        }

        Ok(ArithValue::P10(value))
    }
}

/// Mixed radix value
#[derive(Debug, Clone, PartialEq)]
pub struct MixedRadixValue(Vec<ArithNode<Feed>>);

impl MixedRadixValue {
    pub(crate) fn new(inner: Vec<ArithNode<Feed>>) -> Self {
        Self(inner)
    }

    /// Returns the length of moduli array
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns the moduli array
    pub(crate) fn moduli(&self) -> Vec<u16> {
        self.0.iter().map(|i| i.modulus).collect::<Vec<_>>()
    }

    pub(crate) fn wires(&self) -> &[ArithNode<Feed>] {
        &self.0
    }
}

/// Circuit input type which holds
/// Crt representation and name
#[derive(Debug, Clone, PartialEq)]
pub struct CircInput {
    /// Name of the input
    pub name: String,
    /// Representation of the input
    pub repr: CrtRepr,
}

impl CircInput {
    /// Create new input instance
    pub fn new(name: String, repr: CrtRepr) -> Self {
        Self { name, repr }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crt_repr() {
        let crt = CrtValue::<5>::new_from_id(2);

        let nodes: [ArithNode<Feed>; 5] = [
            ArithNode::<Feed>::new(2, 2),
            ArithNode::<Feed>::new(3, 3),
            ArithNode::<Feed>::new(4, 5),
            ArithNode::<Feed>::new(5, 7),
            ArithNode::<Feed>::new(6, 11),
        ];
        let expected = CrtValue(nodes);

        assert_eq!(crt, expected);
    }
}
