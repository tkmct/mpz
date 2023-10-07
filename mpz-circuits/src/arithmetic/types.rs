use std::marker::PhantomData;

use crate::{
    arithmetic::utils::{NPRIMES, PRIMES},
    {Feed, Sink},
};

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

/// CRT representation of a field element in circuit.
/// This bundles crt wires. each wire has modulus and unique id in a circuit.
#[derive(Debug, Eq, PartialEq)]
pub struct CrtRepr<const N: usize>([ArithNode<Feed>; N]);

impl<const N: usize> CrtRepr<N> {
    pub(crate) fn new(nodes: [ArithNode<Feed>; N]) -> CrtRepr<N> {
        // check if N is less than NPRIMES
        // There is unstable feature to do this compile time using generic_const_expr.
        assert!(N <= NPRIMES, "Const N should be less than NPRIMES");
        CrtRepr(nodes)
    }

    /// generate CrtRepr which takes N ids starting from given id.
    /// eg. CrtRepr<5>::new_from_id(4) will generate CrtRepr with ArithNodes
    /// having the following id and moduli pairs
    /// [(2,4), (3,5), (5,6), (7,7), (11,8)]
    pub(crate) fn new_from_id(id: usize) -> CrtRepr<N> {
        let mut nodes = [ArithNode::<Feed>::new(0, 0); N];
        for (i, p) in (0..N).zip(PRIMES) {
            nodes[i] = ArithNode::<Feed>::new(id + i, p);
        }

        CrtRepr(nodes)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crt_repr() {
        let crt = CrtRepr::<5>::new_from_id(2);

        let nodes: [ArithNode<Feed>; 5] = [
            ArithNode::<Feed>::new(2, 2),
            ArithNode::<Feed>::new(3, 3),
            ArithNode::<Feed>::new(4, 5),
            ArithNode::<Feed>::new(5, 7),
            ArithNode::<Feed>::new(6, 11),
        ];
        let expected = CrtRepr(nodes);

        assert_eq!(crt, expected);
    }
}
