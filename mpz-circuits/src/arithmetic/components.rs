//! Definition of components used to construct arithmetic circuit.
use super::types::ArithNode;
use crate::components::{Feed, Sink};

/// An arithmetic gate.
/// Feed and Sink holds id of set of wires = Node.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
pub enum ArithGate {
    Add {
        // List of wires
        x: ArithNode<Sink>,
        y: ArithNode<Sink>,
        z: ArithNode<Feed>,
    },
    Mul {
        x: ArithNode<Sink>,
        y: ArithNode<Sink>,
        z: ArithNode<Feed>,
    },
    Cmul {
        x: ArithNode<Sink>,
        // constant value
        y: ArithNode<Sink>,
        z: ArithNode<Feed>,
    },
    // TODO: add PROJ gate
}

impl ArithGate {
    /// Returns the type of the gate.
    pub fn gate_type(&self) -> ArithGateType {
        match self {
            ArithGate::Add { .. } => ArithGateType::Add,
            ArithGate::Mul { .. } => ArithGateType::Mul,
            ArithGate::Cmul { .. } => ArithGateType::Cmul,
        }
    }

    /// Returns the x input of the gate.
    pub fn x(&self) -> &ArithNode<Sink> {
        match self {
            ArithGate::Add { x, .. } => x,
            ArithGate::Mul { x, .. } => x,
            ArithGate::Cmul { x, .. } => x,
        }
    }

    /// Returns the y input of the gate.
    /// For cmul operation, y is a constant value.
    pub fn y(&self) -> &ArithNode<Sink> {
        match self {
            ArithGate::Add { y, .. } => y,
            ArithGate::Mul { y, .. } => y,
            ArithGate::Cmul { y, .. } => y,
        }
    }

    /// Returns the z output of the gate.
    pub fn z(&self) -> &ArithNode<Feed> {
        match self {
            ArithGate::Add { z, .. } => z,
            ArithGate::Mul { z, .. } => z,
            ArithGate::Cmul { z, .. } => z,
        }
    }
}

/// Type of an arithmetic gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArithGateType {
    /// Addition gate.
    Add,
    /// Multiplication gate.
    Mul,
    /// Constant multiplication gate.
    Cmul,
}
