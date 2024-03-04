//! Definition of components used to construct arithmetic circuit.
use crate::{
    arithmetic::types::ArithNode,
    components::{Feed, Sink},
};

/// An arithmetic gate.
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
    Sub {
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
        /// constant value should be set as y value
        c: u32,
        z: ArithNode<Feed>,
    },
    Proj {
        x: ArithNode<Sink>,
        // Truth table for input x.
        tt: Vec<u16>,
        z: ArithNode<Feed>,
    },
}

impl ArithGate {
    /// Returns the type of the gate.
    pub fn gate_type(&self) -> ArithGateType {
        match self {
            ArithGate::Add { .. } => ArithGateType::Add,
            ArithGate::Sub { .. } => ArithGateType::Sub,
            ArithGate::Mul { .. } => ArithGateType::Mul,
            ArithGate::Cmul { .. } => ArithGateType::Cmul,
            ArithGate::Proj { .. } => ArithGateType::Proj,
        }
    }

    /// Returns the x input of the gate.
    pub fn x(&self) -> &ArithNode<Sink> {
        match self {
            ArithGate::Add { x, .. } => x,
            ArithGate::Sub { x, .. } => x,
            ArithGate::Mul { x, .. } => x,
            ArithGate::Cmul { x, .. } => x,
            ArithGate::Proj { x, .. } => x,
        }
    }

    /// Returns the y input of the gate.
    /// For cmul operation, y is a constant value.
    pub fn y(&self) -> Option<&ArithNode<Sink>> {
        match self {
            ArithGate::Add { y, .. } => Some(y),
            ArithGate::Sub { y, .. } => Some(y),
            ArithGate::Mul { y, .. } => Some(y),
            ArithGate::Cmul { .. } => None,
            ArithGate::Proj { .. } => None,
        }
    }

    /// Returns the z output of the gate.
    pub fn z(&self) -> &ArithNode<Feed> {
        match self {
            ArithGate::Add { z, .. } => z,
            ArithGate::Sub { z, .. } => z,
            ArithGate::Mul { z, .. } => z,
            ArithGate::Cmul { z, .. } => z,
            ArithGate::Proj { z, .. } => z,
        }
    }
}

/// Type of an arithmetic gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ArithGateType {
    /// Addition gate.
    Add,
    /// Subtraction gate.
    Sub,
    /// Multiplication gate.
    Mul,
    /// Constant multiplication gate.
    Cmul,
    /// Unary projection gate.
    Proj,
}
