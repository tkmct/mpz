//! Arithmetic circuit builder module.
use std::cell::RefCell;

use crate::{BuilderError, Feed};

use super::{
    circuit::ArithmeticCircuit,
    components::ArithGate,
    types::{ArithNode, CrtRepr, Fp},
    utils::PRIMES,
};

/// An error that can occur while building arithmetic circuit.
#[derive(Debug, thiserror::Error)]
pub enum ArithBuilderError {
    /// Moduli are expected to be same but got unequal moduli
    #[error("Moduli does not match: got {0} and {1}")]
    UnequalModuli(u16, u16),
}

/// Result wrapper to wrap ArithBuilderError
pub type BuilderResult<T> = Result<T, ArithBuilderError>;

/// Arithmetic circuit builder.
// FIXME: merge ArithmeticCircuitBuilder and its state.
#[derive(Default)]
pub struct ArithmeticCircuitBuilder {
    state: RefCell<ArithBuilderState>,
}

impl ArithmeticCircuitBuilder {
    /// Creates a new circuit builder.
    pub fn new() -> Self {
        ArithmeticCircuitBuilder {
            state: RefCell::new(ArithBuilderState::default()),
        }
    }

    /// Returns a reference to the internal state of the builder
    pub fn state(&self) -> &RefCell<ArithBuilderState> {
        &self.state
    }

    /// Build a circuit from state.
    pub fn build(self) -> Result<ArithmeticCircuit, BuilderError> {
        self.state.into_inner().build()
    }

    /// Add an input wire to circuit.
    pub fn add_input(&self, modulus: u16) -> ArithNode<Feed> {
        let mut state = self.state.borrow_mut();
        let value = state.add_value(modulus);
        state.inputs.push(value);

        value
    }

    /// Add an output wire to circuit.
    pub fn add_output(&self, value: &ArithNode<Feed>) {
        let mut state = self.state.borrow_mut();
        state.outputs.push(*value);
    }

    /// Add CRT repr input to a circuit
    pub fn add_crt_input<const N: usize>(&mut self) -> CrtRepr<N> {
        let mut state = self.state.borrow_mut();
        let nodes: [ArithNode<Feed>; N] = std::array::from_fn(|i| {
            let node = state.add_value(PRIMES[i]);
            state.inputs.push(node);
            node
        });

        CrtRepr::<N>::new(nodes)
    }

    /// Add CRT repr output to a circuit
    pub fn add_crt_output<const N: usize>(&mut self, output: &CrtRepr<N>) {
        let mut state = self.state.borrow_mut();
        output.nodes().iter().for_each(|n| state.outputs.push(*n));
    }

    /// Add proj gate
    // TODO: maybe we don't need builder state. just put everything under builder
    // TODO: should return Result?
    pub fn add_proj_gate(&mut self, x: &ArithNode<Feed>, tt: Vec<Fp>) -> ArithNode<Feed> {
        let mut state = self.state.borrow_mut();
        state.add_proj_gate(x, tt)
    }

    /// Add add gate wrapper
    pub fn add_add_gate(
        &mut self,
        x: &ArithNode<Feed>,
        y: &ArithNode<Feed>,
    ) -> BuilderResult<ArithNode<Feed>> {
        let mut state = self.state.borrow_mut();
        state.add_add_gate(x, y)
    }
}

/// Arithmetic circuit builder's internal state.
#[derive(Default)]
pub struct ArithBuilderState {
    pub(crate) feed_id: usize,
    inputs: Vec<ArithNode<Feed>>,
    outputs: Vec<ArithNode<Feed>>,
    gates: Vec<ArithGate>,

    add_count: usize,
    mul_count: usize,
    cmul_count: usize,
    proj_count: usize,
}

impl ArithBuilderState {
    pub(crate) fn add_value(&mut self, modulus: u16) -> ArithNode<Feed> {
        let node = ArithNode::<Feed>::new(self.feed_id, modulus);
        self.feed_id += 1;

        node
    }

    /// Add ADD gate to a circuit.
    pub(crate) fn add_add_gate(
        &mut self,
        x: &ArithNode<Feed>,
        y: &ArithNode<Feed>,
    ) -> BuilderResult<ArithNode<Feed>> {
        // check lhs and rhs has same modulus
        if x.modulus() != y.modulus() {
            return Err(ArithBuilderError::UnequalModuli(x.modulus(), y.modulus()));
        }

        let out = self.add_value(x.modulus());

        let gate = ArithGate::Add {
            x: x.into(),
            y: y.into(),
            z: out,
        };
        self.add_count += 1;
        self.gates.push(gate);

        Ok(out)
    }

    /// Add CMUL gate to a circuit
    pub(crate) fn add_cmul_gate(&mut self, x: &ArithNode<Feed>, c: Fp) -> ArithNode<Feed> {
        let out = self.add_value(x.modulus());
        let gate = ArithGate::Cmul {
            x: x.into(),
            c,
            z: out,
        };
        self.cmul_count += 1;
        self.gates.push(gate);
        out
    }

    /// Add MUL gate to a circuit
    pub(crate) fn add_mul_gate(
        &mut self,
        x: &ArithNode<Feed>,
        y: &ArithNode<Feed>,
    ) -> BuilderResult<ArithNode<Feed>> {
        // check lhs and rhs has same modulus
        if x.modulus() != y.modulus() {
            return Err(ArithBuilderError::UnequalModuli(x.modulus(), y.modulus()));
        }

        let out = self.add_value(x.modulus());
        let gate = ArithGate::Mul {
            x: x.into(),
            y: y.into(),
            z: out,
        };
        self.mul_count += 1;
        self.gates.push(gate);

        Ok(out)
    }

    /// Add PROJ gate to a circuit
    #[allow(dead_code)]
    pub(crate) fn add_proj_gate(&mut self, x: &ArithNode<Feed>, tt: Vec<Fp>) -> ArithNode<Feed> {
        // check if the number of tt rows are equal to x's modulus

        let out = self.add_value(x.modulus());
        let gate = ArithGate::Proj {
            x: x.into(),
            tt,
            z: out,
        };
        self.proj_count += 1;
        self.gates.push(gate);
        out
    }

    /// Builds a circuit.
    pub(crate) fn build(self) -> Result<ArithmeticCircuit, BuilderError> {
        Ok(ArithmeticCircuit {
            inputs: self.inputs,
            outputs: self.outputs,
            gates: self.gates,
            feed_count: self.feed_id,
            add_count: self.add_count,
            cmul_count: self.cmul_count,
            mul_count: self.mul_count,
            proj_count: self.proj_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        arithmetic::{circuit::ArithmeticCircuit, components::ArithGate},
        Feed, Sink,
    };

    // test if given circuit are same.
    fn check_circuit_equality(circ: &ArithmeticCircuit, other: &ArithmeticCircuit) -> bool {
        // check each count.
        assert_eq!(circ.feed_count(), other.feed_count());
        assert_eq!(circ.add_count(), other.add_count());
        assert_eq!(circ.cmul_count(), other.cmul_count());
        assert_eq!(circ.mul_count(), other.mul_count());

        // check each vecs.
        assert_eq!(circ.inputs, other.inputs);
        assert_eq!(circ.outputs, other.outputs);
        assert_eq!(circ.gates, other.gates);
        true
    }

    #[test]
    fn test_build_circuit() {
        // test simple circuit by hand.
        // calc: a*b + 3*a
        let builder = ArithmeticCircuitBuilder::new();

        // add input x, y
        let a = builder.add_input(3);
        let b = builder.add_input(3);
        let out = {
            let mut state = builder.state().borrow_mut();
            let c = state.add_mul_gate(&a, &b).unwrap();
            let d = state.add_cmul_gate(&a, Fp(3));

            state.add_add_gate(&c, &d).unwrap()
        };

        builder.add_output(&out);

        let result = builder.build();
        assert!(result.is_ok());
        let circ = result.unwrap();

        // modulus for this test.
        let m = 3;

        // construct expected circuit by hand.
        let gate1 = ArithGate::Mul {
            x: ArithNode::<Sink>::new(0, m),
            y: ArithNode::<Sink>::new(1, m),
            z: ArithNode::<Feed>::new(2, m),
        };

        let gate2 = ArithGate::Cmul {
            x: ArithNode::<Sink>::new(0, m),
            c: Fp(3),
            z: ArithNode::<Feed>::new(3, m),
        };

        let gate3 = ArithGate::Add {
            x: ArithNode::<Sink>::new(2, m),
            y: ArithNode::<Sink>::new(3, m),
            z: ArithNode::<Feed>::new(4, m),
        };

        let expected = ArithmeticCircuit {
            inputs: vec![a, b],
            outputs: vec![out],
            gates: vec![gate1, gate2, gate3],

            feed_count: 5,
            add_count: 1,
            cmul_count: 1,
            mul_count: 1,
            proj_count: 0,
        };

        assert!(check_circuit_equality(&circ, &expected));
    }
}
