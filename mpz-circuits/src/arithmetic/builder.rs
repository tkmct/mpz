//! Arithmetic circuit builder module.
use super::{circuit::ArithmeticCircuit, components::ArithGate, types::ArithNode};
use crate::{BuilderError, Feed};
use std::cell::RefCell;

/// Arithmetic circuit builder.
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
    pub fn add_input(&self) -> ArithNode<Feed> {
        let mut state = self.state.borrow_mut();
        let value = state.add_value();
        state.inputs.push(value);

        value
    }

    /// Add an output wire to circuit.
    pub fn add_output(&self, value: &ArithNode<Feed>) {
        let mut state = self.state.borrow_mut();
        state.outputs.push(*value);
    }

    /// Add ADD gate to a circuit.
    pub fn add_add_gate(
        &mut self,
        lhs: &ArithNode<Feed>,
        rhs: &ArithNode<Feed>,
    ) -> ArithNode<Feed> {
        let mut state = self.state.borrow_mut();
        state.add_add_gate(lhs, rhs)
    }

    /// Add CMUL gate to a circuit.
    pub fn add_cmul_gate(&mut self, x: &ArithNode<Feed>, c: &ArithNode<Feed>) -> ArithNode<Feed> {
        let mut state = self.state.borrow_mut();
        state.add_cmul_gate(x, c)
    }

    /// Add MUL gate to a circuit.
    pub fn add_mul_gate(
        &mut self,
        lhs: &ArithNode<Feed>,
        rhs: &ArithNode<Feed>,
    ) -> ArithNode<Feed> {
        let mut state = self.state.borrow_mut();
        state.add_mul_gate(lhs, rhs)
    }
}

/// Arithmetic circuit builder's internal state.
#[derive(Default)]
pub struct ArithBuilderState {
    feed_id: usize,
    inputs: Vec<ArithNode<Feed>>,
    outputs: Vec<ArithNode<Feed>>,
    gates: Vec<ArithGate>,

    add_count: usize,
    mul_count: usize,
    cmul_count: usize,
    proj_count: usize,
}

impl ArithBuilderState {
    // TODO: make generic over arith node moduli
    fn add_value(&mut self) -> ArithNode<Feed> {
        let node = ArithNode::<Feed>::new_u32(self.feed_id);
        self.feed_id += 1;

        node
    }

    /// Add ADD gate to a circuit.
    pub fn add_add_gate(
        &mut self,
        lhs: &ArithNode<Feed>,
        rhs: &ArithNode<Feed>,
    ) -> ArithNode<Feed> {
        let out = self.add_value();
        let gate = ArithGate::Add {
            x: lhs.into(),
            y: rhs.into(),
            z: out,
        };
        self.add_count += 1;
        self.gates.push(gate);

        out
    }

    /// Add CMUL gate to a circuit
    pub fn add_cmul_gate(
        &mut self,
        lhs: &ArithNode<Feed>,
        rhs: &ArithNode<Feed>,
    ) -> ArithNode<Feed> {
        todo!()
    }

    /// Add MUL gate to a circuit
    pub fn add_mul_gate(
        &mut self,
        lhs: &ArithNode<Feed>,
        rhs: &ArithNode<Feed>,
    ) -> ArithNode<Feed> {
        let out = self.add_value();
        let gate = ArithGate::Mul {
            x: lhs.into(),
            y: rhs.into(),
            z: out,
        };
        self.mul_count += 1;
        self.gates.push(gate);

        out
    }

    /// Add PROJ gate to a circuit
    pub fn add_proj_gate() {
        todo!()
    }

    /// Builds a circuit.
    pub fn build(self) -> Result<ArithmeticCircuit, BuilderError> {
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
    use crate::arithmetic::{circuit::ArithmeticCircuit, components::ArithGate};
    use crate::{Feed, Sink};

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
    fn test_circuit_build() {
        // test simple circuit by hand.
        // calc: x*y + x
        let mut builder = ArithmeticCircuitBuilder::new();

        // add input x, y
        let x = builder.add_input();
        let y = builder.add_input();
        let z = builder.add_mul_gate(&x, &y);
        let out = builder.add_add_gate(&x, &z);

        builder.add_output(&out);
        // add output

        let result = builder.build();
        assert!(result.is_ok());
        let circ = result.unwrap();

        // construct expected circuit by hand.
        //
        let gate1 = ArithGate::Mul {
            x: ArithNode::<Sink>::new_u32(0),
            y: ArithNode::<Sink>::new_u32(1),
            z: ArithNode::<Feed>::new_u32(2),
        };
        let gate2 = ArithGate::Add {
            x: ArithNode::<Sink>::new_u32(0),
            y: ArithNode::<Sink>::new_u32(2),
            z: ArithNode::<Feed>::new_u32(3),
        };

        let expected = ArithmeticCircuit {
            inputs: vec![x, y],
            outputs: vec![out],
            gates: vec![gate1, gate2],

            feed_count: 4,
            add_count: 1,
            cmul_count: 0,
            mul_count: 1,
            proj_count: 0,
        };

        assert!(check_circuit_equality(&circ, &expected));
    }

    // TODO: test circuit with proj gate.
}
