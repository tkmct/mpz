//! Arithmetic circuit builder module.
use std::cell::RefCell;
use std::collections::HashSet;

use crate::{
    arithmetic::{
        circuit::ArithmeticCircuit,
        components::ArithGate,
        types::{ArithNode, CircInput, CrtLen, CrtRepr, ToCrtRepr, TypeError},
        utils::PRIMES,
    },
    BuilderError, Feed,
};

use super::circuit::ArithCircuitError;

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
    pub fn add_input<T: ToCrtRepr + CrtLen>(
        &self,
        name: String,
    ) -> Result<CircInput, ArithCircuitError> {
        let mut state = self.state.borrow_mut();
        let repr = state.add_value::<T>()?;
        let input = CircInput::new(name, repr);
        state.inputs.push(input.clone());

        Ok(input)
    }

    /// Add an output wire to circuit.
    pub fn add_output(&self, value: &CrtRepr) {
        let mut state = self.state.borrow_mut();
        state.outputs.push(value.clone());
    }

    /// Add proj gate
    // TODO: maybe we don't need builder state. just put everything under builder
    // TODO: should return Result?
    pub fn add_proj_gate(
        &mut self,
        x: &ArithNode<Feed>,
        q: u16,
        tt: Vec<u16>,
    ) -> Result<ArithNode<Feed>, ArithCircuitError> {
        let mut state = self.state.borrow_mut();
        state.add_proj_gate(x, q, tt)
    }

    /// Add add gate wrapper
    pub fn add_add_gate(
        &mut self,
        x: &ArithNode<Feed>,
        y: &ArithNode<Feed>,
    ) -> Result<ArithNode<Feed>, ArithCircuitError> {
        let mut state = self.state.borrow_mut();
        state.add_add_gate(x, y)
    }
}

/// Arithmetic circuit builder's internal state.
#[derive(Default)]
pub struct ArithBuilderState {
    pub(crate) feed_id: usize,
    inputs: Vec<CircInput>,
    outputs: Vec<CrtRepr>,
    gates: Vec<ArithGate>,
    // constants: Vec<u32>,
    add_count: usize,
    mul_count: usize,
    cmul_count: usize,
    proj_count: usize,

    // Moduli numbers used in the circuit
    moduli: HashSet<u16>,
}

impl ArithBuilderState {
    pub(crate) fn add_feed(&mut self, modulus: u16) -> ArithNode<Feed> {
        let node = ArithNode::<Feed>::new(self.feed_id, modulus);
        self.feed_id += 1;
        self.moduli.insert(modulus);
        node
    }

    pub(crate) fn add_value<T: ToCrtRepr + CrtLen>(&mut self) -> Result<CrtRepr, TypeError> {
        let nodes: Vec<ArithNode<Feed>> = (0..T::LEN).map(|i| self.add_feed(PRIMES[i])).collect();
        T::new_crt_repr(&nodes)
    }

    /// Add ADD gate to a circuit.
    pub(crate) fn add_add_gate(
        &mut self,
        x: &ArithNode<Feed>,
        y: &ArithNode<Feed>,
    ) -> Result<ArithNode<Feed>, ArithCircuitError> {
        // check lhs and rhs has same modulus
        if x.modulus() != y.modulus() {
            return Err(ArithCircuitError::UnequalModuli(x.modulus(), y.modulus()));
        }

        let out = self.add_feed(x.modulus());

        let gate = ArithGate::Add {
            x: x.into(),
            y: y.into(),
            z: out,
        };
        self.add_count += 1;
        self.gates.push(gate);

        Ok(out)
    }

    /// Add ADD gate to a circuit.
    pub(crate) fn add_sub_gate(
        &mut self,
        x: &ArithNode<Feed>,
        y: &ArithNode<Feed>,
    ) -> Result<ArithNode<Feed>, ArithCircuitError> {
        // check lhs and rhs has same modulus
        if x.modulus() != y.modulus() {
            return Err(ArithCircuitError::UnequalModuli(x.modulus(), y.modulus()));
        }

        let out = self.add_feed(x.modulus());

        let gate = ArithGate::Sub {
            x: x.into(),
            y: y.into(),
            z: out,
        };
        self.add_count += 1;
        self.gates.push(gate);

        Ok(out)
    }

    /// Add CMUL gate to a circuit
    pub(crate) fn add_cmul_gate(&mut self, x: &ArithNode<Feed>, c: u64) -> ArithNode<Feed> {
        let out = self.add_feed(x.modulus());
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
    ) -> Result<ArithNode<Feed>, ArithCircuitError> {
        // check lhs and rhs has same modulus
        if x.modulus() != y.modulus() {
            return Err(ArithCircuitError::UnequalModuli(x.modulus(), y.modulus()));
        }

        let out = self.add_feed(x.modulus());
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
    pub(crate) fn add_proj_gate(
        &mut self,
        x: &ArithNode<Feed>,
        q: u16,
        tt: Vec<u16>,
    ) -> Result<ArithNode<Feed>, ArithCircuitError> {
        // check if the number of tt rows are equal to x's modulus
        let out = self.add_feed(q);

        let gate = ArithGate::Proj {
            x: x.into(),
            tt,
            z: out,
        };
        self.proj_count += 1;
        self.gates.push(gate);

        Ok(out)
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
            moduli: self.moduli.into_iter().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arithmetic::types::CrtValue, Feed};

    #[test]
    fn test_add_input() {
        // test just add_input method
        let builder = ArithmeticCircuitBuilder::new();

        let a = builder.add_input::<u32>("a".into());
        let expected: [ArithNode<Feed>; 10] =
            std::array::from_fn(|i| ArithNode::<Feed>::new(i, PRIMES[i]));
        assert!(a.is_ok());
        assert_eq!(
            a.unwrap(),
            CircInput::new("a".into(), CrtRepr::U32(CrtValue::new(expected)))
        );

        let circ = builder.build().unwrap();
        assert_eq!(circ.inputs().len(), 1);
    }
}
