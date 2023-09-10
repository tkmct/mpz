//! Arithmetic Circuit module.
use crate::{CircuitError, Feed};

use super::{
    components::ArithGate,
    types::{ArithNode, Fp},
};

/// Arithmetic Circuit.
#[derive(Debug, Clone)]
pub struct ArithmeticCircuit {
    pub(crate) inputs: Vec<ArithNode<Feed>>,
    pub(crate) outputs: Vec<ArithNode<Feed>>,
    pub(crate) gates: Vec<ArithGate>,
    pub(crate) feed_count: usize,

    pub(crate) add_count: usize,
    pub(crate) cmul_count: usize,
    pub(crate) mul_count: usize,
    pub(crate) proj_count: usize,
}

impl ArithmeticCircuit {
    /// Returns a reference to the inputs of the circuit.
    pub fn inputs(&self) -> &[ArithNode<Feed>] {
        &self.inputs
    }

    /// Returns a reference to the outputs of the circuit.
    pub fn outputs(&self) -> &[ArithNode<Feed>] {
        &self.outputs
    }

    /// Returns a reference to the gates of the circuit.
    pub fn gates(&self) -> &[ArithGate] {
        &self.gates
    }

    /// Returns the number of feeds in the circuit.
    pub fn feed_count(&self) -> usize {
        self.feed_count
    }

    /// Returns the number of ADD gates in the circuit.
    pub fn add_count(&self) -> usize {
        self.add_count
    }

    /// Returns the number of MUL gates in the circuit.
    pub fn mul_count(&self) -> usize {
        self.mul_count
    }

    /// Returns the number of CMUL gates in the circuit.
    pub fn cmul_count(&self) -> usize {
        self.cmul_count
    }

    /// Returns the number of PROJ gates in the circuit.
    pub fn proj_count(&self) -> usize {
        self.proj_count
    }

    /// Evaluate an arithmetic circuit with given input values.
    pub fn evaluate(&self, values: &[Fp]) -> Result<Vec<Fp>, CircuitError> {
        // zip values and self.inputs and make feed list.
        // evaluate each gate by iterating using the feed.
        Ok(vec![Fp(1)])
    }
}

impl IntoIterator for ArithmeticCircuit {
    type Item = ArithGate;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.gates.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arithmetic::builder::ArithmeticCircuitBuilder;

    #[test]
    fn test_evaluate() {
        // simple circuit
        // calc: x*y + 5*x
        // TODO: use circuit builder
        let builder = ArithmeticCircuitBuilder::new();
        let circ = builder.build().unwrap();

        let values = vec![];
        let res = circ.evaluate(&values).unwrap();
        assert_eq!(res, vec![Fp(12)]);
    }
}
