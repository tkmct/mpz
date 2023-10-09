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

    /// Evaluate a plaintext arithmetic circuit with given plaintext input values.
    pub fn evaluate(&self, values: &[Fp]) -> Result<Vec<Fp>, CircuitError> {
        if values.len() != self.inputs.len() {
            return Err(CircuitError::InvalidInputCount(
                self.inputs.len(),
                values.len(),
            ));
        }

        let mut feeds: Vec<Option<Fp>> = vec![None; self.feed_count()];

        for (input, value) in self.inputs.iter().zip(values) {
            feeds[input.id()] = Some(*value);
        }

        for gate in self.gates.iter() {
            match gate {
                ArithGate::Add { x, y, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");
                    let y = feeds[y.id()].expect("Feed should be set");

                    feeds[z.id()] = Some(Fp(x.0 + y.0));
                }
                ArithGate::Cmul { x, c, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");

                    feeds[z.id()] = Some(Fp(x.0 * c.0));
                }
                ArithGate::Mul { x, y, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");
                    let y = feeds[y.id()].expect("Feed should be set");

                    // TODO: take mod p
                    feeds[z.id()] = Some(Fp(x.0 * y.0));
                }
                ArithGate::Proj { x, tt, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");
                    feeds[z.id()] = Some(tt[x.0 as usize]);
                }
            }
        }

        // collect output
        let outputs = self
            .outputs
            .iter()
            .cloned()
            .map(|out| feeds[out.id()].expect("Feed should be set"))
            .collect();

        Ok(outputs)
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
    // TODO: should test on circuit having CrtRepr

    // #[test]
    // fn test_evaluate() {
    //     // calc: a*b + 3*a
    //     let mut builder = ArithmeticCircuitBuilder::new();
    //
    //     let a = builder.add_input();
    //     let b = builder.add_input();
    //     let c = builder.add_mul_gate(&a, &b);
    //     let d = builder.add_cmul_gate(&a, Fp(3));
    //     let out = builder.add_add_gate(&c, &d);
    //
    //     builder.add_output(&out);
    //
    //     let circ = builder.build().unwrap();
    //
    //     let values = vec![Fp(3), Fp(5)];
    //     let res = circ.evaluate(&values).unwrap();
    //     assert_eq!(res, vec![Fp(24)]);
    // }

    // #[test]
    // fn test_evaluate_proj() {
    //     let mut builder = ArithmeticCircuitBuilder::new();
    //
    //     let x = builder.add_input();
    //     let tt: Vec<Fp> = vec![Fp(1), Fp(2), Fp(3)];
    //
    //     let out = builder.add_proj_gate(&x, tt);
    //     builder.add_output(&out);
    //
    //     let circ = builder.build().unwrap();
    //
    //     let values = vec![Fp(2)];
    //     let res = circ.evaluate(&values).unwrap();
    //     assert_eq!(res, vec![Fp(3)]);
    // }
}
