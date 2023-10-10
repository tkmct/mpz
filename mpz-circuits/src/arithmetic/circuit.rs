//! Arithmetic Circuit module.
use crate::Feed;

use super::{
    components::ArithGate,
    types::{ArithNode, Fp},
};

/// An error that occur when evaluating a circuit.
#[derive(Debug, thiserror::Error, PartialEq)]
#[allow(missing_docs)]
pub enum ArithCircuitError {
    #[error("input feed expected value modulo {0} but got {1}")]
    InvalidInputMod(u16, u32),
    #[error("Invalid number of inputs: expected {0}, got {1}")]
    InvalidInputCount(usize, usize),
    #[error("Invalid number of outputs: expected {0}, got {1}")]
    InvalidOutputCount(usize, usize),
}

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
    pub fn evaluate(&self, values: &[Fp]) -> Result<Vec<Fp>, ArithCircuitError> {
        if values.len() != self.inputs.len() {
            return Err(ArithCircuitError::InvalidInputCount(
                self.inputs.len(),
                values.len(),
            ));
        }

        let mut feeds: Vec<Option<Fp>> = vec![None; self.feed_count()];

        for (input, value) in self.inputs.iter().zip(values) {
            // check if input values are within mod of Node
            if input.modulus() as u32 <= value.0 {
                return Err(ArithCircuitError::InvalidInputMod(input.modulus(), value.0));
            }
            feeds[input.id()] = Some(*value);
        }

        for gate in self.gates.iter() {
            // TODO: not cast directly. how to handle correctly?
            let m = gate.x().modulus() as u32;
            match gate {
                ArithGate::Add { x, y, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");
                    let y = feeds[y.id()].expect("Feed should be set");

                    feeds[z.id()] = Some(Fp((x.0 + y.0) % m));
                }
                ArithGate::Cmul { x, c, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");

                    feeds[z.id()] = Some(Fp(x.0 * c.0 % m));
                }
                ArithGate::Mul { x, y, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");
                    let y = feeds[y.id()].expect("Feed should be set");

                    feeds[z.id()] = Some(Fp(x.0 * y.0 % m));
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
    use crate::arithmetic::{
        builder::ArithmeticCircuitBuilder,
        ops::{add, cmul, mul},
    };

    #[test]
    fn test_evaluate() {
        // calc: a*b + 3*a
        let mut builder = ArithmeticCircuitBuilder::new();

        let a = builder.add_crt_input::<3>();
        let b = builder.add_crt_input::<3>();
        let out;
        // FIXME: how to do it more elegantly?
        {
            let mut state = builder.state().borrow_mut();
            let c = mul(&mut state, &a, &b);
            let d = cmul(&mut state, &a, Fp(3));
            out = add(&mut state, &c, &d);
        }

        builder.add_crt_output(&out);
        let circ = builder.build().unwrap();

        // values have to be CRT represented value.
        // 3, 5
        let values = vec![Fp(1), Fp(0), Fp(3), Fp(1), Fp(2), Fp(0)];
        let res = circ.evaluate(&values).unwrap();
        // Returns 24 in Crt repr
        assert_eq!(res, vec![Fp(0), Fp(0), Fp(4)]);
    }

    #[test]
    fn test_evaluate_proj() {
        let mut builder = ArithmeticCircuitBuilder::new();

        let x = builder.add_input(2);
        let tt: Vec<Fp> = vec![Fp(1), Fp(2)];
        let out = builder.add_proj_gate(&x, tt);
        builder.add_output(&out);

        let circ = builder.build().unwrap();

        let values = vec![Fp(0)];
        let res = circ.evaluate(&values).unwrap();
        assert_eq!(res, vec![Fp(1)]);
    }

    // check if value is greater than mod specified by value
    #[test]
    fn test_input_value_exceed_mod() {
        let mut builder = ArithmeticCircuitBuilder::new();
        let x = builder.add_input(2);
        let y = builder.add_input(2);

        let out = builder.add_add_gate(&x, &y).unwrap();
        builder.add_output(&out);

        let circ = builder.build().unwrap();
        let values = vec![Fp(0), Fp(2)];
        let res = circ.evaluate(&values);
        assert_eq!(res, Err(ArithCircuitError::InvalidInputMod(2, 2)));
    }
}
