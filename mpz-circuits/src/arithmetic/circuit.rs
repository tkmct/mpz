//! Arithmetic Circuit module.
use crate::arithmetic::{
    components::ArithGate,
    types::{CrtRepr, TypeError},
    utils::{convert_crts_to_values, convert_values_to_crts},
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
    #[error("Moduli does not match: got {0} and {1}")]
    UnequalModuli(u16, u16),

    #[error(transparent)]
    TypeError(#[from] TypeError),
}

/// Arithmetic Circuit.
/// Which handles N wire crt representation.
#[derive(Debug, Clone)]
pub struct ArithmeticCircuit {
    pub(crate) inputs: Vec<CrtRepr>,
    pub(crate) outputs: Vec<CrtRepr>,
    pub(crate) gates: Vec<ArithGate>,
    pub(crate) feed_count: usize,

    pub(crate) add_count: usize,
    pub(crate) cmul_count: usize,
    pub(crate) mul_count: usize,
    pub(crate) proj_count: usize,
}

impl ArithmeticCircuit {
    /// Returns a reference to the inputs of the circuit.
    pub fn inputs(&self) -> &[CrtRepr] {
        &self.inputs
    }

    /// Returns a reference to the outputs of the circuit.
    pub fn outputs(&self) -> &[CrtRepr] {
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
    /// TODO: use generic type like ToCrt in inputs values
    pub fn evaluate(&self, values: &[u32]) -> Result<Vec<u32>, ArithCircuitError> {
        if values.len() != self.inputs.len() {
            return Err(ArithCircuitError::InvalidInputCount(
                self.inputs.len(),
                values.len(),
            ));
        }

        let mut feeds: Vec<Option<u16>> = vec![None; self.feed_count()];

        // Convert value to crt representation and add inputs in feeds.
        let actual_values = convert_values_to_crts(self.inputs(), values)?;
        for (repr, val) in self.inputs().iter().zip(actual_values.iter()) {
            for (i, feed) in repr.iter().enumerate() {
                feeds[feed.id()] = Some(val[i]);
            }
        }

        for gate in self.gates.iter() {
            let m = gate.x().modulus();
            match gate {
                ArithGate::Add { x, y, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");
                    let y = feeds[y.id()].expect("Feed should be set");

                    feeds[z.id()] = Some((x + y) % m);
                }
                ArithGate::Cmul { x, c, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");

                    feeds[z.id()] = Some(((x as u32 * c) % m as u32) as u16);
                }
                ArithGate::Mul { x, y, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");
                    let y = feeds[y.id()].expect("Feed should be set");

                    feeds[z.id()] = Some(x * y % m);
                }
                ArithGate::Proj { x, tt, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");
                    feeds[z.id()] = Some(tt[x as usize]);
                }
            }
        }

        let outputs = convert_crts_to_values(self.outputs(), &feeds)?;
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
    use crate::arithmetic::{builder::ArithmeticCircuitBuilder, ops::*};

    #[test]
    fn test_evaluate() {
        // calc: a*b + 3*a
        let builder = ArithmeticCircuitBuilder::new();

        let a = builder.add_input::<u32>().unwrap();
        let b = builder.add_input::<u32>().unwrap();
        let out;
        // FIXME: how to do it more elegantly?
        {
            let mut state = builder.state().borrow_mut();
            let c = mul(&mut state, &a, &b).unwrap();
            let d = cmul(&mut state, &a, 3);
            out = add(&mut state, &c, &d).unwrap();
        }

        builder.add_output(&out);
        let circ = builder.build().unwrap();
        let values = vec![3, 5];
        let res = circ.evaluate(&values).unwrap();
        assert_eq!(res, vec![24]);
    }

    // #[test]
    // fn test_evaluate_proj() {
    //     let mut builder = ArithmeticCircuitBuilder::new();
    //
    //     let x = builder.add_input::<u32>().unwrap();
    //     let tt: Vec<Fp> = vec![Fp(1), Fp(2)];
    //     let out = builder.add_proj_gate(&x, tt);
    //     builder.add_output(&out);
    //
    //     let circ = builder.build().unwrap();
    //
    //     let values = vec![Fp(0)];
    //     let res = circ.evaluate(&values).unwrap();
    //     assert_eq!(res, vec![Fp(1)]);
    // }
}
