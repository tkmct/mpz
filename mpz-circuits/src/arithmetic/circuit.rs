//! Arithmetic Circuit module.

use crate::arithmetic::{
    components::ArithGate,
    types::{CrtRepr, TypeError},
    utils::{convert_crts_to_values, convert_values_to_crts},
};

use super::types::CircInput;

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
    #[error("Invalid number of length: got {0}")]
    InvalidLength(usize),
    #[error("Invalid moduli length: got {0} cannot be handled")]
    InvalidModuliLen(usize),

    #[error(transparent)]
    TypeError(#[from] TypeError),
}

/// Arithmetic Circuit.
/// Which handles N wire crt representation.
#[derive(Debug, Clone)]
pub struct ArithmeticCircuit {
    pub(crate) inputs: Vec<CircInput>,
    pub(crate) outputs: Vec<CrtRepr>,
    pub(crate) gates: Vec<ArithGate>,
    pub(crate) feed_count: usize,
    // pub(crate) constants: Vec<(u32, usize)>,
    pub(crate) add_count: usize,
    pub(crate) cmul_count: usize,
    pub(crate) mul_count: usize,
    pub(crate) proj_count: usize,
    // list of moduli used in the circuit
    pub(crate) moduli: Vec<u16>,
}

impl ArithmeticCircuit {
    ///
    pub fn print_gates(&self) {
        let inputs = self
            .inputs()
            .iter()
            .map(|inp| format!("\tInput{:?} \n", inp))
            .collect::<Vec<_>>()
            .concat();

        let res = self
            .gates
            .iter()
            .map(|gate| match gate {
                ArithGate::Add { x, y, z } => format!("\tADD({},{},{})\n", x.id(), y.id(), z.id()),
                ArithGate::Sub { x, y, z } => format!("\tSUB({},{},{})\n", x.id(), y.id(), z.id()),
                ArithGate::Mul { x, y, z } => format!("\tMUL({},{},{})\n", x.id(), y.id(), z.id()),
                ArithGate::Cmul { x, c, z } => {
                    format!("\tCMUL({},const{},{})\n", x.id(), c, z.id())
                }
                ArithGate::Proj { x, z, .. } => format!("\tPROJ({},TT,{})\n", x.id(), z.id()),
            })
            .collect::<Vec<_>>()
            .concat();
        println!("ArithmeticCircuit \n{}\n{}", inputs, res);
    }

    /// Returns a reference to the inputs of the circuit.
    pub fn inputs(&self) -> &[CircInput] {
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

    /// Returns the slice of constans used in a circuit.
    // pub fn constans(&self) -> &[u32] {
    //     &self.constants
    // }

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

    /// Returns the moduli used in the circuit.
    pub fn moduli(&self) -> &[u16] {
        &self.moduli
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
        let input_reprs = self
            .inputs()
            .iter()
            .map(|i| i.repr.clone())
            .collect::<Vec<_>>();

        // Convert value to crt representation and add inputs in feeds.
        let actual_values = convert_values_to_crts(&input_reprs, values)?;
        for (repr, val) in input_reprs.iter().zip(actual_values.iter()) {
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
                ArithGate::Sub { x, y, z } => {
                    let x = feeds[x.id()].expect("Feed should be set");
                    let y = feeds[y.id()].expect("Feed should be set");

                    feeds[z.id()] = Some((x - y) % m);
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
                    println!("Eval proj gate: {}", tt[x as usize]);
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
    use crate::arithmetic::{builder::ArithmeticCircuitBuilder, ops::*, types::ToCrtRepr};

    #[test]
    fn test_evaluate() {
        // calc: a*b + 3*a
        let builder = ArithmeticCircuitBuilder::new();

        let a = builder.add_input::<u32>("a".into()).unwrap();
        let b = builder.add_input::<u32>("b".into()).unwrap();
        let out;
        {
            let mut state = builder.state().borrow_mut();
            let c = mul(&mut state, &a.repr, &b.repr).unwrap();
            let d = cmul(&mut state, &a.repr, 3);
            out = add(&mut state, &c, &d).unwrap();
        }

        builder.add_output(&out);
        let circ = builder.build().unwrap();
        let values = vec![3, 5];
        let res = circ.evaluate(&values).unwrap();
        assert_eq!(res, vec![24]);
    }

    #[test]
    fn test_evaluate_proj() {
        let builder = ArithmeticCircuitBuilder::new();

        let x = builder.add_input::<bool>("x".into()).unwrap();
        let tt: Vec<u16> = vec![1, 2];
        let node = x.repr.iter().next().unwrap();

        let out_node = builder
            .state()
            .borrow_mut()
            .add_proj_gate(node, node.modulus(), tt)
            .unwrap();
        let out_rep = bool::new_crt_repr(&[out_node]).unwrap();
        builder.add_output(&out_rep);

        let circ = builder.build().unwrap();

        let values = vec![0];
        let res = circ.evaluate(&values).unwrap();
        assert_eq!(res, vec![1]);
    }

    #[test]
    fn test_evaluate_sign() {
        let builder = ArithmeticCircuitBuilder::new();

        let a = builder.add_input::<u32>("a".into()).unwrap();
        let out = {
            let mut state = builder.state().borrow_mut();
            crt_sign::<10>(&mut state, &a.repr, "99.99%").unwrap()
        };

        builder.add_output(&out);

        let circ = builder.build().unwrap();
        let values = vec![3];
        let res = circ.evaluate(&values).unwrap();
        assert_eq!(res, vec![24]);
    }
}
