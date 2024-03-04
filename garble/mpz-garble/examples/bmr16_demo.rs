use futures::SinkExt;
use mpz_circuits::{
    arithmetic::{
        ops::{add, cmul, mul},
        types::{ArithValue, CrtRepr, CrtValueType},
    },
    ArithmeticCircuit, ArithmeticCircuitBuilder, BuilderError,
};
use mpz_garble::{
    bmr16::{
        config::ArithValueIdConfig,
        evaluator::{BMR16Evaluator, BMR16EvaluatorConfig},
        generator::{BMR16Generator, BMR16GeneratorConfig},
    },
    value::{ValueId, ValueRef},
};
use mpz_garble_core::msg::GarbleMessage;
use mpz_ot::mock::mock_ot_shared_pair;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::{collections::HashMap, error, fs};
use utils_aio::duplex::MemoryDuplex;

// Set of structs from circom2mpc compiler
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum AGateType {
    ANone,
    AAdd,
    ASub,
    AMul,
    ADiv,
    AEq,
    ANeq,
    ALEq,
    AGEq,
    ALt,
    AGt,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ArithmeticGate {
    id: u32,
    gate_type: AGateType,
    lh_input: u32,
    rh_input: u32,
    output: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Node {
    id: u32,
    signals: Vec<u32>,
    names: Vec<String>,
    is_const: bool,
    const_value: u32,
}

/// Represents an arithmetic circuit, with a set of variables and gates.
#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct RawCircuit {
    vars: HashMap<u32, Option<u32>>,
    nodes: Vec<Node>,
    gates: Vec<ArithmeticGate>,
}

impl RawCircuit {
    fn get_node_by_id(&self, id: u32) -> Option<Node> {
        for node in &self.nodes {
            if node.id == id {
                return Some(node.clone());
            }
        }
        None
    }
}

#[derive(Debug)]
pub struct CircuitConfig {
    /// list of ids which is from Alice
    pub a_private_inputs: Vec<(String, u32)>,
    pub b_private_inputs: Vec<(String, u32)>,
    pub outputs: Vec<(String, u32)>,
}

impl CircuitConfig {
    pub fn new() -> Self {
        Self {
            a_private_inputs: vec![],
            b_private_inputs: vec![],
            outputs: vec![],
        }
    }

    pub fn gen_a_input_config(&self) -> Vec<ArithValueIdConfig> {
        let mut config = vec![];
        for a_priv in self.a_private_inputs.iter() {
            config.push(ArithValueIdConfig::Private {
                id: ValueId::new(&a_priv.0),
                ty: CrtValueType::U32,
                value: Some(ArithValue::U32(10)),
            })
        }
        for b_priv in self.b_private_inputs.iter() {
            config.push(ArithValueIdConfig::Private {
                id: ValueId::new(&b_priv.0),
                ty: CrtValueType::U32,
                value: None,
            })
        }
        config
    }

    pub fn gen_b_input_config(&self) -> Vec<ArithValueIdConfig> {
        let mut config = vec![];
        for a_priv in self.a_private_inputs.iter() {
            config.push(ArithValueIdConfig::Private {
                id: ValueId::new(&a_priv.0),
                ty: CrtValueType::U32,
                value: None,
            })
        }
        for b_priv in self.b_private_inputs.iter() {
            config.push(ArithValueIdConfig::Private {
                id: ValueId::new(&b_priv.0),
                ty: CrtValueType::U32,
                value: Some(ArithValue::U32(10)),
            })
        }
        config
    }

    pub fn get_input_refs(&self) -> Vec<ValueRef> {
        let mut refs = vec![];
        for a_priv in self.a_private_inputs.iter() {
            refs.push(ValueRef::Value {
                id: ValueId::new(&a_priv.0),
            })
        }
        for b_priv in self.b_private_inputs.iter() {
            refs.push(ValueRef::Value {
                id: ValueId::new(&b_priv.0),
            })
        }
        refs
    }
}

enum Wire {
    Var(CrtRepr),
    Const(u32),
}

fn parse_raw_circuit(
    raw_circ: &RawCircuit,
) -> Result<(ArithmeticCircuit, CircuitConfig), BuilderError> {
    let circ = raw_circ.clone();
    let mut config = CircuitConfig::new();
    let builder = ArithmeticCircuitBuilder::new();
    // take each gate and append in the builder
    // mark input wire
    let mut used_vars = HashMap::<u32, CrtRepr>::new();

    // TODO: load output from config file?
    // for now, use output of last gate as an output of a circuit.

    // controlling inputs/outputs here
    // Define which variables are from which party.
    // loaded from config file in the future
    let input_a_names = vec!["0.input_A", "0.w0", "0.b0", "0.w1", "0.b1"];
    let input_b_names = vec!["0.input_B"];
    let output_names = vec!["0.ip"];

    let mut output = None;
    let mut o = 0;

    for gate in circ.gates.iter() {
        println!("Gate: {:?}", gate);
        let lhs_var = circ.get_node_by_id(gate.lh_input).unwrap();
        let rhs_var = circ.get_node_by_id(gate.rh_input).unwrap();
        let out_var = circ.get_node_by_id(gate.output).unwrap();

        println!("LHS Node: {:?}", lhs_var);
        println!("RHS Node: {:?}", rhs_var);
        println!("OUT Node: {:?}", out_var);

        let mut lhs_name = "";
        let mut rhs_name = "";

        // get name of the lhs_var and append to private input list
        // if the same name presented in the predefined input_a_names/input_b_names
        // TODO: better search the variable names and id.
        for name_v in lhs_var.names.iter() {
            for name_a in input_a_names.as_slice() {
                if name_v.contains(name_a) {
                    println!("name_a, name_v: {:?} {:?}", name_a, name_v);
                    config.a_private_inputs.push((name_v.into(), lhs_var.id));
                    lhs_name = name_v;
                }
            }
            for name_b in input_b_names.as_slice() {
                if name_v.contains(name_b) {
                    println!("name_b, name_v: {:?} {:?}", name_b, name_v);
                    config.b_private_inputs.push((name_v.into(), lhs_var.id));
                    lhs_name = name_v;
                }
            }
        }

        for name_v in rhs_var.names.as_slice() {
            for name_a in input_a_names.as_slice() {
                if name_v.contains(name_a) {
                    config.a_private_inputs.push((name_v.into(), rhs_var.id));
                    rhs_name = name_v;
                }
            }
            for name_b in input_b_names.as_slice() {
                if name_v.contains(name_b) {
                    config.b_private_inputs.push((name_v.into(), rhs_var.id));
                    rhs_name = name_v;
                }
            }
        }

        for name_v in out_var.names.as_slice() {
            for name_o in output_names.as_slice() {
                if name_v.contains(name_o) {
                    config.outputs.push((name_v.into(), out_var.id));
                }
            }
        }

        let lhs = if lhs_var.is_const {
            Wire::Const(lhs_var.const_value)
        } else {
            Wire::Var(if let Some(crt) = used_vars.get(&gate.lh_input) {
                crt.clone()
            } else {
                // check if const or not
                println!("Input added lh: {:?} {:?}", lhs_name, gate.lh_input);
                // TODO: better way to handle name
                let v = builder.add_input::<u32>(lhs_name.into()).unwrap();
                used_vars.insert(gate.lh_input, v.repr.clone());
                v.repr
            })
        };

        let rhs = if rhs_var.is_const {
            Wire::Const(rhs_var.const_value)
        } else {
            Wire::Var(if let Some(crt) = used_vars.get(&gate.rh_input) {
                crt.clone()
            } else {
                // check if const or not
                let v = builder.add_input::<u32>(rhs_name.into()).unwrap();
                println!("Input added rh: {:?} {:?}", rhs_name, gate.rh_input);
                used_vars.insert(gate.rh_input, v.repr.clone());
                v.repr
            })
        };

        match (lhs, rhs) {
            (Wire::Const(c), Wire::Var(v)) | (Wire::Var(v), Wire::Const(c)) => {
                match gate.gate_type {
                    AGateType::AMul => {
                        // add cmul gate.
                        let mut state = builder.state().borrow_mut();
                        let out = cmul(&mut state, &v, c);
                        used_vars.insert(gate.output, out.clone());

                        o = gate.output;
                        output = Some(out);
                    }
                    AGateType::AAdd => {
                        // add cmul gate.
                        let mut state = builder.state().borrow_mut();
                        let out = cmul(&mut state, &v, c);
                        used_vars.insert(gate.output, out.clone());

                        o = gate.output;
                        output = Some(out);
                    }

                    _ => panic!("This gate type not supported yet. {:?}", gate.gate_type),
                }
            }
            (Wire::Var(lhs), Wire::Var(rhs)) => {
                match gate.gate_type {
                    AGateType::AAdd => {
                        // add add gate to builder
                        // check if crt repr exists in the used_vars
                        // if not, create new feed and put in the map
                        let mut state = builder.state().borrow_mut();
                        let out = add(&mut state, &lhs, &rhs).unwrap();
                        used_vars.insert(gate.output, out.clone());

                        o = gate.output;
                        output = Some(out);
                    }
                    AGateType::AMul => {
                        // add mul gate or cmul gate
                        let mut state = builder.state().borrow_mut();
                        let out = mul(&mut state, &lhs, &rhs).unwrap();
                        used_vars.insert(gate.output, out.clone());

                        o = gate.output;
                        output = Some(out);
                    }
                    AGateType::ALt => {
                        // call gadgets here
                        // sub
                        // sign
                    }
                    _ => panic!("This gate type not supported yet. {:?}", gate.gate_type),
                }
            }
            _ => {
                panic!("Unsupported operation for two const values. Consider pre calculation.")
            }
        };
    }

    // add output
    if output.is_none() {
        panic!("Output is not defined");
    }
    builder.add_output(&output.unwrap());
    builder.build().and_then(|circ| Ok((circ, config)))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
    // Load circuit file
    let raw = fs::read_to_string("./examples/circ.json")?;
    let raw_circ: RawCircuit = serde_json::from_str(&raw)?;

    let (circ, config) = parse_raw_circuit(&raw_circ)?;
    println!("Config: {:?}", config);
    println!(
        "Circuit inputs: {:?}",
        circ.inputs()
            .iter()
            .map(|i| i.name.clone())
            .collect::<Vec<_>>()
    );
    // todo!();
    let circ = Arc::new(circ);
    println!("[MPZ circ] inputs: {:?}", circ.inputs().len());
    println!("[MPZ circ] outputs: {:#?}", circ.outputs());

    let (mut generator_channel, mut evaluator_channel) = MemoryDuplex::<GarbleMessage>::new();
    let (generator_ot_send, evaluator_ot_recv) = mock_ot_shared_pair();
    // setup generator and evaluator
    let gen_config = BMR16GeneratorConfig {
        encoding_commitments: false,
        batch_size: 1024,
        num_wires: 10,
    };
    let seed = [0; 32];
    let generator = BMR16Generator::<10>::new(gen_config, seed);

    let ev_config = BMR16EvaluatorConfig { batch_size: 1024 };
    let evaluator = BMR16Evaluator::<10>::new(ev_config);

    let generator_fut = {
        // println!("[GEN]-----------start generator--------------");
        let out_ref = ValueRef::Value {
            id: ValueId::new("output"),
        };

        // TODO: need to check if the input of the arithmetic circuit corresponds to intended input variables.
        let input_config = config.gen_a_input_config();
        let input_refs = config.get_input_refs();
        let circ = circ.clone();

        async move {
            generator
                .setup_inputs(
                    "test_gc",
                    &input_config,
                    &mut generator_channel,
                    &generator_ot_send,
                )
                .await
                .unwrap();

            generator_channel
                .send(GarbleMessage::ArithEncryptedGates(vec![]))
                .await
                .unwrap();
            let _encoded_outputs = generator
                .generate(
                    circ,
                    &input_refs,
                    &[out_ref.clone()],
                    &mut generator_channel,
                )
                .await
                .unwrap();
            generator
                .decode(&[out_ref], &mut generator_channel)
                .await
                .unwrap();
        }
    };

    let evaluator_fut = {
        println!("[EV]-----------start evaluator--------------");
        let out_ref = ValueRef::Value {
            id: ValueId::new("output"),
        };

        let input_config = config.gen_b_input_config();
        let input_refs = config.get_input_refs();
        println!("[EV] async move");
        async move {
            println!("[EV] setup inputs start");
            evaluator
                .setup_inputs(
                    "test_gc",
                    &input_config,
                    &mut evaluator_channel,
                    &evaluator_ot_recv,
                )
                .await
                .unwrap();
            println!("[EV] setup inputs done");
            println!("[EV] start evaluator.evaluate()");

            let _encoded_outputs = evaluator
                .evaluate(
                    circ.clone(),
                    &input_refs,
                    &[out_ref.clone()],
                    &mut evaluator_channel,
                )
                .await
                .unwrap();
            let decoded = evaluator
                .decode(&[out_ref], &mut evaluator_channel)
                .await
                .unwrap();
            Some(decoded)
        }
    };

    let (_, evaluator_output) = tokio::join!(generator_fut, evaluator_fut);
    println!("Decoded evaluator output: {:?}", evaluator_output);

    Ok(())
}
