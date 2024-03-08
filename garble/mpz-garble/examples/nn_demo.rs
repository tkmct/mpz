use futures::SinkExt;
use mpz_circuits::{
    arithmetic::{
        ops::{add, cmul, crt_sign, mul, sub},
        types::{ArithValue, CircInput, CrtRepr, CrtValueType},
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

use regex::Regex;
use std::cmp::Ordering;

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
    pub public_constants: Vec<(String, u32)>,
    pub outputs: Vec<(String, u32)>,
}

impl CircuitConfig {
    pub fn new() -> Self {
        Self {
            a_private_inputs: vec![],
            b_private_inputs: vec![],
            public_constants: vec![],
            outputs: vec![],
        }
    }

    fn finalize(&mut self) {
        let re = Regex::new(r"(\w)\[?(\d+)\]?\[?(\d+)?\]?").unwrap();

        let s = |a: &(String, u32), b: &(String, u32)| {
            let a = &a.0;
            let b = &b.0;
            let caps_a = re.captures(a).unwrap();
            let caps_b = re.captures(b).unwrap();

            let a_head = caps_a.get(1).unwrap().as_str();
            let b_head = caps_b.get(1).unwrap().as_str();

            // Compare the first char
            match a_head.cmp(&b_head) {
                Ordering::Equal => {
                    // compare the first number
                    let a_index1: i32 = caps_a.get(2).unwrap().as_str().parse().unwrap();
                    let b_index1: i32 = caps_b.get(2).unwrap().as_str().parse().unwrap();

                    match a_index1.cmp(&b_index1) {
                        Ordering::Equal => {
                            // compare the second number if exists
                            match (caps_a.get(3), caps_b.get(3)) {
                                (Some(a_index2), Some(b_index2)) => {
                                    let a_index2_val: i32 = a_index2.as_str().parse().unwrap();
                                    let b_index2_val: i32 = b_index2.as_str().parse().unwrap();
                                    a_index2_val.cmp(&b_index2_val)
                                }
                                (None, None) => Ordering::Equal,
                                (Some(_), None) => Ordering::Greater,
                                (None, Some(_)) => Ordering::Less,
                            }
                        }
                        other => other,
                    }
                }
                other => other,
            }
        };
        self.a_private_inputs.sort_by(s.clone());
        // self.a_private_inputs.sort_by(|a, b| a.0.cmp(&b.0));
        self.a_private_inputs.dedup_by(|a, b| a.1 == b.1);

        self.b_private_inputs.sort_by(s.clone());
        self.b_private_inputs.dedup_by(|a, b| a.1 == b.1);

        self.outputs.sort_by(s.clone());
        self.outputs.dedup_by(|a, b| a.1 == b.1);

        self.public_constants.sort_by(s.clone());
        self.public_constants.dedup_by(|a, b| a.1 == b.1);
    }

    /// generate input config for the first party.
    /// - a_input: list of actual values for input from Alice. this shoule be sorted with variable name.
    pub fn gen_a_input_config(
        &self,
        a_input: &[ArithValue],
        ordering: &[String],
    ) -> Result<Vec<ArithValueIdConfig>, Box<dyn std::error::Error>> {
        if self.a_private_inputs.len() != a_input.len() {
            return Err("Input length does not match".into());
        }
        let mut store = HashMap::<String, ArithValueIdConfig>::new();

        for (a_priv, v) in self.a_private_inputs.iter().zip(a_input) {
            store.insert(
                a_priv.0.clone(),
                ArithValueIdConfig::Private {
                    id: ValueId::new(&a_priv.0),
                    ty: CrtValueType::U32,
                    value: Some(*v),
                },
            );
        }

        for b_priv in self.b_private_inputs.iter() {
            store.insert(
                b_priv.0.clone(),
                ArithValueIdConfig::Private {
                    id: ValueId::new(&b_priv.0),
                    ty: CrtValueType::U32,
                    value: None,
                },
            );
        }

        for c in self.public_constants.iter() {
            store.insert(
                c.0.clone(),
                ArithValueIdConfig::Public {
                    id: ValueId::new(&c.0),
                    ty: CrtValueType::U32,
                    value: ArithValue::from(c.1),
                },
            );
        }

        Ok(ordering
            .iter()
            .map(|name| store.get(name).unwrap().clone())
            .collect::<Vec<_>>())
    }

    /// generate input config for the second party.
    /// - b_input: list of actual values for input from Alice. this should be sorted with variable name.
    pub fn gen_b_input_config(
        &self,
        b_input: &[ArithValue],
        ordering: &[String],
    ) -> Result<Vec<ArithValueIdConfig>, Box<dyn std::error::Error>> {
        if self.b_private_inputs.len() != b_input.len() {
            return Err("Input length does not match".into());
        }

        let mut store = HashMap::<String, ArithValueIdConfig>::new();

        for a_priv in self.a_private_inputs.iter() {
            store.insert(
                a_priv.0.clone(),
                ArithValueIdConfig::Private {
                    id: ValueId::new(&a_priv.0),
                    ty: CrtValueType::U32,
                    value: None,
                },
            );
        }

        for (b_priv, v) in self.b_private_inputs.iter().zip(b_input) {
            store.insert(
                b_priv.0.clone(),
                ArithValueIdConfig::Private {
                    id: ValueId::new(&b_priv.0),
                    ty: CrtValueType::U32,
                    value: Some(*v),
                },
            );
        }

        for c in self.public_constants.iter() {
            store.insert(
                c.0.clone(),
                ArithValueIdConfig::Public {
                    id: ValueId::new(&c.0),
                    ty: CrtValueType::U32,
                    value: ArithValue::from(c.1),
                },
            );
        }

        Ok(ordering
            .iter()
            .map(|name| store.get(name).unwrap().clone())
            .collect::<Vec<_>>())
    }

    pub fn get_input_refs(&self, ordering: &[String]) -> Vec<ValueRef> {
        let mut store = HashMap::<String, ValueRef>::new();
        for a_priv in self.a_private_inputs.iter() {
            store.insert(
                a_priv.0.clone(),
                ValueRef::Value {
                    id: ValueId::new(&a_priv.0),
                },
            );
        }
        for b_priv in self.b_private_inputs.iter() {
            store.insert(
                b_priv.0.clone(),
                ValueRef::Value {
                    id: ValueId::new(&b_priv.0),
                },
            );
        }
        for c in self.public_constants.iter() {
            store.insert(
                c.0.clone(),
                ValueRef::Value {
                    id: ValueId::new(&c.0),
                },
            );
        }

        ordering
            .iter()
            .map(|name| store.get(name).unwrap().clone())
            .collect::<Vec<_>>()
    }

    pub fn get_output_refs(&self) -> Vec<ValueRef> {
        self.outputs
            .iter()
            .map(|o| ValueRef::Value {
                id: ValueId::new(&o.0),
            })
            .collect::<Vec<_>>()
    }
}

enum Wire {
    Var(CrtRepr),
    Const(u32),
}

const ACCURACY: &str = "100%";

/// Parse raw circuit to bmr16 arithmeti circuit representation
/// specify private inputs from both parties
fn parse_raw_circuit(
    raw_circ: &RawCircuit,
    private_inputs_from_a: &[&str],
    private_inputs_from_b: &[&str],
    outputs: &[&str],
) -> Result<(ArithmeticCircuit, CircuitConfig), BuilderError> {
    let mut config = CircuitConfig::new();
    let builder = ArithmeticCircuitBuilder::new();
    // take each gate and append in the builder
    // mark input wire
    let mut used_vars = HashMap::<u32, CrtRepr>::new();
    let mut used_constants = HashMap::<u32, CircInput>::new();

    for gate in raw_circ.gates.iter() {
        let lhs_var = raw_circ.get_node_by_id(gate.lh_input).unwrap();
        let rhs_var = raw_circ.get_node_by_id(gate.rh_input).unwrap();
        let out_var = raw_circ.get_node_by_id(gate.output).unwrap();

        let lhs_name = lhs_var
            .names
            .clone()
            .into_iter()
            .find(|name_v| {
                if private_inputs_from_a
                    .iter()
                    .any(|&name_a| name_v.contains(name_a))
                {
                    config.a_private_inputs.push((name_v.into(), lhs_var.id));
                    return true;
                }
                if private_inputs_from_b
                    .iter()
                    .any(|&name_b| name_v.contains(name_b))
                {
                    config.b_private_inputs.push((name_v.into(), lhs_var.id));
                    return true;
                }
                false
            })
            .unwrap_or("".into());
        let rhs_name = rhs_var
            .names
            .clone()
            .into_iter()
            .find(|name_v| {
                if private_inputs_from_a
                    .iter()
                    .any(|&name_a| name_v.contains(name_a))
                {
                    config.a_private_inputs.push((name_v.into(), rhs_var.id));
                    return true;
                }
                if private_inputs_from_b
                    .iter()
                    .any(|&name_b| name_v.contains(name_b))
                {
                    config.b_private_inputs.push((name_v.into(), rhs_var.id));
                    return true;
                }
                false
            })
            .unwrap_or("".into());

        for name_v in out_var.names.as_slice() {
            if outputs.iter().any(|&name_o| name_v.contains(name_o)) {
                config.outputs.push((name_v.into(), out_var.id));
            }
        }

        let lhs = if lhs_var.is_const {
            Wire::Const(lhs_var.const_value)
        } else {
            Wire::Var(if let Some(crt) = used_vars.get(&gate.lh_input) {
                crt.clone()
            } else {
                let v = builder.add_input::<u32>(lhs_name.clone()).unwrap();
                // println!(
                //     "Input added lh: {:?} {:?} {:?}",
                //     lhs_name, gate.lh_input, v.name
                // );
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
                let v = builder.add_input::<u32>(rhs_name.clone()).unwrap();
                // println!(
                //     "Input added rh: {:?} {:?} {:?}",
                //     rhs_name, gate.rh_input, v.name
                // );
                used_vars.insert(gate.rh_input, v.repr.clone());
                v.repr
            })
        };

        match (lhs, rhs) {
            (Wire::Const(c), Wire::Var(v)) => match gate.gate_type {
                AGateType::AMul => {
                    let mut state = builder.state().borrow_mut();
                    let out = cmul(&mut state, &v, c);
                    used_vars.insert(gate.output, out.clone());
                }
                AGateType::AAdd => {
                    let c_repr = if let Some(c_in) = used_constants.get(&c) {
                        c_in.repr.clone()
                    } else {
                        let name = format!("CONST_{}", c);
                        let v = builder.add_input::<u32>(name.clone()).unwrap();
                        config.public_constants.push((name, c));
                        used_constants.insert(c, v.clone());
                        v.repr
                    };

                    let mut state = builder.state().borrow_mut();
                    let out = add(&mut state, &v, &c_repr).unwrap();
                    used_vars.insert(gate.output, out.clone());
                }
                AGateType::ASub => {
                    let c_repr = if let Some(c_in) = used_constants.get(&c) {
                        c_in.repr.clone()
                    } else {
                        let name = format!("CONST_{}", c);
                        let v = builder.add_input::<u32>(name.clone()).unwrap();
                        config.public_constants.push((name, c));
                        used_constants.insert(c, v.clone());
                        v.repr
                    };

                    let mut state = builder.state().borrow_mut();
                    let out = sub(&mut state, &c_repr, &v).unwrap();
                    used_vars.insert(gate.output, out.clone());
                }
                AGateType::ALt => {
                    let c_repr = if let Some(c_in) = used_constants.get(&c) {
                        c_in.repr.clone()
                    } else {
                        let name = format!("CONST_{}", c);
                        let v = builder.add_input::<u32>(name.clone()).unwrap();
                        config.public_constants.push((name, c));
                        used_constants.insert(c, v.clone());
                        v.repr
                    };

                    let mut state = builder.state().borrow_mut();
                    let z = sub(&mut state, &c_repr, &v).unwrap();
                    let out = crt_sign::<10>(&mut state, &z, ACCURACY).unwrap();
                    used_vars.insert(gate.output, out.clone());
                }

                _ => panic!("This gate type not supported yet. {:?}", gate.gate_type),
            },
            (Wire::Var(v), Wire::Const(c)) => match gate.gate_type {
                AGateType::AMul => {
                    let mut state = builder.state().borrow_mut();
                    let out = cmul(&mut state, &v, c);
                    used_vars.insert(gate.output, out.clone());
                }
                AGateType::AAdd => {
                    let c_repr = if let Some(c_in) = used_constants.get(&c) {
                        c_in.repr.clone()
                    } else {
                        let name = format!("CONST_{}", c);
                        let v = builder.add_input::<u32>(name.clone()).unwrap();
                        config.public_constants.push((name, c));
                        used_constants.insert(c, v.clone());
                        v.repr
                    };

                    let mut state = builder.state().borrow_mut();
                    let out = add(&mut state, &v, &c_repr).unwrap();
                    used_vars.insert(gate.output, out.clone());
                }
                AGateType::ASub => {
                    let c_repr = if let Some(c_in) = used_constants.get(&c) {
                        c_in.repr.clone()
                    } else {
                        let name = format!("CONST_{}", c);
                        let v = builder.add_input::<u32>(name.clone()).unwrap();
                        config.public_constants.push((name, c));
                        used_constants.insert(c, v.clone());
                        v.repr
                    };

                    let mut state = builder.state().borrow_mut();
                    let out = sub(&mut state, &v, &c_repr).unwrap();
                    used_vars.insert(gate.output, out.clone());
                }
                AGateType::ALt => {
                    let c_repr = if let Some(c_in) = used_constants.get(&c) {
                        c_in.repr.clone()
                    } else {
                        let name = format!("CONST_{}", c);
                        let v = builder.add_input::<u32>(name.clone()).unwrap();
                        config.public_constants.push((name, c));
                        used_constants.insert(c, v.clone());
                        v.repr
                    };

                    let mut state = builder.state().borrow_mut();
                    let z = sub(&mut state, &v, &c_repr).unwrap();
                    let out = crt_sign::<10>(&mut state, &z, ACCURACY).unwrap();
                    used_vars.insert(gate.output, out.clone());
                }

                _ => panic!("This gate type not supported yet. {:?}", gate.gate_type),
            },
            (Wire::Var(lhs), Wire::Var(rhs)) => {
                match gate.gate_type {
                    AGateType::AAdd => {
                        // check if crt repr exists in the used_vars
                        // if not, create new feed and put in the map
                        let mut state = builder.state().borrow_mut();
                        let out = add(&mut state, &lhs, &rhs).unwrap();
                        used_vars.insert(gate.output, out.clone());
                    }
                    AGateType::AMul => {
                        let mut state = builder.state().borrow_mut();
                        let out = mul(&mut state, &lhs, &rhs).unwrap();
                        used_vars.insert(gate.output, out.clone());
                    }
                    AGateType::ASub => {
                        let mut state = builder.state().borrow_mut();
                        let out = sub(&mut state, &lhs, &rhs).unwrap();
                        used_vars.insert(gate.output, out.clone());
                    }
                    AGateType::ALt => {
                        let mut state = builder.state().borrow_mut();
                        let z = sub(&mut state, &lhs, &rhs).unwrap();
                        let out = crt_sign::<10>(&mut state, &z, ACCURACY).unwrap();
                        used_vars.insert(gate.output, out.clone());
                    }
                    _ => panic!("This gate type not supported yet. {:?}", gate.gate_type),
                }
            }
            _ => {
                panic!("Unsupported operation for two const values. Consider pre calculation.")
            }
        };
    }
    for out in config.outputs.iter() {
        if let Some(crt) = used_vars.get(&out.1) {
            builder.add_output(crt);
        }
    }
    config.finalize();
    builder.build().map(|circ| (circ, config))
}

// TODO: split the input config into A input and B input
// then when provided to gen_config, recursively flat the values and create ArithValue from them.
#[derive(Deserialize, Debug)]
struct RawNNInput {
    #[serde(rename(deserialize = "in"))]
    input: Vec<[u64; 2]>,
    out: Vec<[u64; 4]>,
    #[serde(rename(deserialize = "w1"))]
    w0: Vec<Vec<u64>>,
    #[serde(rename(deserialize = "w2"))]
    w1: Vec<Vec<u64>>,
    #[serde(rename(deserialize = "w3"))]
    w2: Vec<Vec<u64>>,
    #[serde(rename(deserialize = "w4"))]
    w3: Vec<Vec<u64>>,
    #[serde(rename(deserialize = "b1"))]
    b0: Vec<u64>,
    #[serde(rename(deserialize = "b2"))]
    b1: Vec<u64>,
    #[serde(rename(deserialize = "b3"))]
    b2: Vec<u64>,
    #[serde(rename(deserialize = "b4"))]
    b3: Vec<u64>,
}

// This method is designed to specifically parse nn fc 2_5_7_11_4 model.
fn parse_input() -> RawNNInput {
    let raw = fs::read_to_string("./examples/nn_input.json").unwrap();
    let raw_input: RawNNInput = serde_json::from_str(&raw).unwrap();
    raw_input
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
    // Load circuit file
    let raw = fs::read_to_string("./examples/nn_circuit.json")?;
    let raw_circ: RawCircuit = serde_json::from_str(&raw)?;
    let raw_input = parse_input();
    // println!("Raw_input: {:?}", raw_input.w0);

    let (circ, config) = parse_raw_circuit(
        &raw_circ,
        &vec![
            "0.w0", "0.w1", "0.w2", "0.w3", "0.b0", "0.b1", "0.b2", "0.b3",
        ],
        &vec!["0.in"],
        &vec!["0.out"],
    )?;
    let ordering = circ
        .inputs()
        .iter()
        .map(|i| i.name.clone())
        .collect::<Vec<_>>();

    // println!("Config: {:?}", config);
    println!(
        "Circuit inputs: {:?}",
        circ.inputs()
            .iter()
            .map(|i| i.name.clone())
            .collect::<Vec<_>>()
    );
    let circ = Arc::new(circ);
    // println!("[MPZ circ] inputs: {:?}", circ.inputs().len());
    println!("[MPZ circ] outputs: {:#?}", circ.outputs().len());

    let (mut generator_channel, mut evaluator_channel) = MemoryDuplex::<GarbleMessage>::new();
    let (generator_ot_send, evaluator_ot_recv) = mock_ot_shared_pair();
    // setup generator and evaluator
    let gen_config =
        BMR16GeneratorConfig::new_with_moduli_list(false, 1024, 10, circ.moduli().into());
    let seed = [0; 32];
    let generator = BMR16Generator::new(gen_config, seed);

    let ev_config = BMR16EvaluatorConfig::new(1024);
    let evaluator = BMR16Evaluator::new(ev_config);

    let generator_fut = {
        let out_refs = config.get_output_refs();
        // TODO: need to check if the input of the arithmetic circuit corresponds to intended input variables.
        // -> this can be possible only if
        // TODO: load actual data from the model file and pass it to gen_input_config
        let a_input = vec![
            raw_input.b0.to_vec(),
            raw_input.b1.to_vec(),
            raw_input.b2.to_vec(),
            raw_input.b3.to_vec(),
            raw_input
                .w0
                .iter()
                .flat_map(|i| i.clone())
                .collect::<Vec<_>>(),
            raw_input
                .w1
                .iter()
                .flat_map(|i| i.clone())
                .collect::<Vec<_>>(),
            raw_input
                .w2
                .iter()
                .flat_map(|i| i.clone())
                .collect::<Vec<_>>(),
            raw_input
                .w3
                .iter()
                .flat_map(|i| i.clone())
                .collect::<Vec<_>>(),
        ]
        .into_iter()
        .flat_map(|i| i)
        .map(|v| ArithValue::try_from(v).unwrap())
        .collect::<Vec<ArithValue>>();

        let input_config = config.gen_a_input_config(&a_input, &ordering).unwrap();
        println!("config.output: {:?}", config.outputs);

        // This should be sorted same with circ.inputs
        let input_refs = config.get_input_refs(&ordering);
        let circ = circ.clone();
        println!("input_config",);
        input_config.clone().iter().for_each(|c| {
            println!("ValueID {:?} value: {:?}", c.id(), c.value());
        });

        // todo!();
        // println!("input_refs: {:?}", input_refs);

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
                .generate(circ, &input_refs, &out_refs.clone(), &mut generator_channel)
                .await
                .unwrap();
            generator
                .decode(&out_refs, &mut generator_channel)
                .await
                .unwrap();
        }
    };

    let evaluator_fut = {
        println!("[EV]-----------start evaluator--------------");
        let out_refs = config.get_output_refs();
        let b_input = raw_input.input[0]
            .into_iter()
            .map(|v| ArithValue::try_from(v).unwrap())
            .collect::<Vec<ArithValue>>();

        println!("b_input: {:?}", b_input);
        let input_config = config.gen_b_input_config(&b_input, &ordering).unwrap();
        // println!("b_input_config: {:?}", input_config);
        let input_refs = config.get_input_refs(&ordering);
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
                    &out_refs.clone(),
                    &mut evaluator_channel,
                )
                .await
                .unwrap();
            let decoded = evaluator
                .decode(&out_refs, &mut evaluator_channel)
                .await
                .unwrap();
            Some(decoded)
        }
    };

    let (_, evaluator_output) = tokio::join!(generator_fut, evaluator_fut);
    println!("Decoded evaluator output: {:?}", evaluator_output);

    Ok(())
}
