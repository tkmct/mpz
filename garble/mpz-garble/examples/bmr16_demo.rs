use std::sync::Arc;

use futures::SinkExt;
use mpz_core::value::{ValueId, ValueRef};
use utils_aio::duplex::MemoryDuplex;

use mpz_circuits::{
    arithmetic::{
        ops::{add, cmul, mul},
        types::{ArithValue, CrtRepr, CrtValueType},
    },
    ArithmeticCircuit, ArithmeticCircuitBuilder, BuilderError,
};
use mpz_garble_core::msg::GarbleMessage;
use mpz_ot::mock::mock_ot_shared_pair;

use mpz_garble::bmr16::{
    config::ArithValueIdConfig,
    evaluator::{BMR16Evaluator, BMR16EvaluatorConfig},
    generator::{BMR16Generator, BMR16GeneratorConfig},
};

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, error, fs};

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
pub struct ArithmeticVar {
    pub var_id: u32,
    pub var_name: String,
    pub is_const: bool,
    pub const_value: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ArithmeticNode {
    pub gate_id: u32,
    pub gate_type: AGateType,
    pub input_lhs_id: u32,
    pub input_rhs_id: u32,
    pub output_id: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RawCircuit {
    pub gate_count: u32,
    pub var_count: u32,
    pub vars: HashMap<u32, ArithmeticVar>,
    pub gates: HashMap<u32, ArithmeticNode>,
}

enum Wire {
    Var(CrtRepr),
    Const(u32),
}

impl TryInto<ArithmeticCircuit> for RawCircuit {
    type Error = BuilderError;

    fn try_into(self) -> Result<ArithmeticCircuit, BuilderError> {
        let builder = ArithmeticCircuitBuilder::new();
        // take each gate and append in the builder
        // mark input wire
        let mut used_vars = HashMap::<u32, CrtRepr>::new();
        // TODO: load output from config file?
        // for now, use output of last gate as an output of a circuit.
        let mut output = None;
        let mut o = 0;

        let mut keys = self.gates.keys().collect::<Vec<_>>();
        keys.sort();

        for id in keys.iter() {
            let gate = self.gates.get(id).expect("gate should be set");
            let lhs_var = self.vars.get(&gate.input_lhs_id).unwrap();
            let rhs_var = self.vars.get(&gate.input_rhs_id).unwrap();

            let lhs = if lhs_var.is_const {
                Wire::Const(lhs_var.const_value)
            } else {
                Wire::Var(if let Some(crt) = used_vars.get(&gate.input_lhs_id) {
                    crt.clone()
                } else {
                    // check if const or not
                    println!("Input added: {:?}", gate.input_lhs_id);
                    let v = builder.add_input::<u32>().unwrap();
                    used_vars.insert(gate.input_lhs_id, v.clone());
                    v
                })
            };

            let rhs = if rhs_var.is_const {
                Wire::Const(rhs_var.const_value)
            } else {
                Wire::Var(if let Some(crt) = used_vars.get(&gate.input_rhs_id) {
                    crt.clone()
                } else {
                    // check if const or not
                    let v = builder.add_input::<u32>().unwrap();
                    println!("Input added: {:?}", gate.input_rhs_id);
                    used_vars.insert(gate.input_rhs_id, v.clone());
                    v
                })
            };

            match (lhs, rhs) {
                (Wire::Const(c), Wire::Var(v)) | (Wire::Var(v), Wire::Const(c)) => {
                    match gate.gate_type {
                        AGateType::AMul => {
                            // add cmul gate.
                            let mut state = builder.state().borrow_mut();
                            let out = cmul(&mut state, &v, c);
                            used_vars.insert(gate.output_id, out.clone());

                            o = gate.output_id;
                            output = Some(out);
                        }
                        AGateType::AAdd => {
                            // add cmul gate.
                            let mut state = builder.state().borrow_mut();
                            let out = cmul(&mut state, &v, c);
                            used_vars.insert(gate.output_id, out.clone());

                            o = gate.output_id;
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
                            used_vars.insert(gate.output_id, out.clone());

                            o = gate.output_id;
                            output = Some(out);
                        }
                        AGateType::AMul => {
                            // add mul gate or cmul gate
                            let mut state = builder.state().borrow_mut();
                            let out = mul(&mut state, &lhs, &rhs).unwrap();
                            used_vars.insert(gate.output_id, out.clone());

                            o = gate.output_id;
                            output = Some(out);
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
        println!("output_id: {:?}", o);
        if output.is_none() {
            panic!("Output is not defined");
        }
        builder.add_output(&output.unwrap());
        builder.build()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn error::Error>> {
    // Load circuit file
    let raw = fs::read_to_string("./examples/circ.json")?;
    let raw_circ: RawCircuit = serde_json::from_str(&raw)?;
    // dbg!(circ.clone());

    let circ: Arc<ArithmeticCircuit> = Arc::new(raw_circ.try_into()?);
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
        let a_0 = ValueRef::Value {
            id: ValueId::new("a_0"),
        };
        let a_1 = ValueRef::Value {
            id: ValueId::new("a_1"),
        };
        let a_2 = ValueRef::Value {
            id: ValueId::new("a_2"),
        };
        let b_0 = ValueRef::Value {
            id: ValueId::new("b_0"),
        };
        let b_1 = ValueRef::Value {
            id: ValueId::new("b_1"),
        };
        let b_2 = ValueRef::Value {
            id: ValueId::new("b_2"),
        };

        let out_ref = ValueRef::Value {
            id: ValueId::new("output"),
        };

        // list input configs
        let input_configs = vec![
            ArithValueIdConfig::Private {
                id: ValueId::new("a_0"),
                ty: CrtValueType::U32,
                value: Some(ArithValue::U32(10)),
            },
            ArithValueIdConfig::Private {
                id: ValueId::new("a_1"),
                ty: CrtValueType::U32,
                value: Some(ArithValue::U32(10)),
            },
            ArithValueIdConfig::Private {
                id: ValueId::new("a_2"),
                ty: CrtValueType::U32,
                value: Some(ArithValue::U32(10)),
            },
            ArithValueIdConfig::Private {
                id: ValueId::new("b_0"),
                ty: CrtValueType::U32,
                value: None,
            },
            ArithValueIdConfig::Private {
                id: ValueId::new("b_1"),
                ty: CrtValueType::U32,
                value: None,
            },
            ArithValueIdConfig::Private {
                id: ValueId::new("b_2"),
                ty: CrtValueType::U32,
                value: None,
            },
        ];

        let circ = circ.clone();

        // prepare input configs for a, b
        async move {
            generator
                .setup_inputs(
                    "test_gc",
                    &input_configs,
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
                    &[a_0, a_1, a_2, b_0, b_1, b_2],
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
        // let (_sink, mut stream) = evaluator_channel.split();
        let a_0 = ValueRef::Value {
            id: ValueId::new("a_0"),
        };
        let a_1 = ValueRef::Value {
            id: ValueId::new("a_1"),
        };
        let a_2 = ValueRef::Value {
            id: ValueId::new("a_2"),
        };
        let b_0 = ValueRef::Value {
            id: ValueId::new("b_0"),
        };
        let b_1 = ValueRef::Value {
            id: ValueId::new("b_1"),
        };
        let b_2 = ValueRef::Value {
            id: ValueId::new("b_2"),
        };

        let out_ref = ValueRef::Value {
            id: ValueId::new("output"),
        };

        // list input configs
        let input_configs = vec![
            ArithValueIdConfig::Private {
                id: ValueId::new("a_0"),
                ty: CrtValueType::U32,
                value: None,
            },
            ArithValueIdConfig::Private {
                id: ValueId::new("a_1"),
                ty: CrtValueType::U32,
                value: None,
            },
            ArithValueIdConfig::Private {
                id: ValueId::new("a_2"),
                ty: CrtValueType::U32,
                value: None,
            },
            ArithValueIdConfig::Private {
                id: ValueId::new("b_0"),
                ty: CrtValueType::U32,
                value: Some(ArithValue::U32(10)),
            },
            ArithValueIdConfig::Private {
                id: ValueId::new("b_1"),
                ty: CrtValueType::U32,
                value: Some(ArithValue::U32(10)),
            },
            ArithValueIdConfig::Private {
                id: ValueId::new("b_2"),
                ty: CrtValueType::U32,
                value: Some(ArithValue::U32(10)),
            },
        ];

        println!("[EV] async move");
        async move {
            println!("[EV] setup inputs start");
            evaluator
                .setup_inputs(
                    "test_gc",
                    &input_configs,
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
                    &[a_0, a_1, a_2, b_0, b_1, b_2],
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
