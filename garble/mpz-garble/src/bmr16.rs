//! bmr16 generator, evaluator

pub mod config;
pub mod evaluator;
pub mod generator;
pub mod ot;
pub mod registry;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::value::{ValueId, ValueRef};
    use futures::SinkExt;
    use utils_aio::duplex::MemoryDuplex;

    use mpz_circuits::{
        arithmetic::{
            ops::{crt_sign, mul, sub},
            types::{ArithValue, CrtValueType},
        },
        ArithmeticCircuit, ArithmeticCircuitBuilder,
    };
    use mpz_garble_core::msg::GarbleMessage;
    use mpz_ot::mock::mock_ot_shared_pair;

    use crate::bmr16::{
        config::ArithValueIdConfig,
        evaluator::{BMR16Evaluator, BMR16EvaluatorConfig},
        generator::{BMR16Generator, BMR16GeneratorConfig},
    };

    // calculate  a * b < x
    fn simple_circ() -> Arc<ArithmeticCircuit> {
        let builder = ArithmeticCircuitBuilder::new();

        let a = builder.add_input::<u32>("a".into()).unwrap();
        let b = builder.add_input::<u32>("b".into()).unwrap();
        let c = builder.add_input::<u32>("c".into()).unwrap();
        let out;
        {
            let mut state = builder.state().borrow_mut();
            let d = mul(&mut state, &a.repr, &b.repr).unwrap();
            // if a*b > x => 1, otherwise 0
            let e = sub(&mut state, &d, &c.repr).unwrap();
            let f = crt_sign::<10>(&mut state, &e, "100%").unwrap();
            out = mul(&mut state, &a.repr, &f).unwrap();
        }

        builder.add_output(&out);
        Arc::new(builder.build().unwrap())
    }

    #[tokio::test]
    async fn test_bmr16() {
        // let (mut sink, stream) = mpsc::channel::<GarbleMessage>(10);

        let (mut generator_channel, mut evaluator_channel) = MemoryDuplex::<GarbleMessage>::new();
        let (generator_ot_send, evaluator_ot_recv) = mock_ot_shared_pair();
        let circ = simple_circ();

        // setup generator and evaluator
        let gen_config =
            BMR16GeneratorConfig::new_with_moduli_list(false, 1024, 10, circ.moduli().into());
        let seed = [0; 32];
        let generator = BMR16Generator::new(gen_config, seed);

        let ev_config = BMR16EvaluatorConfig { batch_size: 1024 };
        let evaluator = BMR16Evaluator::new(ev_config);

        let generator_fut = {
            println!("[GEN]-----------start generator--------------");
            // let (mut sink, _stream) = generator_channel.split();
            let a_ref = ValueRef::Value {
                id: ValueId::new("input/a"),
            };
            let b_ref = ValueRef::Value {
                id: ValueId::new("input/b"),
            };
            let c_ref = ValueRef::Value {
                id: ValueId::new("input/c"),
            };
            let out_ref = ValueRef::Value {
                id: ValueId::new("output"),
            };
            // list input configs
            let input_configs = vec![
                ArithValueIdConfig::Private {
                    id: ValueId::new("input/a"),
                    ty: CrtValueType::U32,
                    value: Some(ArithValue::from(10)),
                },
                ArithValueIdConfig::Private {
                    id: ValueId::new("input/b"),
                    ty: CrtValueType::U32,
                    value: None,
                },
                ArithValueIdConfig::Private {
                    id: ValueId::new("input/c"),
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
                        &[a_ref, b_ref, c_ref],
                        &[out_ref.clone()],
                        &mut generator_channel,
                    )
                    .await
                    .unwrap();
                println!("generator: decode start");
                generator
                    .decode(&[out_ref], &mut generator_channel)
                    .await
                    .unwrap();
                println!("generator: decode done");
            }
        };

        let evaluator_fut = {
            println!("[EV]-----------start evaluator--------------");
            // let (_sink, mut stream) = evaluator_channel.split();
            let a_ref = ValueRef::Value {
                id: ValueId::new("input/a"),
            };
            let b_ref = ValueRef::Value {
                id: ValueId::new("input/b"),
            };
            let c_ref = ValueRef::Value {
                id: ValueId::new("input/c"),
            };
            let out_ref = ValueRef::Value {
                id: ValueId::new("output"),
            };

            // prepare input configs for a, b
            let input_configs = vec![
                ArithValueIdConfig::Private {
                    id: ValueId::new("input/a"),
                    ty: CrtValueType::U32,
                    value: None,
                },
                ArithValueIdConfig::Private {
                    id: ValueId::new("input/b"),
                    ty: CrtValueType::U32,
                    value: Some(ArithValue::from(31)),
                },
                ArithValueIdConfig::Private {
                    id: ValueId::new("input/c"),
                    ty: CrtValueType::U32,
                    value: Some(ArithValue::from(320)),
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
                        &[a_ref, b_ref, c_ref],
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

        assert_eq!(evaluator_output.unwrap(), vec![ArithValue::from(10)]);
    }
}
