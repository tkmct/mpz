//! BMR16 generator implementation

use futures::{Sink, SinkExt};
use std::{
    borrow::BorrowMut,
    collections::HashSet,
    ops::DerefMut,
    sync::{Arc, Mutex},
};
use utils_aio::non_blocking_backend::{Backend, NonBlockingBackend};

use mpz_circuits::{
    arithmetic::{
        types::{ArithValue, CrtValueType},
        utils::{convert_value_to_crt, PRIMES},
    },
    ArithmeticCircuit,
};
use mpz_core::aes::{FixedKeyAes, FIXED_KEY_AES};
use mpz_garble_core::{
    bmr16::generator::{BMR16Generator as GeneratorCore, GeneratorError as BMR16GeneratorError},
    encoding::{crt_encoding_state as encoding_state, ChaChaCrtEncoder, EncodedCrtValue},
    msg::GarbleMessage,
    ValueError,
};

use crate::{
    bmr16::{config::ArithValueIdConfig, ot::OTSendCrtEncoding, registry::CrtEncodingRegistry},
    value::{ValueId, ValueRef},
};

/// Errors that can occur while performing the role of a generator
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum GeneratorError {
    #[error(transparent)]
    CoreError(#[from] BMR16GeneratorError),
    // TODO: Fix the size of this error
    #[error(transparent)]
    OTError(Box<mpz_ot::OTError>),
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error(transparent)]
    ValueError(#[from] ValueError),
    #[error("missing encoding for value")]
    MissingEncoding(ValueRef),
    #[error(transparent)]
    MemoryError(#[from] crate::MemoryError),
}

impl From<mpz_ot::OTError> for GeneratorError {
    fn from(err: mpz_ot::OTError) -> Self {
        Self::OTError(Box::new(err))
    }
}

/// Config for BMR16Generator
pub struct BMR16GeneratorConfig {
    /// Commitments to encodings
    pub encoding_commitments: bool,
    /// Batch size of encrypted gates streamed
    pub batch_size: usize,
    /// Maximum number of wires used in a circuit for single value
    pub num_wires: usize,
    /// list of modulus numbers used in the circuit.
    pub moduli_list: Vec<u16>,
}

impl BMR16GeneratorConfig {
    /// Create new config instance.
    /// Moduli used in the circuit is implied by num_wires
    pub fn new(encoding_commitments: bool, batch_size: usize, num_wires: usize) -> Self {
        Self::new_with_moduli_list(
            encoding_commitments,
            batch_size,
            num_wires,
            PRIMES[0..num_wires].into(),
        )
    }

    /// Create new config instance with moduli_list.
    pub fn new_with_moduli_list(
        encoding_commitments: bool,
        batch_size: usize,
        num_wires: usize,
        moduli_list: Vec<u16>,
    ) -> Self {
        Self {
            encoding_commitments,
            batch_size,
            num_wires,
            moduli_list,
        }
    }
}

/// BMR16 generator struct
pub struct BMR16Generator {
    config: BMR16GeneratorConfig,
    state: Mutex<State>,
    cipher: &'static FixedKeyAes,
}

impl BMR16Generator {
    /// create new generator instance
    pub fn new(config: BMR16GeneratorConfig, encoder_seed: [u8; 32]) -> Self {
        Self {
            state: Mutex::new(State::new(ChaChaCrtEncoder::new(
                encoder_seed,
                &config.moduli_list,
            ))),
            config,
            cipher: &(FIXED_KEY_AES),
        }
    }

    /// Convenience method for grabbing a lock to the state.
    fn state(&self) -> impl DerefMut<Target = State> + '_ {
        self.state.lock().unwrap()
    }

    /// setup inputs by sending wires to evaluator
    pub async fn setup_inputs<
        S: Sink<GarbleMessage, Error = std::io::Error> + Unpin,
        OT: OTSendCrtEncoding,
    >(
        &self,
        id: &str,
        input_configs: &[ArithValueIdConfig],
        sink: &mut S,
        ot: &OT,
    ) -> Result<(), GeneratorError> {
        // println!("[GEN] setup_inputs() start");
        let mut ot_send_values = Vec::new();
        let mut direct_send_values = Vec::new();
        for config in input_configs.iter().cloned() {
            match config {
                ArithValueIdConfig::Public { id, value, .. } => {
                    direct_send_values.push((id, value));
                }
                ArithValueIdConfig::Private { id, value, ty } => {
                    if let Some(value) = value {
                        direct_send_values.push((id, value));
                    } else {
                        ot_send_values.push((id, ty));
                    }
                }
            }
        }

        futures::try_join!(
            self.ot_send_active_encodings(id, &ot_send_values, ot),
            self.direct_send_active_encodings(&direct_send_values, sink)
        )?;

        // println!("[GEN] setup_inputs() done");
        Ok(())
    }

    async fn ot_send_active_encodings<OT: OTSendCrtEncoding>(
        &self,
        id: &str,
        values: &[(ValueId, CrtValueType)],
        ot: &OT,
    ) -> Result<(), GeneratorError> {
        // println!("[GEN] ot_send_active_encodings() start");
        if values.is_empty() {
            return Ok(());
        }

        let mut state = self.state();
        let full_encodings = {
            // Filter out any values that are already active, setting them active otherwise.
            let mut values = values
                .iter()
                .filter(|(id, _)| state.active.insert(id.clone()))
                .collect::<Vec<_>>();
            values.sort_by_key(|(id, _)| id.clone());

            // create a encoding for sending but do not save encodings
            let encodings = values
                .iter()
                .map(|(id, ty)| state.encoder.encode_by_type(id.to_u64(), ty.clone()))
                .collect::<Vec<_>>();
            encodings
        };

        let encoder = state.encoder.borrow_mut();
        let deltas = encoder.deltas();
        let mut rng = encoder.get_rng(0);

        // println!("[GEN] ot.send_arith() start");
        let bases = ot.send_arith(id, full_encodings, deltas, &mut rng).await?;
        // println!("[GEN] ot.send_arith() done");
        for (labels, (v_id, _)) in bases.into_iter().zip(values) {
            let encoding = EncodedCrtValue::<encoding_state::Full>::from(labels);
            let _ = state.encoding_registry.set_encoding_by_id(v_id, encoding);
        }
        // println!("[GEN] ot_send_active_encodings() done");

        Ok(())
    }

    /// Directly sends the active encodings of the provided values to the evaluator.
    ///
    /// # Arguments
    ///
    /// - `values` - The values to send
    /// - `sink` - The sink to send the encodings to the evaluator
    async fn direct_send_active_encodings<
        S: Sink<GarbleMessage, Error = std::io::Error> + Unpin,
    >(
        &self,
        values: &[(ValueId, ArithValue)],
        sink: &mut S,
    ) -> Result<(), GeneratorError> {
        // println!("[GEN] direct_send_active_encodings() start");
        if values.is_empty() {
            return Ok(());
        }

        let active_encodings = {
            let mut state = self.state();
            // Filter out any values that are already active, setting them active otherwise.
            let mut values = values
                .iter()
                .filter(|(id, _)| state.active.insert(id.clone()))
                .collect::<Vec<_>>();
            values.sort_by_key(|(id, _)| id.clone());

            values
                .iter()
                .map(|(id, value)| {
                    let crt_vec = convert_value_to_crt(*value);
                    let full_encoding = state.encode_by_id(id, &value.value_type())?;

                    Ok(full_encoding.select(state.encoder.deltas(), crt_vec))
                })
                .collect::<Result<Vec<_>, GeneratorError>>()?
        };

        // println!("[GEN] sink.send(ActiveCrtValues) start");
        sink.send(GarbleMessage::ActiveCrtValues(active_encodings))
            .await?;
        // println!("[GEN] sink.send(ActiveCrtValues) done");
        // println!("[GEN] direct_send_active_encodings() done");

        Ok(())
    }

    /// generate a garbled circuit,
    pub async fn generate<
        S: Sink<GarbleMessage, Error = std::io::Error> + Unpin + std::fmt::Debug,
    >(
        &self,
        circ: Arc<ArithmeticCircuit>,
        inputs: &[ValueRef],
        outputs: &[ValueRef],
        sink: &mut S,
    ) -> Result<Vec<EncodedCrtValue<encoding_state::Full>>, GeneratorError> {
        // println!("[GEN] generate() start");
        // get encodings from inputs
        let mut state = self.state();
        let inputs = inputs
            .iter()
            .map(|value| {
                // println!("name: {:?}", value);
                state
                    .encoding_registry
                    .get_encoding(value)
                    .ok_or(GeneratorError::MissingEncoding(value.clone()))
            })
            .collect::<Result<Vec<EncodedCrtValue<encoding_state::Full>>, _>>()?;

        let mut gen = GeneratorCore::new(circ, state.encoder.deltas().clone(), inputs)?;

        let mut batch: Vec<_>;
        let batch_size = self.config.batch_size;
        let mut block_count = 0;

        sink.send(GarbleMessage::ArithEncryptedGates(vec![]))
            .await
            .unwrap();
        while !gen.is_complete() {
            // println!("[GEN] generating the encrypted gates");
            // Move the generator to another thread to produce the next batch
            // then send it back
            (gen, batch) = Backend::spawn(move || {
                let batch = gen.by_ref().take(batch_size).collect::<Vec<_>>();
                (gen, batch)
            })
            .await;

            // println!("[GEN] generating finished: {:?}", !batch.is_empty());
            // println!("[GEN] sending the gate");
            // let result = sink.send(GarbleMessage::ArithEncryptedGates(vec![])).await;
            // println!("[GEN] Result: {:?}", result);

            if !batch.is_empty() {
                // println!("[GEN] send the message: ArithEncryptedGates",);
                block_count += batch.len();
                sink.send(GarbleMessage::ArithEncryptedGates(batch))
                    .await
                    .unwrap();

                // println!("[GEN] message sent",);
            }
            // println!("[GEN] gen.is_complete(): {}", gen.is_complete());
        }
        println!("Block count sent to evaluator: {}", block_count);
        println!("Data size in kb: {}", block_count * 128 / 8 / 1024);

        // println!("[GEN] gen.outputs()");
        let encoded_outputs = gen.outputs()?;

        // Add the outputs to the encoding registry and set as active.
        // println!("[GEN] set_encoding()");
        for (output, encoding) in outputs.iter().zip(encoded_outputs.iter()) {
            state
                .encoding_registry
                .set_encoding(output, encoding.clone())?;
            output.iter().for_each(|id| {
                state.active.insert(id.clone());
            });
        }

        // println!("[GEN] generate() done");
        Ok(encoded_outputs)
    }

    /// Send value decoding information to the evaluator.
    ///
    /// # Arguments
    ///
    /// * `values` - The values to decode
    /// * `sink` - The sink to send the decodings with
    pub async fn decode<S: Sink<GarbleMessage, Error = std::io::Error> + Unpin>(
        &self,
        values: &[ValueRef],
        sink: &mut S,
    ) -> Result<(), GeneratorError> {
        // println!("[GEN] decode() start");
        let state = self.state();
        let decodings = {
            values
                .iter()
                .enumerate()
                .map(|(idx, value)| {
                    state
                        .encoding_registry
                        .get_encoding(value)
                        .ok_or(GeneratorError::MissingEncoding(value.clone()))
                        .map(|encoding| encoding.decoding(idx, state.encoder.deltas(), self.cipher))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        sink.send(GarbleMessage::CrtValueDecodings(decodings))
            .await?;
        // println!("[GEN] decode() done");

        Ok(())
    }
}

struct State {
    // number of wire label
    encoder: ChaChaCrtEncoder,
    /// Encodings of values
    encoding_registry: CrtEncodingRegistry<encoding_state::Full>,
    /// The set of values that are currently active.
    ///
    /// A value is considered active when it has been encoded and sent to the evaluator.
    ///
    /// This is used to guarantee that the same encoding is never used
    /// with different active values.
    active: HashSet<ValueId>,
}

impl State {
    fn new(encoder: ChaChaCrtEncoder) -> Self {
        Self {
            encoder,
            encoding_registry: CrtEncodingRegistry::default(),
            active: HashSet::new(),
        }
    }

    #[allow(dead_code)]
    fn encode(
        &mut self,
        value: &ValueRef,
        ty: &CrtValueType,
    ) -> Result<EncodedCrtValue<encoding_state::Full>, GeneratorError> {
        match (value, ty) {
            (ValueRef::Value { id }, ty) => self.encode_by_id(id, ty),
            (ValueRef::Array(_), _) => {
                panic!("array type not supported")
            }
        }
    }

    fn encode_by_id(
        &mut self,
        id: &ValueId,
        ty: &CrtValueType,
    ) -> Result<EncodedCrtValue<encoding_state::Full>, GeneratorError> {
        let encoding = self.encoder.encode_by_type(id.to_u64(), ty.clone());

        // Returns error if the encoding already exists
        self.encoding_registry
            .set_encoding_by_id(id, encoding.clone())?;

        Ok(encoding)
    }
}
