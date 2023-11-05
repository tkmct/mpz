//! BMR16 evaluator module
use mpz_circuits::ArithmeticCircuit;

use derive_builder::Builder;

use futures::{Stream, StreamExt};
use std::{
    collections::{HashMap, HashSet},
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use mpz_circuits::arithmetic::types::{ArithValue, CrtValueType, TypeError};
use mpz_core::{
    hash::Hash,
    value::{ValueId, ValueRef},
};
use mpz_garble_core::{
    bmr16::evaluator::{BMR16Evaluator as EvaluatorCore, EvaluatorError as BMR16EvaluatorError},
    encoding::{crt_encoding_state as encoding_state, CrtDecoding, DecodeError, EncodedCrtValue},
    msg::GarbleMessage,
};

use utils_aio::{
    expect_msg_or_err,
    non_blocking_backend::{Backend, NonBlockingBackend},
};

use crate::{
    bmr16::{config::ArithValueIdConfig, ot::OTReceiveCrtEncoding, registry::CrtEncodingRegistry},
    ot::OTVerifyEncoding,
};

/// Errors that can occur while performing the role of an evaluator
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum EvaluatorError {
    #[error(transparent)]
    CoreError(#[from] BMR16EvaluatorError),
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    // TODO: Fix the size of this error
    #[error(transparent)]
    OTError(#[from] mpz_ot::OTError),
    #[error("incorrect number of values: expected {expected}, got {actual}")]
    IncorrectValueCount { expected: usize, actual: usize },
    #[error(transparent)]
    TypeError(#[from] TypeError),
    #[error(transparent)]
    ValueError(#[from] mpz_garble_core::ValueError),
    #[error(transparent)]
    EncodingRegistryError(#[from] crate::registry::EncodingRegistryError),
    #[error("missing active encoding for value")]
    MissingEncoding(ValueRef),
    #[error("duplicate decoding for value: {0:?}")]
    DuplicateDecoding(ValueId),
    #[error(transparent)]
    DecodeError(#[from] DecodeError),
}

/// Config struct for BMR16 evaluator
#[derive(Debug, Clone, Builder)]
pub struct BMR16EvaluatorConfig {
    /// The number of encrypted gates to evaluate per batch.
    #[builder(default = "1024")]
    pub(crate) batch_size: usize,
}

//// A garbled circuit evaluator.
#[allow(missing_docs)]
pub struct BMR16Evaluator<const N: usize> {
    config: BMR16EvaluatorConfig,
    state: Mutex<State>,
}

impl<const N: usize> BMR16Evaluator<N> {
    /// Create new BMR16 evaluator instance
    pub fn new(config: BMR16EvaluatorConfig) -> Self {
        Self {
            config,
            state: Mutex::new(State::default()),
        }
    }

    /// Convenience method for grabbing a lock to the state.
    fn state(&self) -> impl DerefMut<Target = State> + '_ {
        self.state.lock().unwrap()
    }

    /// Sets a value as decoded.
    ///
    /// # Errors
    ///
    /// Returns an error if the value has already been decoded.
    pub(crate) fn set_decoded(&self, value: &ValueRef) -> Result<(), EvaluatorError> {
        let mut state = self.state();
        // Check that none of the values in this reference have already been decoded.
        // We track every individual value of an array separately to ensure that a decoding
        // is never overwritten.
        for id in value.iter() {
            if !state.decoded_values.insert(id.clone()) {
                return Err(EvaluatorError::DuplicateDecoding(id.clone()));
            }
        }

        Ok(())
    }

    /// Returns the encoding for a value.
    pub fn get_encoding(
        &self,
        value: &ValueRef,
    ) -> Option<EncodedCrtValue<encoding_state::Active>> {
        self.state().encoding_registry.get_encoding(value)
    }

    /// Adds a decoding log entry.
    pub(crate) fn add_decoding_log(&self, value: &ValueRef, decoding: CrtDecoding) {
        self.state().decoding_logs.insert(value.clone(), decoding);
    }

    /// setup inputs for evaluator
    pub async fn setup_inputs<
        S: Stream<Item = Result<GarbleMessage, std::io::Error>> + Unpin,
        OT: OTReceiveCrtEncoding,
    >(
        &self,
        id: &str,
        input_configs: &[ArithValueIdConfig],
        stream: &mut S,
        ot: &OT,
    ) -> Result<(), EvaluatorError> {
        println!("[EV] setup_inputs() start");

        let (ot_recv_values, direct_recv_values) = {
            let state = self.state();

            // Filter out any values that are already active.
            let mut input_configs: Vec<ArithValueIdConfig> = input_configs
                .iter()
                .filter(|config| !state.encoding_registry.contains(config.id()))
                .cloned()
                .collect();

            input_configs.sort_by_key(|config| config.id().clone());

            let mut ot_recv_values = Vec::new();
            let mut direct_recv_values = Vec::new();
            for config in input_configs.into_iter() {
                match config {
                    ArithValueIdConfig::Public { id, ty, .. } => {
                        direct_recv_values.push((id, ty));
                    }
                    ArithValueIdConfig::Private { id, ty, value } => {
                        if let Some(value) = value {
                            ot_recv_values.push((id, value));
                        } else {
                            direct_recv_values.push((id, ty));
                        }
                    }
                }
            }

            (ot_recv_values, direct_recv_values)
        };

        futures::try_join!(
            self.ot_receive_active_encodings(id, &ot_recv_values, ot),
            self.direct_receive_active_encodings(&direct_recv_values, stream)
        )?;

        println!("[EV] setup_inputs() done");
        Ok(())
    }

    async fn ot_receive_active_encodings<OT: OTReceiveCrtEncoding>(
        &self,
        id: &str,
        values: &[(ValueId, ArithValue)],
        ot: &OT,
    ) -> Result<(), EvaluatorError> {
        println!("[EV] ot_receive_active_encodings() start");
        if values.is_empty() {
            return Ok(());
        }

        let (ot_recv_ids, ot_recv_values): (Vec<ValueId>, Vec<ArithValue>) =
            values.iter().cloned().unzip();

        let active_encodings = ot.receive_arith(id, ot_recv_values).await?;

        // Make sure the generator sent the expected number of values.
        // This should be handled by the ot receiver, but we double-check anyways :)
        if active_encodings.len() != values.len() {
            return Err(EvaluatorError::IncorrectValueCount {
                expected: values.len(),
                actual: active_encodings.len(),
            });
        }

        let mut state = self.state();

        // Add the OT log
        state.ot_log.insert(id.to_string(), ot_recv_ids);

        for ((id, value), active_encoding) in values.iter().zip(active_encodings) {
            let num_wire = value.num_wire();
            let expected_ty = value.value_type();
            // Make sure the generator sent the expected type.
            // This is also handled by the ot receiver, but we're paranoid.
            if active_encoding.len() != num_wire {
                return Err(TypeError::InvalidLength {
                    expected: num_wire,
                    actual: active_encoding.len(),
                })?;
            }
            // Add the received values to the encoding registry.
            state
                .encoding_registry
                .set_encoding_by_id(id, active_encoding)?;
            state.received_values.insert(id.clone(), expected_ty);
        }
        println!("[EV] ot_receive_active_encodings() done");

        Ok(())
    }

    async fn direct_receive_active_encodings<
        S: Stream<Item = Result<GarbleMessage, std::io::Error>> + Unpin,
    >(
        &self,
        values: &[(ValueId, CrtValueType)],
        stream: &mut S,
    ) -> Result<(), EvaluatorError> {
        println!("[EV] direct_receive_active_encodings() start");
        if values.is_empty() {
            return Ok(());
        }

        let active_encodings = expect_msg_or_err!(stream, GarbleMessage::ActiveCrtValues)?;

        // Make sure the generator sent the expected number of values.
        if active_encodings.len() != values.len() {
            return Err(EvaluatorError::IncorrectValueCount {
                expected: values.len(),
                actual: active_encodings.len(),
            });
        }

        let mut state = self.state();
        for ((id, expected_ty), active_encoding) in values.iter().zip(active_encodings) {
            // Make sure the generator sent the expected type.
            if active_encoding.len() != expected_ty.len() {
                return Err(TypeError::InvalidLength {
                    expected: expected_ty.len(),
                    actual: active_encoding.len(),
                })?;
            }
            // Add the received values to the encoding registry.
            state
                .encoding_registry
                .set_encoding_by_id(id, active_encoding)?;
            state
                .received_values
                .insert(id.clone(), expected_ty.clone());
        }

        println!("[EV] direct_receive_active_encodings() done");
        Ok(())
    }

    /// Evaluate circuit
    pub async fn evaluate<
        S: Stream<Item = Result<GarbleMessage, std::io::Error>> + Unpin + std::fmt::Debug,
    >(
        &self,
        circ: Arc<ArithmeticCircuit>,
        inputs: &[ValueRef],
        outputs: &[ValueRef],
        stream: &mut S,
    ) -> Result<Vec<EncodedCrtValue<encoding_state::Active>>, EvaluatorError> {
        println!("[EV] evaluate() start");
        let mut state = self.state();
        let encoded_inputs = {
            inputs
                .iter()
                .map(|value_ref| {
                    state
                        .encoding_registry
                        .get_encoding(value_ref)
                        .ok_or_else(|| EvaluatorError::MissingEncoding(value_ref.clone()))
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        let mut ev = EvaluatorCore::<N>::new(circ.clone(), encoded_inputs)?;
        println!("[EV] process encrypted gates {}", !ev.is_complete());
        while !ev.is_complete() {
            println!("[EV] Waiting for incoming message...");
            let encrypted_gates = expect_msg_or_err!(stream, GarbleMessage::ArithEncryptedGates)?;
            println!("[EV] Got encrypted gate message");

            for batch in encrypted_gates.chunks(self.config.batch_size) {
                let batch = batch.to_vec();

                ev = Backend::spawn(move || {
                    ev.evaluate(batch.iter());
                    ev
                })
                .await;
            }
        }

        let encoded_outputs = ev.outputs()?;

        // TODO: check commitment and verify if configured

        // Add the output encodings to the encoding registry.
        for (output, encoding) in outputs.iter().zip(encoded_outputs.iter()) {
            state
                .encoding_registry
                .set_encoding(output, encoding.clone())?;
        }

        println!("[EV] evaluate() done");
        Ok(encoded_outputs)
    }

    /// Decode the output value.
    pub async fn decode<S: Stream<Item = Result<GarbleMessage, std::io::Error>> + Unpin>(
        &self,
        values: &[ValueRef],
        stream: &mut S,
    ) -> Result<Vec<ArithValue>, EvaluatorError> {
        println!("[EV] decode() start");
        let decodings = expect_msg_or_err!(stream, GarbleMessage::CrtValueDecodings)?;

        // Make sure the generator sent the expected number of decodings.
        if decodings.len() != values.len() {
            return Err(EvaluatorError::IncorrectValueCount {
                expected: values.len(),
                actual: decodings.len(),
            });
        }

        for (value, _decoding) in values.iter().zip(decodings.iter()) {
            self.set_decoded(value)?;
            // TODO: later
            // if self.config.log_decodings {
            //     self.add_decoding_log(value, decoding.clone());
            // }
        }

        let active_encodings = values
            .iter()
            .map(|value| {
                self.get_encoding(value)
                    .ok_or_else(|| EvaluatorError::MissingEncoding(value.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let decoded_values = decodings
            .iter()
            .zip(active_encodings.iter())
            .enumerate()
            .map(|(i, (decoding, encoding))| encoding.decode(i, decoding))
            .collect::<Result<Vec<_>, _>>()?;

        println!("[EV] decode() done");
        Ok(decoded_values)
    }

    /// TODO
    pub async fn verify<T: OTVerifyEncoding>(
        &mut self,
        _encoder_seed: [u8; 32],
        _ot: &T,
    ) -> Result<(), EvaluatorError> {
        todo!()
    }
}

#[derive(Debug, Default)]
struct State {
    /// Encodings of values
    encoding_registry: CrtEncodingRegistry<encoding_state::Active>,
    /// Encoded values which were received either directly or via OT
    received_values: HashMap<ValueId, CrtValueType>,
    /// Values which have been decoded
    decoded_values: HashSet<ValueId>,
    /// OT logs
    ot_log: HashMap<String, Vec<ValueId>>,
    /// Garbled circuit logs
    circuit_logs: Vec<BMR16EvaluatorLog>,
    /// Decodings of values received from the generator
    decoding_logs: HashMap<ValueRef, CrtDecoding>,
}

#[derive(Debug)]
pub(crate) struct BMR16EvaluatorLog {
    inputs: Vec<ValueRef>,
    outputs: Vec<ValueRef>,
    circ: Arc<ArithmeticCircuit>,
    hash: Hash,
}

impl BMR16EvaluatorLog {
    pub(crate) fn new(
        inputs: Vec<ValueRef>,
        outputs: Vec<ValueRef>,
        circ: Arc<ArithmeticCircuit>,
        digest: Hash,
    ) -> Self {
        Self {
            inputs,
            outputs,
            circ,
            hash: digest,
        }
    }
}
