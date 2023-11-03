//! BMR16 generator implementation

use std::{
    collections::HashSet,
    ops::DerefMut,
    sync::{Arc, Mutex},
};

use futures::{Sink, SinkExt};
use mpz_circuits::{
    arithmetic::{
        types::{ArithValue, CrtRepr, CrtValue, CrtValueType},
        utils::{convert_value_to_crt, convert_values_to_crts},
    },
    types::{Value, ValueType},
    ArithmeticCircuit,
};
use mpz_core::{
    aes::{FixedKeyAes, FIXED_KEY_AES},
    hash::Hash,
    value::{ValueId, ValueRef},
};
use mpz_garble_core::{
    bmr16::generator::{BMR16Generator as GeneratorCore, GeneratorError as BMR16GeneratorError},
    encoding::{crt_encoding_state as encoding_state, ChaChaCrtEncoder, EncodedCrtValue},
    msg::GarbleMessage,
    Encoder,
};

use utils_aio::non_blocking_backend::{Backend, NonBlockingBackend};

use crate::bmr16::registry::CrtEncodingRegistry;
use crate::{config::ValueIdConfig, ot::OTSendEncoding, registry::EncodingRegistry};

use mpz_garble_core::ValueError;

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
    EncodingRegistryError(#[from] crate::registry::EncodingRegistryError),
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
}

/// BMR16 generator struct
pub struct BMR16Generator<const N: usize> {
    config: BMR16GeneratorConfig,
    state: Mutex<State<N>>,
    cipher: &'static FixedKeyAes,
}

impl<const N: usize> BMR16Generator<N> {
    /// create new generator instance
    pub fn new(config: BMR16GeneratorConfig, encoder_seed: [u8; 32]) -> Self {
        Self {
            state: Mutex::new(State::new(ChaChaCrtEncoder::<N>::new(
                encoder_seed,
                config.num_wires,
            ))),
            config,
            cipher: &(FIXED_KEY_AES),
        }
    }

    /// Convenience method for grabbing a lock to the state.
    fn state(&self) -> impl DerefMut<Target = State<N>> + '_ {
        self.state.lock().unwrap()
    }

    /// Returns the seed used to generate encodings.
    pub(crate) fn seed(&self) -> Vec<u8> {
        self.state().encoder.seed()
    }

    /// Returns the encoding for a value.
    pub fn get_encoding(&self, value: &ValueRef) -> Option<EncodedCrtValue<encoding_state::Full>> {
        self.state().encoding_registry.get_encoding(value)
    }

    pub(crate) fn get_encodings_by_id(
        &self,
        ids: &[ValueId],
    ) -> Option<Vec<EncodedCrtValue<encoding_state::Full>>> {
        let state = self.state();

        ids.iter()
            .map(|id| state.encoding_registry.get_encoding_by_id(id))
            .collect::<Option<Vec<_>>>()
    }

    /// Generate encodings for a slice of values
    pub(crate) fn generate_encodings(
        &self,
        values: &[(ValueId, CrtValueType)],
    ) -> Result<(), GeneratorError> {
        let mut state = self.state();

        for (id, ty) in values {
            _ = state.encode_by_id(id, ty)?;
        }

        Ok(())
    }

    /// setup inputs by sending wires to evaluator
    pub async fn setup_inputs() -> Result<(), GeneratorError> {
        todo!()
    }

    async fn ot_send_active_encodings<OT: OTSendEncoding>(
        &self,
        id: &str,
        values: &[(ValueId, CrtValueType)],
        ot: &OT,
    ) -> Result<(), GeneratorError> {
        todo!()
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
                    let crt_vec = convert_value_to_crt(value.clone());
                    let full_encoding = state.encode_by_id(id, &value.value_type())?;

                    Ok(full_encoding.select(self.state().encoder.deltas(), crt_vec))
                })
                .collect::<Result<Vec<_>, GeneratorError>>()?
        };

        sink.send(GarbleMessage::ActiveCrtValues(active_encodings))
            .await?;

        Ok(())
    }

    /// generate a garbled circuit,
    pub async fn generate<S: Sink<GarbleMessage, Error = std::io::Error> + Unpin>(
        &self,
        circ: Arc<ArithmeticCircuit>,
        inputs: &[ValueRef],
        outputs: &[ValueRef],
        sink: &mut S,
    ) -> Result<Vec<EncodedCrtValue<encoding_state::Full>>, GeneratorError> {
        // get encodings from inputs
        let state = self.state();
        let inputs = inputs
            .iter()
            .map(|value| {
                state
                    .encoding_registry
                    .get_encoding(value)
                    .ok_or(GeneratorError::MissingEncoding(value.clone()))
            })
            .collect::<Result<Vec<EncodedCrtValue<encoding_state::Full>>, _>>()?;

        let mut gen = GeneratorCore::<N>::new(circ, state.encoder.deltas().clone(), inputs)?;

        let mut batch: Vec<_>;
        let batch_size = self.config.batch_size;

        while !gen.is_complete() {
            // Move the generator to another thread to produce the next batch
            // then send it back
            (gen, batch) = Backend::spawn(move || {
                let batch = gen.by_ref().take(batch_size).collect();
                (gen, batch)
            })
            .await;

            if !batch.is_empty() {
                sink.send(GarbleMessage::ArithEncryptedGates(batch)).await?;
            }
        }

        let encoded_outputs = gen.outputs()?;

        // TODO: implement commitment
        // if self.config.encoding_commitments {
        //     let commitments = encoded_outputs
        //         .iter()
        //         .map(|output| output.commit())
        //         .collect();
        //
        //     sink.send(GarbleMessage::EncodingCommitments(commitments))
        //         .await?;
        // }

        // Add the outputs to the encoding registry and set as active.
        let mut state = self.state();
        for (output, encoding) in outputs.iter().zip(encoded_outputs.iter()) {
            state
                .encoding_registry
                .set_encoding(output, encoding.clone())?;
            output.iter().for_each(|id| {
                state.active.insert(id.clone());
            });
        }

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
        let decodings = {
            let state = self.state();
            values
                .iter()
                .enumerate()
                .map(|(idx, value)| {
                    state
                        .encoding_registry
                        .get_encoding(value)
                        .ok_or(GeneratorError::MissingEncoding(value.clone()))
                        .map(|encoding| {
                            encoding.decoding(idx, self.state().encoder.deltas(), self.cipher)
                        })
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        sink.send(GarbleMessage::CrtValueDecodings(decodings))
            .await?;

        Ok(())
    }
}

struct State<const N: usize> {
    // number of wire label
    encoder: ChaChaCrtEncoder<N>,
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

impl<const N: usize> State<N> {
    fn new(encoder: ChaChaCrtEncoder<N>) -> Self {
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
            _ => panic!("invalid value and type combination: {:?} {:?}", value, ty),
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
