//! BMR16 registry module
use std::collections::HashMap;

use mpz_circuits::arithmetic::types::CrtValueType;
use mpz_core::value::{ValueId, ValueRef};
use mpz_garble_core::encoding::{crt_encoding_state::LabelState as CrtLabelState, EncodedCrtValue};

use crate::registry::{EncodingId, EncodingRegistryError};
use crate::MemoryError;

/// A registry of values.
///
/// This registry is used to track all the values that exist in a session.
///
/// It enforces that a value is only defined once, returning an error otherwise.
#[derive(Debug, Default)]
pub struct CrtValueRegistry {
    /// A map of value IDs to their types.
    values: HashMap<ValueId, CrtValueType>,
    /// A map of value IDs to their references.
    refs: HashMap<String, ValueRef>,
}

impl CrtValueRegistry {
    /// Adds a value to the registry.
    pub fn add_value(&mut self, id: &str, ty: CrtValueType) -> Result<ValueRef, MemoryError> {
        self.add_value_with_offset(id, ty, 0)
    }

    /// Adds a value to the registry, applying an offset to the ids of the elements if the
    /// value is an array.
    pub fn add_value_with_offset(
        &mut self,
        id: &str,
        ty: CrtValueType,
        _offset: usize,
    ) -> Result<ValueRef, MemoryError> {
        let value_ref = {
            let id = ValueId::new(id);
            self.add_value_id(id.clone(), ty)?;
            ValueRef::Value { id }
        };

        self.refs.insert(id.to_string(), value_ref.clone());

        Ok(value_ref)
    }

    fn add_value_id(&mut self, id: ValueId, ty: CrtValueType) -> Result<(), MemoryError> {
        // Ensure that the value is not already defined.
        if self.values.contains_key(&id) {
            return Err(MemoryError::DuplicateValueId(id));
        }

        self.values.insert(id, ty);

        Ok(())
    }
}

/// A registry of encodings.
///
/// This registry is used to store encodings for values.
///
/// It enforces that an encoding for a value is only set once.
#[derive(Debug)]
pub(crate) struct CrtEncodingRegistry<T>
where
    T: CrtLabelState,
{
    encodings: HashMap<EncodingId, EncodedCrtValue<T>>,
}

impl<T> Default for CrtEncodingRegistry<T>
where
    T: CrtLabelState,
{
    fn default() -> Self {
        Self {
            encodings: HashMap::new(),
        }
    }
}

impl<T> CrtEncodingRegistry<T>
where
    T: CrtLabelState,
{
    /// Set the encoding for a value id.
    pub(crate) fn set_encoding_by_id(
        &mut self,
        id: &ValueId,
        encoding: EncodedCrtValue<T>,
    ) -> Result<(), EncodingRegistryError> {
        let encoding_id = EncodingId::new(id.to_u64());
        if self.encodings.contains_key(&encoding_id) {
            return Err(EncodingRegistryError::DuplicateId(id.clone()));
        }

        self.encodings.insert(encoding_id, encoding);

        Ok(())
    }

    /// Set the encoding for a value.
    ///
    /// # Panics
    ///
    /// Panics if the encoding for the value has already been set, or if the value
    /// type does not match the encoding type.
    pub(crate) fn set_encoding(
        &mut self,
        value: &ValueRef,
        encoding: EncodedCrtValue<T>,
    ) -> Result<(), EncodingRegistryError> {
        match (value, encoding) {
            (ValueRef::Value { id }, encoding) => self.set_encoding_by_id(id, encoding)?,
            (ValueRef::Array(..), _) => panic!("not supported"),
        }

        Ok(())
    }

    /// Get the encoding for a value if it exists.
    ///
    /// # Panics
    ///
    /// Panics if the value is an array and if the type of its elements are not consistent.
    pub(crate) fn get_encoding(&self, value: &ValueRef) -> Option<EncodedCrtValue<T>> {
        match value {
            ValueRef::Value { id, .. } => self.encodings.get(&id.to_u64().into()).cloned(),
            ValueRef::Array(..) => {
                panic!("not supported")
            }
        }
    }

    /// Returns whether an encoding is present for a value id.
    pub(crate) fn contains(&self, id: &ValueId) -> bool {
        self.encodings.contains_key(&id.to_u64().into())
    }
}
