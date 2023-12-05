//! BMR16 configs
use mpz_circuits::arithmetic::types::{ArithValue, CrtValueType};

use crate::value::{ValueId, ValueRef};

/// configuration of a value
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum ArithValueConfig {
    Public {
        value_ref: ValueRef,
        ty: CrtValueType,
        value: ArithValue,
    },
    Private {
        value_ref: ValueRef,
        ty: CrtValueType,
        value: Option<ArithValue>,
    },
}

impl ArithValueConfig {
    /// Create new private value config.
    pub fn new_private(value_ref: ValueRef, ty: CrtValueType, value: Option<ArithValue>) -> Self {
        Self::Private {
            value_ref,
            ty,
            value,
        }
    }
}

/// configuration of a value
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum ArithValueIdConfig {
    Public {
        id: ValueId,
        ty: CrtValueType,
        value: ArithValue,
    },
    Private {
        id: ValueId,
        ty: CrtValueType,
        value: Option<ArithValue>,
    },
}

impl ArithValueIdConfig {
    /// Get value id
    pub fn id(&self) -> &ValueId {
        match self {
            ArithValueIdConfig::Public { id, .. } => id,
            ArithValueIdConfig::Private { id, .. } => id,
        }
    }
}
