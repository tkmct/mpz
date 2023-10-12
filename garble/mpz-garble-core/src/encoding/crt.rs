use crate::encoding::{Delta, Label};

/// Encoding state for CRT representation
pub mod state {
    /// Label state trait
    pub trait LabelState {}

    /// Full state
    pub struct Full {}

    impl LabelState for Full {}

    /// Active state
    pub struct Active {}

    impl LabelState for Active {}
}

use state::*;

/// Set of labels.
/// This struct corresponds to one CrtValue
pub struct Labels<const N: usize, S: LabelState> {
    state: S,
    labels: [Label; N],
}

/// Encoded CRT Value.
pub struct EncodedCrtValue<const N: usize, S: LabelState>(Labels<N, S>);

impl<const N: usize, S: LabelState> EncodedCrtValue<N, S> {
    /// return iterator of Labels.
    pub fn iter(&self) -> Box<dyn Iterator<Item = &Label> + '_> {
        Box::new(self.0.labels.iter())
    }
}

impl<const N: usize> EncodedCrtValue<N, state::Full> {}
