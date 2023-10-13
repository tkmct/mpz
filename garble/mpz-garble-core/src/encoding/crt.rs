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
pub struct Labels<S: LabelState> {
    state: S,
    labels: Vec<Label>,
}

/// Encoded CRT Value.
pub struct EncodedCrtValue<S: LabelState>(Labels<S>);

impl<S: LabelState> EncodedCrtValue<S> {
    /// returns iterator of Labels.
    pub fn iter(&self) -> Box<dyn Iterator<Item = &Label> + '_> {
        Box::new(self.0.labels.iter())
    }

    /// returns length of labels
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.0.labels.len()
    }
}

impl EncodedCrtValue<state::Full> {}
