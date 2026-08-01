use std::fmt;

use swingai_core::ValidationErrors;

/// Why a contract document could not be accepted.
///
/// The split matters for reporting: a [`Parse`](Self::Parse) failure means the
/// document is not shaped like the contract at all (wrong type, missing required
/// field, unsupported schema version), while [`Invalid`](Self::Invalid) means it
/// parsed but says something impossible.
#[derive(Debug)]
pub enum ContractError {
    Parse(serde_json::Error),
    Invalid(ValidationErrors),
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(f, "could not parse: {error}"),
            Self::Invalid(errors) => {
                let count = errors.len();
                let noun = if count == 1 { "problem" } else { "problems" };
                write!(f, "{count} {noun}:\n{errors}")
            }
        }
    }
}

impl std::error::Error for ContractError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::Invalid(errors) => Some(errors),
        }
    }
}

impl From<serde_json::Error> for ContractError {
    fn from(error: serde_json::Error) -> Self {
        Self::Parse(error)
    }
}

impl From<ValidationErrors> for ContractError {
    fn from(errors: ValidationErrors) -> Self {
        Self::Invalid(errors)
    }
}
