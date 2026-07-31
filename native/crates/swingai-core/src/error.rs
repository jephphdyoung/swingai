use std::fmt;

/// One thing wrong with a value, and where it was found.
///
/// `path` is a dotted/indexed location within the document being checked, e.g.
/// `streams[1].last_timestamp_ns`. It is what makes the CLI output actionable,
/// so callers should keep it specific.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    path: String,
    message: String,
}

impl ValidationError {
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    /// Prefix the location, for reporting an inner value's error from an outer
    /// context: `at("streams[0]")` turns `frame_count` into
    /// `streams[0].frame_count`.
    #[must_use]
    pub fn at(mut self, prefix: &str) -> Self {
        if prefix.is_empty() {
            return self;
        }
        self.path = if self.path.is_empty() {
            prefix.to_owned()
        } else {
            format!("{prefix}.{}", self.path)
        };
        self
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            f.write_str(&self.message)
        } else {
            write!(f, "{}: {}", self.path, self.message)
        }
    }
}

impl std::error::Error for ValidationError {}

/// Every problem found in one document.
///
/// Validation collects rather than short-circuits: fixing a manifest is much
/// faster when you can see all of what is wrong with it at once.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ValidationErrors(Vec<ValidationError>);

impl ValidationErrors {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, path: impl Into<String>, message: impl Into<String>) {
        self.0.push(ValidationError::new(path, message));
    }

    /// Absorb errors from an inner value that shares this one's path.
    pub fn extend(&mut self, inner: ValidationErrors) {
        self.0.extend(inner.0);
    }

    /// Absorb errors from an inner value, prefixing each with `prefix`. An empty
    /// prefix leaves the paths alone.
    pub fn extend_at(&mut self, prefix: &str, inner: ValidationErrors) {
        self.0
            .extend(inner.0.into_iter().map(|error| error.at(prefix)));
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ValidationError> {
        self.0.iter()
    }

    /// `Ok(())` when nothing was found, so callers can use `?`.
    pub fn into_result(self) -> Result<(), ValidationErrors> {
        if self.is_empty() { Ok(()) } else { Err(self) }
    }
}

impl<'a> IntoIterator for &'a ValidationErrors {
    type Item = &'a ValidationError;
    type IntoIter = std::slice::Iter<'a, ValidationError>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl IntoIterator for ValidationErrors {
    type Item = ValidationError;
    type IntoIter = std::vec::IntoIter<ValidationError>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromIterator<ValidationError> for ValidationErrors {
    fn from_iter<T: IntoIterator<Item = ValidationError>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl fmt::Display for ValidationErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, error) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{error}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationErrors {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_ok() {
        assert!(ValidationErrors::new().into_result().is_ok());
    }

    #[test]
    fn prefixing_builds_a_path() {
        let error = ValidationError::new("frame_count", "must be at least 1").at("streams[0]");
        assert_eq!(error.path(), "streams[0].frame_count");
        assert_eq!(
            error.to_string(),
            "streams[0].frame_count: must be at least 1"
        );
    }

    #[test]
    fn prefixing_a_rootless_error_keeps_just_the_prefix() {
        let error = ValidationError::new("", "unusable").at("streams[0]");
        assert_eq!(error.to_string(), "streams[0]: unusable");
    }

    #[test]
    fn collected_errors_display_one_per_line() {
        let mut errors = ValidationErrors::new();
        errors.push("a", "first");
        errors.push("b", "second");
        assert_eq!(errors.len(), 2);
        assert_eq!(errors.to_string(), "a: first\nb: second");
    }
}
