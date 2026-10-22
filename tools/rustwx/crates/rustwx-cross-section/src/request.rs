use crate::data::SectionMetadata;
use crate::error::CrossSectionError;
use crate::geo::{SampledPath, SectionPath};

/// Horizontal sampling choice for building a [`SectionLayout`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SamplingStrategy {
    Count(usize),
    SpacingKm(f64),
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::Count(101)
    }
}

/// Public request builder for clients that want path geometry and layout decisions without
/// committing to any specific data backend.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossSectionRequest {
    path: SectionPath,
    sampling: SamplingStrategy,
    field_key: Option<String>,
    metadata: SectionMetadata,
}

impl CrossSectionRequest {
    pub fn new(path: SectionPath) -> Self {
        Self {
            path,
            sampling: SamplingStrategy::default(),
            field_key: None,
            metadata: SectionMetadata::default(),
        }
    }

    pub fn with_sampling(mut self, sampling: SamplingStrategy) -> Self {
        self.sampling = sampling;
        self
    }

    pub fn with_field_key(mut self, field_key: impl Into<String>) -> Self {
        self.field_key = Some(field_key.into());
        self
    }

    pub fn with_metadata(mut self, metadata: SectionMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn build_layout(&self) -> Result<SectionLayout, CrossSectionError> {
        let sampled_path = match self.sampling {
            SamplingStrategy::Count(count) => self.path.sample_count(count)?,
            SamplingStrategy::SpacingKm(spacing_km) => self.path.sample_spacing_km(spacing_km)?,
        };

        Ok(SectionLayout {
            sampled_path,
            field_key: self.field_key.clone(),
            metadata: self.metadata.clone(),
        })
    }
}

/// Concrete geometry/layout output derived from a [`CrossSectionRequest`].
#[derive(Debug, Clone, PartialEq)]
pub struct SectionLayout {
    pub sampled_path: SampledPath,
    pub field_key: Option<String>,
    pub metadata: SectionMetadata,
}

impl SectionLayout {
    pub fn field_key(&self) -> Option<&str> {
        self.field_key.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::GeoPoint;

    #[test]
    fn request_layout_uses_sampling_strategy_and_metadata() {
        let path = SectionPath::endpoints(
            GeoPoint::new(39.0, -105.0).unwrap(),
            GeoPoint::new(41.0, -88.0).unwrap(),
        )
        .unwrap();

        let layout = CrossSectionRequest::new(path)
            .with_sampling(SamplingStrategy::Count(7))
            .with_field_key("temperature")
            .with_metadata(SectionMetadata::new().field("temperature", "K"))
            .build_layout()
            .unwrap();

        assert_eq!(layout.sampled_path.len(), 7);
        assert_eq!(layout.field_key(), Some("temperature"));
        assert_eq!(layout.metadata.field_units.as_deref(), Some("K"));
    }
}
