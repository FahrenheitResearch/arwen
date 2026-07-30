use crate::error::ContourError;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExtendMode {
    #[default]
    None,
    Below,
    Above,
    Both,
}

impl ExtendMode {
    pub const fn includes_below(self) -> bool {
        matches!(self, Self::Below | Self::Both)
    }

    pub const fn includes_above(self) -> bool {
        matches!(self, Self::Above | Self::Both)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ContourLevels {
    values: Vec<f64>,
}

impl ContourLevels {
    pub fn new(values: Vec<f64>) -> Result<Self, ContourError> {
        if values.is_empty() {
            return Err(ContourError::EmptyContourLevels);
        }
        validate_strictly_ascending(&values)?;
        Ok(Self { values })
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn bins(&self, extend: ExtendMode) -> Result<LevelBins, ContourError> {
        LevelBins::with_extend(self.values.clone(), extend)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LevelBound {
    NegInfinity,
    Finite(f64),
    PosInfinity,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelBin {
    pub index: usize,
    pub lower: LevelBound,
    pub upper: LevelBound,
    pub upper_inclusive: bool,
}

impl LevelBin {
    pub fn contains(&self, value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }

        let lower_ok = match self.lower {
            LevelBound::NegInfinity => true,
            LevelBound::Finite(bound) => value >= bound,
            LevelBound::PosInfinity => false,
        };

        let upper_ok = match self.upper {
            LevelBound::NegInfinity => false,
            LevelBound::Finite(bound) => {
                if self.upper_inclusive {
                    value <= bound
                } else {
                    value < bound
                }
            }
            LevelBound::PosInfinity => true,
        };

        lower_ok && upper_ok
    }

    pub fn lower_value(self) -> Option<f64> {
        match self.lower {
            LevelBound::Finite(value) => Some(value),
            LevelBound::NegInfinity | LevelBound::PosInfinity => None,
        }
    }

    pub fn upper_value(self) -> Option<f64> {
        match self.upper {
            LevelBound::Finite(value) => Some(value),
            LevelBound::NegInfinity | LevelBound::PosInfinity => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LevelBins {
    thresholds: Vec<f64>,
    extend: ExtendMode,
    bins: Vec<LevelBin>,
}

impl LevelBins {
    pub fn bounded(thresholds: Vec<f64>) -> Result<Self, ContourError> {
        Self::with_extend(thresholds, ExtendMode::None)
    }

    pub fn with_extend(thresholds: Vec<f64>, extend: ExtendMode) -> Result<Self, ContourError> {
        if thresholds.len() < 2 {
            return Err(ContourError::NeedAtLeastTwoThresholds);
        }
        validate_strictly_ascending(&thresholds)?;

        let mut bins = Vec::with_capacity(
            thresholds.len() - 1
                + usize::from(extend.includes_below())
                + usize::from(extend.includes_above()),
        );

        if extend.includes_below() {
            bins.push(LevelBin {
                index: 0,
                lower: LevelBound::NegInfinity,
                upper: LevelBound::Finite(thresholds[0]),
                upper_inclusive: false,
            });
        }

        for pair_index in 0..thresholds.len() - 1 {
            let is_last_finite_bin =
                pair_index + 1 == thresholds.len() - 1 && !extend.includes_above();
            bins.push(LevelBin {
                index: bins.len(),
                lower: LevelBound::Finite(thresholds[pair_index]),
                upper: LevelBound::Finite(thresholds[pair_index + 1]),
                upper_inclusive: is_last_finite_bin,
            });
        }

        if extend.includes_above() {
            bins.push(LevelBin {
                index: bins.len(),
                lower: LevelBound::Finite(*thresholds.last().expect("validated thresholds")),
                upper: LevelBound::PosInfinity,
                upper_inclusive: true,
            });
        }

        Ok(Self {
            thresholds,
            extend,
            bins,
        })
    }

    pub fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }

    pub fn extend_mode(&self) -> ExtendMode {
        self.extend
    }

    pub fn bins(&self) -> &[LevelBin] {
        &self.bins
    }

    pub fn bin(&self, index: usize) -> Option<&LevelBin> {
        self.bins.get(index)
    }

    pub fn bin_index(&self, value: f64) -> Option<usize> {
        if !value.is_finite() {
            return None;
        }

        let first = *self.thresholds.first()?;
        let last = *self.thresholds.last()?;
        let finite_offset = usize::from(self.extend.includes_below());

        if value < first {
            return self.extend.includes_below().then_some(0);
        }

        if value > last {
            return self
                .extend
                .includes_above()
                .then_some(self.bins.len().saturating_sub(1));
        }

        if value == last {
            return if self.extend.includes_above() {
                Some(self.bins.len().saturating_sub(1))
            } else {
                Some(finite_offset + self.thresholds.len().saturating_sub(2))
            };
        }

        let upper_threshold_index = self
            .thresholds
            .partition_point(|threshold| *threshold <= value);
        Some(finite_offset + upper_threshold_index.saturating_sub(1))
    }
}

fn validate_strictly_ascending(levels: &[f64]) -> Result<(), ContourError> {
    for (index, value) in levels.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(ContourError::NonFiniteLevel { index, value });
        }
    }

    for index in 0..levels.len() - 1 {
        let previous = levels[index];
        let current = levels[index + 1];
        if current <= previous {
            return Err(ContourError::LevelsNotStrictlyAscending {
                index,
                previous,
                current,
            });
        }
    }

    Ok(())
}
