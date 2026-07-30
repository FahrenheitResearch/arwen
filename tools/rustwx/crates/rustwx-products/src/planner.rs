//! Typed product planner.
//!
//! Turns a set of requested products into a deduplicated `ExecutionPlan`
//! over typed bundle requirements. The planner is the single runtime
//! truth for *what fetches need to happen for this run*; lane batches
//! (`direct`, `derived`, `severe`, `ecape`, `windowed`) and the HRRR
//! bundled wrappers all build a plan, run the planner-backed loader to
//! materialize bundles, then dispatch to compute/render kernels.
//!
//! Identity model:
//! - `BundleRequirement` is what a product asks for ("I need a surface
//!   bundle at forecast hour F, optionally with this native override").
//! - `CanonicalBundleId` is the decoded-bundle identity that lets the
//!   planner dedupe across products. Two requirements collapse to one
//!   `CanonicalBundleId` when they resolve to the same
//!   `(model, cycle, forecast_hour, source, bundle, native_product)`.
//! - `BundleFetchKey` is the *physical fetch* identity — the unique GRIB
//!   file fetched. Multiple `CanonicalBundleId`s can share a fetch key
//!   when one file feeds both surface and pressure decodes (GFS/ECMWF/
//!   RRFS-A all serve both lanes from the same file).
//! - `PlannedFamilyAlias` records the logical family name that asked for
//!   the bundle (e.g., "nat") so manifests can publish both the
//!   canonical fetch family and the alias slugs that merged into it.

use rustwx_core::{
    BundleRequirement, CanonicalBundleDescriptor, CanonicalBundleId, CycleSpec, ModelId, SourceId,
};
use rustwx_models::{
    LatestRun, ResolvedCanonicalBundleProduct, resolve_canonical_bundle_id,
    resolve_canonical_bundle_product,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Logical alias attached to a planned bundle. Captures the original
/// `BundleRequirement` plus any explicit family slug surfaced through
/// the planner. Used by manifest builders to keep `planned_family_aliases`
/// honest when the planner reroutes a logical family onto a different
/// canonical bundle (HRRR `nat -> sfc`).
#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub struct PlannedFamilyAlias {
    pub bundle: CanonicalBundleDescriptor,
    pub native_override: Option<String>,
    pub logical_family: Option<String>,
    pub variable_patterns: Vec<String>,
}

impl PlannedFamilyAlias {
    pub fn from_requirement(requirement: &BundleRequirement) -> Self {
        Self {
            bundle: requirement.bundle,
            native_override: requirement.native_override.clone(),
            logical_family: None,
            variable_patterns: Vec::new(),
        }
    }

    pub fn with_logical_family<S: Into<String>>(mut self, family: S) -> Self {
        self.logical_family = Some(family.into());
        self
    }

    pub fn with_variable_patterns<I, S>(mut self, patterns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for pattern in patterns {
            let pattern = pattern.into();
            if !self.variable_patterns.contains(&pattern) {
                self.variable_patterns.push(pattern);
            }
        }
        self.variable_patterns.sort();
        self
    }
}

/// Coarser identity than `CanonicalBundleId`: two bundles share a fetch
/// key when the planner can satisfy both decodes from the same fetched
/// GRIB bytes (e.g., GFS surface + pressure both pull `pgrb2.0p25`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct BundleFetchKey {
    pub model: ModelId,
    pub cycle: CycleSpec,
    pub forecast_hour: u16,
    pub source: SourceId,
    pub native_product: String,
}

impl BundleFetchKey {
    pub fn from_id(id: &CanonicalBundleId) -> Self {
        Self {
            model: id.model,
            cycle: id.cycle.clone(),
            forecast_hour: id.forecast_hour,
            source: id.source,
            native_product: id.native_product.clone(),
        }
    }
}

/// One node in the execution plan: a unique decoded bundle we plan to
/// load. Carries the resolved native product and every alias that asked
/// for it.
#[derive(Debug, Clone)]
pub struct PlannedBundle {
    pub id: CanonicalBundleId,
    pub resolved: ResolvedCanonicalBundleProduct,
    pub aliases: BTreeSet<PlannedFamilyAlias>,
}

impl PlannedBundle {
    pub fn fetch_key(&self) -> BundleFetchKey {
        BundleFetchKey::from_id(&self.id)
    }

    /// Sorted, deduplicated logical-family slugs (with native overrides
    /// preserved as suffixes) for this bundle. Used by manifest builders
    /// that want to surface "this fetch served these planned families".
    pub fn planned_family_slugs(&self) -> Vec<String> {
        let mut slugs: BTreeSet<String> = BTreeSet::new();
        for alias in &self.aliases {
            if let Some(name) = alias.logical_family.as_deref() {
                slugs.insert(name.to_string());
            }
            // Always include the alias bundle as an indirect alias slug
            // when it differs from the canonical (e.g., NativeAnalysis
            // -> SurfaceAnalysis on HRRR).
            if alias.bundle != self.id.bundle {
                slugs.insert(self.canonical_planned_family_for(alias.bundle));
            }
        }
        slugs.into_iter().collect()
    }

    fn canonical_planned_family_for(&self, bundle: CanonicalBundleDescriptor) -> String {
        // Resolve what bundle would have been chosen if requested directly
        // (with no native override). Handy for manifest aliasing.
        resolve_canonical_bundle_product(self.id.model, bundle, None).native_product
    }
}

/// The planner's deduplicated view of a single run.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub model: ModelId,
    pub cycle: CycleSpec,
    pub source: SourceId,
    pub forecast_hour: u16,
    pub bundles: Vec<PlannedBundle>,
}

impl ExecutionPlan {
    pub fn latest(&self) -> LatestRun {
        LatestRun {
            model: self.model,
            cycle: self.cycle.clone(),
            source: self.source,
        }
    }

    /// All distinct fetch keys, one per physical GRIB file the loader
    /// will pull. Multiple `PlannedBundle`s can map onto the same key.
    pub fn fetch_keys(&self) -> Vec<BundleFetchKey> {
        let mut keys: BTreeSet<BundleFetchKey> = BTreeSet::new();
        for bundle in &self.bundles {
            keys.insert(bundle.fetch_key());
        }
        keys.into_iter().collect()
    }

    pub fn bundle_for(
        &self,
        bundle: CanonicalBundleDescriptor,
        forecast_hour: u16,
    ) -> Option<&PlannedBundle> {
        self.bundles.iter().find(|planned| {
            planned.id.bundle == bundle && planned.id.forecast_hour == forecast_hour
        })
    }

    pub fn bundles_by_family(&self, bundle: CanonicalBundleDescriptor) -> Vec<&PlannedBundle> {
        self.bundles
            .iter()
            .filter(|planned| planned.id.bundle == bundle)
            .collect()
    }
}

/// Builder used by the per-lane batches and unified runners to assemble
/// an `ExecutionPlan` incrementally.
#[derive(Debug)]
pub struct ExecutionPlanBuilder {
    model: ModelId,
    cycle: CycleSpec,
    forecast_hour: u16,
    source: SourceId,
    bundles: BTreeMap<CanonicalBundleId, PlannedBundle>,
}

impl ExecutionPlanBuilder {
    pub fn new(latest: &LatestRun, forecast_hour: u16) -> Self {
        Self {
            model: latest.model,
            cycle: latest.cycle.clone(),
            forecast_hour,
            source: latest.source,
            bundles: BTreeMap::new(),
        }
    }

    pub fn model(&self) -> ModelId {
        self.model
    }

    pub fn forecast_hour(&self) -> u16 {
        self.forecast_hour
    }

    pub fn require(&mut self, requirement: &BundleRequirement) -> CanonicalBundleId {
        self.require_with_logical_family(requirement, None)
    }

    pub fn require_with_logical_family(
        &mut self,
        requirement: &BundleRequirement,
        logical_family: Option<&str>,
    ) -> CanonicalBundleId {
        self.require_with_logical_family_and_patterns::<std::iter::Empty<String>, String>(
            requirement,
            logical_family,
            std::iter::empty(),
        )
    }

    pub fn require_with_logical_family_and_patterns<I, S>(
        &mut self,
        requirement: &BundleRequirement,
        logical_family: Option<&str>,
        variable_patterns: I,
    ) -> CanonicalBundleId
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let id = resolve_canonical_bundle_id(
            self.model,
            self.cycle.clone(),
            requirement.forecast_hour,
            self.source,
            requirement,
        );
        let resolved = resolve_canonical_bundle_product(
            self.model,
            requirement.bundle,
            requirement.native_override.as_deref(),
        );
        let mut alias = PlannedFamilyAlias::from_requirement(requirement);
        if let Some(family) = logical_family {
            alias = alias.with_logical_family(family);
        }
        alias = alias.with_variable_patterns(variable_patterns);
        let entry = self
            .bundles
            .entry(id.clone())
            .or_insert_with(|| PlannedBundle {
                id: id.clone(),
                resolved,
                aliases: BTreeSet::new(),
            });
        entry.aliases.insert(alias);
        id
    }

    pub fn build(self) -> ExecutionPlan {
        ExecutionPlan {
            model: self.model,
            cycle: self.cycle,
            forecast_hour: self.forecast_hour,
            source: self.source,
            bundles: self.bundles.into_values().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_core::{CycleSpec, ModelId};

    fn latest_for(model: ModelId, source: SourceId) -> LatestRun {
        LatestRun {
            model,
            cycle: CycleSpec::new("20260415", 18).unwrap(),
            source,
        }
    }

    #[test]
    fn hrrr_native_and_surface_share_a_canonical_bundle_after_dedup() {
        // HRRR `nat`-planned recipes get serviced by the wrfsfc file.
        // The planner expresses this by routing both NativeAnalysis and
        // SurfaceAnalysis requirements at the same forecast hour to the
        // SurfaceAnalysis bundle (since the resolved native product is
        // identical when overridden), keeping both alias names so
        // manifest publication still surfaces the planned `nat` slug.
        let mut builder = ExecutionPlanBuilder::new(&latest_for(ModelId::Hrrr, SourceId::Aws), 6);
        let surface_id = builder.require_with_logical_family(
            &BundleRequirement::new(CanonicalBundleDescriptor::SurfaceAnalysis, 6),
            Some("sfc"),
        );
        // HRRR `nat` direct recipes route onto the surface fetch by
        // overriding the native product to "sfc"; the alias is preserved.
        let nat_id = builder.require_with_logical_family(
            &BundleRequirement::new(CanonicalBundleDescriptor::SurfaceAnalysis, 6)
                .with_native_override("sfc"),
            Some("nat"),
        );
        assert_eq!(surface_id, nat_id);
        let plan = builder.build();
        assert_eq!(plan.bundles.len(), 1);
        let only = &plan.bundles[0];
        assert_eq!(only.resolved.native_product, "sfc");
        assert!(
            only.aliases
                .iter()
                .any(|alias| alias.logical_family.as_deref() == Some("nat"))
        );
        assert!(
            only.aliases
                .iter()
                .any(|alias| alias.logical_family.as_deref() == Some("sfc"))
        );
    }

    #[test]
    fn gfs_surface_and_pressure_remain_distinct_canonical_bundles_but_share_fetch_key() {
        // GFS surface + pressure both come from `pgrb2.0p25`. Two
        // distinct CanonicalBundleIds (different bundle discriminator)
        // but a single fetch key the loader can satisfy with one HTTP
        // request.
        let mut builder =
            ExecutionPlanBuilder::new(&latest_for(ModelId::Gfs, SourceId::Nomads), 12);
        let surface_id = builder.require(&BundleRequirement::new(
            CanonicalBundleDescriptor::SurfaceAnalysis,
            12,
        ));
        let pressure_id = builder.require(&BundleRequirement::new(
            CanonicalBundleDescriptor::PressureAnalysis,
            12,
        ));
        assert_ne!(surface_id, pressure_id);
        let plan = builder.build();
        assert_eq!(plan.bundles.len(), 2);
        let fetch_keys = plan.fetch_keys();
        assert_eq!(fetch_keys.len(), 1);
        assert_eq!(fetch_keys[0].native_product, "pgrb2.0p25");
    }

    #[test]
    fn distinct_forecast_hours_do_not_collapse() {
        let mut builder = ExecutionPlanBuilder::new(&latest_for(ModelId::Hrrr, SourceId::Aws), 6);
        let f1 = builder.require(&BundleRequirement::new(
            CanonicalBundleDescriptor::SurfaceAnalysis,
            1,
        ));
        let f2 = builder.require(&BundleRequirement::new(
            CanonicalBundleDescriptor::SurfaceAnalysis,
            2,
        ));
        assert_ne!(f1, f2);
        let plan = builder.build();
        assert_eq!(plan.bundles.len(), 2);
    }
}
