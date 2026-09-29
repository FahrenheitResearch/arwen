//! Received GRIB identity, checked before any individual-member store is opened.
use crate::{Grib2File, Grib2Message, IoError, PreparedSelector, match_prepared_selectors};
use rustwx_core::{CycleSpec, FieldSelector, ModelId};
use rustwx_models::{DeclaredMember, DeclaredRewrite};

/// Verify every message against the selected preparation grammar and cycle.
/// This reads headers only; no weather values are unpacked or altered.
pub fn verify_model_member_bytes(
    model: ModelId,
    bytes: &[u8],
    member: &DeclaredMember,
    cycle: &CycleSpec,
) -> Result<(), IoError> {
    let grib = Grib2File::from_bytes(bytes).map_err(|error| IoError::Grib(error.to_string()))?;
    verify_member_grib(model, &grib, member, cycle)
}

pub(crate) fn verify_member_grib(model: ModelId, grib: &Grib2File,
                                member: &DeclaredMember, cycle: &CycleSpec) -> Result<(), IoError> {
    verify_cycle(grib, cycle)?;
    let messages: Vec<&Grib2Message> = grib.messages.iter().collect();
    verify_member_records(model, &messages, member)
}

/// The producer's own encoding first, as the preparation's verifier does;
/// only when it refuses, a declared rewrite whose writer octets every
/// message carries, held to its whole contract.  Bytes whose writer no
/// rewrite declares keep the producer's refusal.
fn verify_member_records(model: ModelId, messages: &[&Grib2Message],
                         member: &DeclaredMember) -> Result<(), IoError> {
    let producer = messages
        .iter()
        .try_for_each(|message| verify_member_record(model, message, member));
    if producer.is_ok() {
        return producer;
    }
    let rewritten = member.rewrites.iter().any(|rewrite| {
        messages.iter().all(|message| writer_admits(rewrite, message))
            && messages.iter().all(|message| rewrite_admits(rewrite, message, member))
    });
    if rewritten { Ok(()) } else { producer }
}

fn writer_admits(rewrite: &DeclaredRewrite, message: &Grib2Message) -> bool {
    let id = &message.identification;
    let pins_nothing = rewrite.centers.is_empty()
        && rewrite.subcenters.is_empty()
        && rewrite.master_table_versions.is_empty()
        && rewrite.local_table_versions.is_empty();
    !pins_nothing
        && (rewrite.centers.is_empty() || rewrite.centers.contains(&id.center_id))
        && (rewrite.subcenters.is_empty() || rewrite.subcenters.contains(&id.subcenter_id))
        && (rewrite.master_table_versions.is_empty()
            || rewrite.master_table_versions.contains(&id.master_table_version))
        && (rewrite.local_table_versions.is_empty()
            || rewrite.local_table_versions.contains(&id.local_table_version))
}

fn rewrite_admits(rewrite: &DeclaredRewrite, message: &Grib2Message,
                  member: &DeclaredMember) -> bool {
    let product = &message.product;
    rewrite.product_definition_templates.contains(&product.template)
        && rewrite.generating_process.is_none_or(|value| product.generating_process == value)
        && rewrite
            .forecast_generating_process_id
            .is_none_or(|value| product.forecast_generating_process_id == value)
        && match &rewrite.ensemble_types {
            // No ensemble octets survived the rewrite: the member is the path
            // component the caller fetched these bytes by.
            None => product.ensemble_type.is_none()
                && product.perturbation_number.is_none()
                && product.derived_forecast_type.is_none(),
            Some(types) => product.ensemble_type.is_some_and(|value| types.contains(&value))
                && product.perturbation_number == Some(member.ordinal)
                && rewrite
                    .encoded_ensemble_size
                    .is_none_or(|size| product.num_forecasts_in_ensemble == Some(size)),
        }
}

fn verify_cycle(grib: &Grib2File, cycle: &CycleSpec) -> Result<(), IoError> {
    if grib.messages.is_empty() { return Err(IoError::Grib("received file has no messages".into())); }
    let expected_cycle = format!("{}{:02}0000", cycle.date_yyyymmdd, cycle.hour_utc);
    for message in &grib.messages {
        if message.reference_time.format("%Y%m%d%H%M%S").to_string() != expected_cycle {
            return Err(IoError::Grib(format!("received GRIB does not match cycle {expected_cycle}")));
        }
    }
    Ok(())
}

fn verify_member_record(model: ModelId, message: &Grib2Message,
                        member: &DeclaredMember) -> Result<(), IoError> {
        let product = &message.product;
        if !member.product_definition_templates.contains(&product.template)
            || !product.ensemble_type.is_some_and(|value| member.ensemble_types.contains(&value))
            || product.perturbation_number != Some(member.ordinal)
            || product.num_forecasts_in_ensemble != Some(member.encoded_ensemble_size)
            || product.generating_process != member.generating_process
            || product.forecast_generating_process_id != member.forecast_generating_process_id
        {
            return Err(IoError::Grib(format!(
                "received GRIB does not match {model} member {}",
                member.id
            )));
        }
    Ok(())
}

/// Verify exact selected planes, including mixed statistical/member products.
pub fn verify_model_selected_bytes(model: ModelId, bytes: &[u8], member: Option<&DeclaredMember>,
    cycle: &CycleSpec, selectors: &[FieldSelector], forecast_hour: u16) -> Result<(), IoError> {
    let grib = Grib2File::from_bytes(bytes).map_err(|error| IoError::Grib(error.to_string()))?;
    verify_selected_grib(model, &grib, member, cycle, selectors, forecast_hour)
}

pub(crate) fn verify_selected_grib(model: ModelId, grib: &Grib2File,
    member: Option<&DeclaredMember>, cycle: &CycleSpec, selectors: &[FieldSelector],
    forecast_hour: u16) -> Result<(), IoError> {
    verify_cycle(grib, cycle)?;
    if selectors.is_empty() { return Err(IoError::Grib("selected field set is empty".into())); }
    let has_individual = selectors.iter().any(|selector| selector.product.is_default());
    let has_statistics = selectors.iter().any(|selector| !selector.product.is_default());
    if let Some(member) = member.filter(|_| has_individual) {
        let individual: Vec<&Grib2Message> = grib
            .messages
            .iter()
            .filter(|message| !(has_statistics && message.product.derived_forecast_type.is_some()))
            .collect();
        verify_member_records(model, &individual, member)?;
    }
    let prepared = selectors.iter().copied().map(PreparedSelector::new).collect::<Result<Vec<_>, _>>()?;
    let matched = match_prepared_selectors(grib, &prepared, Some(forecast_hour));
    for (selector, message) in selectors.iter().zip(matched) {
        let Some((message, _)) = message else {
            return Err(IoError::Grib(format!("requested native field '{}' is absent at f{forecast_hour:03}", selector.key())));
        };
        if selector.product.is_default() && message.product.derived_forecast_type.is_some() {
            return Err(IoError::Grib(format!("statistical GRIB cannot represent the default field '{}'; select its explicit statistic", selector.key())));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustwx_models::declared_member;
    use std::path::Path;

    fn fixture(relative: &str) -> Vec<u8> {
        // Existing, unmodified public production envelopes; hashes and the
        // original retrieval are recorded in the corpus README.
        std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../tests/fixtures/ensemble-member-identity")
            .join(relative)).expect("retained real member fixture")
    }

    fn gefs(name: &str) -> Vec<u8> {
        fixture(&format!("gefs.20260817/00/atmos/pgrb2ap5/{name}.t00z.pgrb2a.0p50.f000"))
    }

    #[test]
    fn actual_members_and_accumulation_records_verify_by_their_own_grammar() {
        let cycle = CycleSpec::new("20260817", 0).unwrap();
        for (model, ordinal, bytes) in [
            (ModelId::Gefs, 0, gefs("gec00")),
            (ModelId::Gefs, 1, gefs("gep01")),
            (ModelId::Gefs, 0, fixture("gefs.20260817/00/atmos/pgrb2ap5/gec00.t00z.pgrb2a.0p50.f003")),
            (ModelId::Aigefs, 0, fixture("aigefs.20260817/00/mem000/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2")),
            (ModelId::Aigefs, 1, fixture("aigefs.20260817/00/mem001/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2")),
            (ModelId::Aigefs, 0, fixture("aigefs.20260817/00/mem000/model/atmos/grib2/aigefs.t00z.sfc.f006.grib2")),
        ] {
            let member = declared_member(model, ordinal).unwrap().unwrap();
            verify_model_member_bytes(model, &bytes, &member, &cycle).unwrap();
        }
    }

    #[test]
    fn the_operational_door_stamp_verifies_as_the_same_member() {
        // NOMADS stamps typeOfEnsembleForecast 6 where the mirror's copy says 3.
        let bytes = fixture("aigefs.20260927/12/mem000/model/atmos/grib2/aigefs.t12z.sfc.f000.grib2");
        let control = declared_member(ModelId::Aigefs, 0).unwrap().unwrap();
        let cycle = CycleSpec::new("20260927", 12).unwrap();
        verify_model_member_bytes(ModelId::Aigefs, &bytes, &control, &cycle).unwrap();
        let mut mirror_only = control.clone();
        mirror_only.ensemble_types = vec![3];
        assert!(verify_model_member_bytes(ModelId::Aigefs, &bytes, &mirror_only, &cycle).is_err());
        let member = declared_member(ModelId::Aigefs, 1).unwrap().unwrap();
        assert!(verify_model_member_bytes(ModelId::Aigefs, &bytes, &member, &cycle).is_err());
    }

    #[test]
    fn the_mirror_rewrites_verify_by_their_declared_writer_and_nothing_else() {
        // Whole envelopes of the mirror's re-encoded archive: 2026-01-20 00Z
        // carries no ensemble octets (PDT 0, centre 74 and 7 on the sea-level
        // pressure record, master 4, local 0), and its mem001 100 m wind pair
        // is the same bytes as mem000's; 2026-04-10 00Z keeps them (PDT 1)
        // with the generating process missing (255).
        let rewrite = |cycle: &str, member: &str| {
            fixture(&format!("aigefs.{cycle}/00/{member}/model/atmos/grib2/aigefs.t00z.sfc.f000.grib2"))
        };
        let winter = CycleSpec::new("20260120", 0).unwrap();
        let spring = CycleSpec::new("20260410", 0).unwrap();
        let control = declared_member(ModelId::Aigefs, 0).unwrap().unwrap();
        let member = declared_member(ModelId::Aigefs, 1).unwrap().unwrap();
        verify_model_member_bytes(ModelId::Aigefs, &rewrite("20260120", "mem000"), &control, &winter).unwrap();
        verify_model_member_bytes(ModelId::Aigefs, &rewrite("20260120", "mem001"), &member, &winter).unwrap();
        verify_model_member_bytes(ModelId::Aigefs, &rewrite("20260410", "mem000"), &control, &spring).unwrap();
        verify_model_member_bytes(ModelId::Aigefs, &rewrite("20260410", "mem001"), &member, &spring).unwrap();
        // The octets that survived still name their member.
        assert!(verify_model_member_bytes(ModelId::Aigefs, &rewrite("20260410", "mem001"), &control, &spring).is_err());
        // Without the declared rewrites the producer's contract refuses both.
        let mut producer_only = control.clone();
        producer_only.rewrites.clear();
        assert!(verify_model_member_bytes(ModelId::Aigefs, &rewrite("20260120", "mem000"), &producer_only, &winter).is_err());
        assert!(verify_model_member_bytes(ModelId::Aigefs, &rewrite("20260410", "mem000"), &producer_only, &spring).is_err());
        // A deterministic GFS-family writer (master 2, local 1) is no rewrite.
        let gdas = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../tests/fixtures/gdas-process-id")
            .join("nomads-gdas-20260729t12z-f000.grib2"))
            .expect("retained real GDAS fixture");
        let gdas_cycle = CycleSpec::new("20260729", 12).unwrap();
        assert!(verify_model_member_bytes(ModelId::Aigefs, &gdas, &control, &gdas_cycle).is_err());
    }

    #[test]
    fn actual_statistics_foreign_members_and_foreign_models_cannot_claim_a_member() {
        let cycle = CycleSpec::new("20260817", 0).unwrap();
        let control = declared_member(ModelId::Gefs, 0).unwrap().unwrap();
        for bytes in [gefs("geavg"), gefs("gespr"), gefs("gep01")] {
            assert!(verify_model_member_bytes(ModelId::Gefs, &bytes, &control, &cycle).is_err());
        }
        let ai_member = declared_member(ModelId::Aigefs, 1).unwrap().unwrap();
        assert!(verify_model_member_bytes(ModelId::Aigefs, &gefs("gep01"), &ai_member, &cycle).is_err());
        let mut mixed = gefs("gec00");
        mixed.extend(gefs("gep01"));
        assert!(verify_model_member_bytes(ModelId::Gefs, &mixed, &control, &cycle).is_err());
    }

    #[test]
    fn exact_cycle_and_declared_generating_process_are_identity() {
        let member = declared_member(ModelId::Gefs, 0).unwrap().unwrap();
        let cycle = CycleSpec::new("20260817", 0).unwrap();
        let mut bytes = gefs("gec00");
        assert!(verify_model_member_bytes(ModelId::Gefs, &bytes, &member,
            &CycleSpec::new("20260817", 6).unwrap()).is_err());
        // Section1 octet18 is the minute of reference time. A different
        // minute must not compare equal merely because the hour is unchanged.
        assert_eq!(bytes[16 + 4], 1);
        bytes[16 + 17] = 1;
        assert!(verify_model_member_bytes(ModelId::Gefs, &bytes, &member, &cycle).is_err());
        let mut wrong_process = member.clone();
        wrong_process.forecast_generating_process_id ^= 1;
        assert!(verify_model_member_bytes(ModelId::Gefs, &gefs("gec00"), &wrong_process, &cycle).is_err());
        assert!(verify_model_member_bytes(ModelId::Gefs, &[], &member, &cycle).is_err());
    }
}
