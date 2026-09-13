//! Received GRIB identity, checked before any individual-member store is opened.
use crate::{Grib2File, Grib2Message, IoError, PreparedSelector, match_prepared_selectors};
use rustwx_core::{CycleSpec, FieldSelector, ModelId};
use rustwx_models::DeclaredMember;

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
    for message in &grib.messages {
        verify_member_record(model, message, member)?;
    }
    Ok(())
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
            || product.ensemble_type != Some(member.ensemble_type)
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
        for message in &grib.messages {
            if has_statistics && message.product.derived_forecast_type.is_some() { continue; }
            verify_member_record(model, message, member)?;
        }
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
