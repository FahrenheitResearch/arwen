//! Individual member identities and URLs from the preparation source grammar.
use rustwx_core::{ModelId, ModelRunRequest, SourceId};
use serde_json::Value;
use std::sync::LazyLock;

struct SourceGrammar {
    model: ModelId,
    document: Value,
    products: &'static [(&'static str, &'static str)],
}

static GRAMMARS: LazyLock<Vec<SourceGrammar>> = LazyLock::new(|| {
    vec![
        SourceGrammar {
            model: ModelId::Gefs,
            document: serde_json::from_str(include_str!("member_tables/gefs.json"))
                .expect("packaged member grammar"),
            products: &[
                ("pgrb2ap5", "pgrb2a"),
                ("pgrb2bp5", "pgrb2b"),
                ("pgrb2sp25", "pgrb2s"),
            ],
        },
        SourceGrammar {
            model: ModelId::Aigefs,
            document: serde_json::from_str(include_str!("member_tables/aigefs.json"))
                .expect("packaged member grammar"),
            products: &[("pres", "pres"), ("sfc", "sfc")],
        },
    ]
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredMember {
    pub ordinal: u8,
    pub id: String,
    pub token: String,
    pub product_definition_templates: Vec<u16>,
    pub ensemble_type: u8,
    pub encoded_ensemble_size: u8,
    pub generating_process: u8,
    pub forecast_generating_process_id: u8,
}

fn grammar(model: ModelId) -> Option<&'static SourceGrammar> {
    GRAMMARS.iter().find(|grammar| grammar.model == model)
}

fn expand_ordinal(template: &str, ordinal: u8) -> String {
    template
        .replace("{ordinal:03d}", &format!("{ordinal:03}"))
        .replace("{ordinal:02d}", &format!("{ordinal:02}"))
        .replace("{ordinal}", &ordinal.to_string())
}

pub fn declared_member(model: ModelId, ordinal: u8) -> Result<Option<DeclaredMember>, String> {
    let Some(grammar) = grammar(model) else {
        return Ok(None);
    };
    for class in grammar.document["classes"]
        .as_object()
        .expect("member classes")
        .values()
    {
        let ordinals = &class["ordinals"];
        let included = if let Some(values) = ordinals.as_array() {
            values
                .iter()
                .any(|value| value.as_u64() == Some(u64::from(ordinal)))
        } else {
            ordinals["first"]
                .as_u64()
                .is_some_and(|first| first <= u64::from(ordinal))
                && ordinals["last"]
                    .as_u64()
                    .is_some_and(|last| last >= u64::from(ordinal))
        };
        if included {
            let verification = &class["verification"];
            return Ok(Some(DeclaredMember {
                ordinal,
                id: expand_ordinal(class["member_id"].as_str().expect("member id"), ordinal),
                token: expand_ordinal(class["token"].as_str().expect("member token"), ordinal),
                product_definition_templates: verification["product_definition_templates"]
                    .as_array()
                    .expect("PDTs")
                    .iter()
                    .map(|value| value.as_u64().expect("PDT") as u16)
                    .collect(),
                ensemble_type: verification["type_of_ensemble_forecast"]
                    .as_u64()
                    .expect("ensemble type") as u8,
                encoded_ensemble_size: verification["ensemble_size"]
                    .as_u64()
                    .expect("ensemble size") as u8,
                generating_process: verification["type_of_generating_process"]
                    .as_u64()
                    .expect("generating process") as u8,
                forecast_generating_process_id: verification["forecast_generating_process_id"]
                    .as_u64()
                    .expect("forecast process")
                    as u8,
            }));
        }
    }
    Err(format!("{model} has no declared member ordinal {ordinal}"))
}

pub fn selected_member_product(
    model: ModelId,
    product: &str,
    ordinal: u8,
) -> Result<String, String> {
    let Some(member) = declared_member(model, ordinal)? else {
        if ordinal != 0 {
            return Err(format!(
                "{model} has no declared individual member selection"
            ));
        }
        return Ok(product.to_string());
    };
    let family = product.split('/').next().unwrap_or(product);
    if !grammar(model)
        .unwrap()
        .products
        .iter()
        .any(|(alias, _)| *alias == family)
    {
        return Err(format!(
            "{model} member products do not declare family {family}"
        ));
    }
    if let Some((_, selected)) = product.split_once('/') {
        if grammar(model).unwrap().document["statistics"]
            .get(selected)
            .is_some()
        {
            return Err(format!(
                "{model} product {product} is a statistic, not an individual member product"
            ));
        }
        if product_member(model, product).is_none() {
            return Err(format!(
                "{model} has no declared individual product {product}"
            ));
        }
    }
    Ok(format!("{family}/{}", member.token))
}

/// Resolve the full token against declared members, never trailing digits.
pub fn product_member(model: ModelId, product: &str) -> Option<DeclaredMember> {
    let (family, token) = product.split_once('/')?;
    let grammar = grammar(model)?;
    if !grammar.products.iter().any(|(alias, _)| *alias == family) {
        return None;
    }
    let count = grammar.document["declared_member_count"].as_u64()?;
    (0..count)
        .filter_map(|ordinal| declared_member(model, ordinal as u8).ok().flatten())
        .find(|member| member.token == token)
}

pub fn declared_members(model: ModelId) -> Vec<DeclaredMember> {
    let Some(grammar) = grammar(model) else {
        return Vec::new();
    };
    let count = grammar.document["declared_member_count"]
        .as_u64()
        .expect("member count");
    (0..count)
        .filter_map(|ordinal| declared_member(model, ordinal as u8).ok().flatten())
        .collect()
}

pub fn member_url(source: SourceId, request: &ModelRunRequest) -> Option<String> {
    let grammar = grammar(request.model)?;
    let member = product_member(request.model, &request.product)?;
    let (family, _) = request.product.split_once('/')?;
    let (_, product_key) = grammar
        .products
        .iter()
        .find(|(alias, _)| *alias == family)?;
    let door = match source {
        SourceId::Nomads => "nomads",
        SourceId::Aws => "aws-open-data",
        _ => return None,
    };
    let base = grammar.document["front_doors"]
        .as_array()?
        .iter()
        .find(|entry| entry["name"] == door)?["base_url"]
        .as_str()?;
    let relative = grammar.document["products"][*product_key]["relative_path"]
        .as_str()?
        .replace("{yyyymmdd}", &request.cycle.date_yyyymmdd)
        .replace("{hh}", &format!("{:02}", request.cycle.hour_utc))
        .replace("{fff}", &format!("{:03}", request.forecast_hour))
        .replace("{token}", &member.token);
    Some(format!("{base}{relative}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_members_follow_the_preparation_grammar() {
        assert_eq!(
            declared_member(ModelId::Aigefs, 0).unwrap().unwrap().token,
            "mem000"
        );
        assert_eq!(
            declared_member(ModelId::Aigefs, 17).unwrap().unwrap().token,
            "mem017"
        );
        assert_eq!(
            declared_member(ModelId::Gefs, 3).unwrap().unwrap().token,
            "gep03"
        );
        assert_eq!(
            declared_member(ModelId::Gefs, 0)
                .unwrap()
                .unwrap()
                .encoded_ensemble_size,
            30
        );
        assert_eq!(
            declared_member(ModelId::Aigefs, 0)
                .unwrap()
                .unwrap()
                .encoded_ensemble_size,
            31
        );
        assert!(declared_member(ModelId::Aigefs, 31).is_err());
        assert!(product_member(ModelId::Aigefs, "sfc/not-a-member001").is_none());
        assert!(product_member(ModelId::Aigefs, "foo/mem001").is_none());
        assert_eq!(
            selected_member_product(ModelId::Aigefs, "pres/mem000", 17).unwrap(),
            "pres/mem017"
        );
        assert!(selected_member_product(ModelId::Aigefs, "sfc/avg", 0).is_err());
        assert!(selected_member_product(ModelId::Aigefs, "pres/spr", 17).is_err());
        assert!(selected_member_product(ModelId::Gefs, "pgrb2ap5/geavg", 0).is_err());
        assert!(selected_member_product(ModelId::Gefs, "pgrb2ap5/gespr", 3).is_err());
        assert!(selected_member_product(ModelId::Aigefs, "sfc/foo", 0).is_err());
    }
}
