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
    /// Every declared typeOfEnsembleForecast value: a grammar declares a
    /// list when the same member arrives stamped differently by each door.
    pub ensemble_types: Vec<u8>,
    pub encoded_ensemble_size: u8,
    pub generating_process: u8,
    pub forecast_generating_process_id: u8,
    /// Copies of this member a front door serves under another writer's
    /// octets, in the order the grammar declares them.
    pub rewrites: Vec<DeclaredRewrite>,
}

/// One declared rewrite of a member class, read from the same grammar the
/// preparation verifies with: selected only for bytes whose every message
/// carries its writer (Section 1) octets, then held to its whole contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredRewrite {
    pub name: String,
    /// (centre, subcentre, master table, local table) values the writer
    /// stamps; an empty list pins nothing for that octet.
    pub centers: Vec<u16>,
    pub subcenters: Vec<u16>,
    pub master_table_versions: Vec<u8>,
    pub local_table_versions: Vec<u8>,
    pub product_definition_templates: Vec<u16>,
    /// None when the writer dropped the ensemble octets and the member's
    /// path component is its only identity; the bytes a caller verifies
    /// were then fetched by that member's own declared path.
    pub ensemble_types: Option<Vec<u8>>,
    pub encoded_ensemble_size: Option<u8>,
    pub generating_process: Option<u8>,
    pub forecast_generating_process_id: Option<u8>,
}

fn grammar(model: ModelId) -> Option<&'static SourceGrammar> {
    GRAMMARS.iter().find(|grammar| grammar.model == model)
}

fn ensemble_types(value: &Value) -> Vec<u8> {
    match value.as_array() {
        Some(values) => values
            .iter()
            .map(|value| value.as_u64().expect("ensemble type") as u8)
            .collect(),
        None => vec![value.as_u64().expect("ensemble type") as u8],
    }
}

fn integers(value: &Value) -> Vec<u64> {
    match value {
        Value::Null => Vec::new(),
        Value::Array(values) => values
            .iter()
            .map(|value| value.as_u64().expect("declared integer"))
            .collect(),
        other => vec![other.as_u64().expect("declared integer")],
    }
}

fn rewrites(class: &Value) -> Vec<DeclaredRewrite> {
    let Some(declared) = class.get("rewrites").and_then(Value::as_array) else {
        return Vec::new();
    };
    declared
        .iter()
        .map(|rewrite| {
            let writer = &rewrite["writer"];
            let by_path = rewrite["perturbation_number"] == "path";
            DeclaredRewrite {
                name: rewrite["name"].as_str().expect("rewrite name").to_string(),
                centers: integers(&writer["center"]).into_iter().map(|v| v as u16).collect(),
                subcenters: integers(&writer["subcenter"]).into_iter().map(|v| v as u16).collect(),
                master_table_versions: integers(&writer["master_table_version"])
                    .into_iter()
                    .map(|v| v as u8)
                    .collect(),
                local_table_versions: integers(&writer["local_table_version"])
                    .into_iter()
                    .map(|v| v as u8)
                    .collect(),
                product_definition_templates: integers(&rewrite["product_definition_templates"])
                    .into_iter()
                    .map(|v| v as u16)
                    .collect(),
                ensemble_types: (!by_path).then(|| ensemble_types(&rewrite["type_of_ensemble_forecast"])),
                encoded_ensemble_size: rewrite["ensemble_size"].as_u64().map(|v| v as u8),
                generating_process: rewrite["type_of_generating_process"].as_u64().map(|v| v as u8),
                forecast_generating_process_id: rewrite["forecast_generating_process_id"]
                    .as_u64()
                    .map(|v| v as u8),
            }
        })
        .collect()
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
                ensemble_types: ensemble_types(&verification["type_of_ensemble_forecast"]),
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
                rewrites: rewrites(class),
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
        assert_eq!(
            declared_member(ModelId::Aigefs, 0).unwrap().unwrap().ensemble_types,
            vec![3, 6]
        );
        assert_eq!(
            declared_member(ModelId::Gefs, 0).unwrap().unwrap().ensemble_types,
            vec![1]
        );
        // The mirror's two archive rewrites, read from the same table the
        // preparation verifies with; GEFS declares none.
        let rewrites = declared_member(ModelId::Aigefs, 17).unwrap().unwrap().rewrites;
        assert_eq!(rewrites.len(), 2);
        assert_eq!(rewrites[0].ensemble_types, None);
        assert_eq!(rewrites[0].product_definition_templates, vec![0, 8]);
        assert_eq!(rewrites[0].centers, vec![74, 7]);
        assert_eq!((rewrites[0].master_table_versions.clone(), rewrites[0].local_table_versions.clone()), (vec![4], vec![0]));
        assert_eq!(rewrites[1].ensemble_types, Some(vec![3]));
        assert_eq!(rewrites[1].generating_process, Some(255));
        assert!(declared_member(ModelId::Gefs, 0).unwrap().unwrap().rewrites.is_empty());
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
