//! An order-preserving, duplicate-refusing JSON tree.
//!
//! Two properties of gpuwm's Python engine are load-carrying and neither
//! survives a plain `serde_json::Value`:
//!
//! * **Declared order.**  A frame states its fields "in the mapping's own
//!   declared order" (`mapped_source._materialize_frames`), which is the
//!   order the keys appear in the mapping DOCUMENT.  `serde_json::Value`
//!   sorts object keys (BTreeMap), and turning on `preserve_order` would
//!   change `rw-wps`'s own serialization through feature unification, so
//!   the order is carried here instead.
//! * **Duplicate refusal.**  `mapped_source._reject_duplicate_object_pairs`
//!   refuses a document that names one key twice, because the last-wins
//!   rule silently discards a producer's declaration.  `serde_json`
//!   last-wins by default.
//!
//! Accessors are deliberately shaped like the Python engine's `mapping[...]`
//! reads so the port stays readable beside its source.

use std::fmt;

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};

/// One JSON value, objects in document order.
#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Null,
    Bool(bool),
    Number(f64),
    Integer(i64),
    String(String),
    Array(Vec<Node>),
    Object(Vec<(String, Node)>),
}

impl Node {
    /// Parse a document, refusing duplicate object keys anywhere in it.
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let mut deserializer = serde_json::Deserializer::from_slice(bytes);
        let node = Node::deserialize(&mut deserializer).map_err(|error| error.to_string())?;
        deserializer.end().map_err(|error| error.to_string())?;
        Ok(node)
    }

    pub fn get(&self, key: &str) -> Option<&Node> {
        match self {
            Node::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    /// `get`, treating an explicit JSON null as absent — the Python
    /// `mapping.get(key)` idiom, where a null declaration and a missing
    /// one reach the same `is None` branch.
    pub fn field(&self, key: &str) -> Option<&Node> {
        match self.get(key) {
            Some(Node::Null) | None => None,
            other => other,
        }
    }

    pub fn entries(&self) -> &[(String, Node)] {
        match self {
            Node::Object(entries) => entries,
            _ => &[],
        }
    }

    pub fn items(&self) -> &[Node] {
        match self {
            Node::Array(items) => items,
            _ => &[],
        }
    }

    pub fn is_object(&self) -> bool {
        matches!(self, Node::Object(_))
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Node::Array(_))
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Node::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Node::Number(value) => Some(*value),
            Node::Integer(value) => Some(*value as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Node::Integer(value) => Some(*value),
            // A JSON `2.0` is an integer in every producer's intent; the
            // Python engine's `_integer` accepts it only when it is exact.
            Node::Number(value) if value.fract() == 0.0 && value.is_finite() => Some(*value as i64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Node::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// A scalar-or-list name spec, flattened to the accepted spellings.
    /// `rw-wps.mapping.v1` lets a selector list every spelling it accepts
    /// (`level` and `pressure_level`), so one string and a list of strings
    /// are the same shape here.
    pub fn as_string_list(&self) -> Option<Vec<String>> {
        match self {
            Node::String(value) => Some(vec![value.clone()]),
            Node::Array(items) => items
                .iter()
                .map(|item| item.as_str().map(str::to_owned))
                .collect(),
            _ => None,
        }
    }

    /// This object with `key` bound to `value`.
    ///
    /// Replaced IN PLACE when the key is already declared, appended in
    /// declared order otherwise, because the order of `fields` is the
    /// order a frame states its fields in.  A non-object answers a
    /// one-entry object: callers here always hold an object (every one
    /// of them read it through an accessor that already refused a
    /// non-object), and returning `self` unchanged would let a
    /// mis-shaped document silently keep the value the caller meant to
    /// replace.
    ///
    /// The composition port builds two documents this way — the
    /// partitioned decode views and the union mapping — rather than by
    /// re-parsing edited JSON text.  The crate's own `Cargo.toml`
    /// records why: a float that round-trips through text can move by
    /// one ULP, and a scaled field's every cell moves with it.  Nothing
    /// built here passes a number through a parser.
    pub fn with_entry(&self, key: &str, value: Node) -> Node {
        let mut entries: Vec<(String, Node)> = match self {
            Node::Object(existing) => existing.clone(),
            _ => Vec::new(),
        };
        match entries.iter_mut().find(|(name, _)| name == key) {
            Some(slot) => slot.1 = value,
            None => entries.push((key.to_owned(), value)),
        }
        Node::Object(entries)
    }

    /// This object keeping only the entries `keep` admits, in order.
    pub fn object_filtered(&self, mut keep: impl FnMut(&str, &Node) -> bool) -> Node {
        Node::Object(
            self.entries()
                .iter()
                .filter(|(name, value)| keep(name, value))
                .cloned()
                .collect(),
        )
    }

    /// Convert to a `serde_json::Value` for output documents.
    pub fn to_value(&self) -> serde_json::Value {
        match self {
            Node::Null => serde_json::Value::Null,
            Node::Bool(value) => serde_json::Value::Bool(*value),
            Node::Integer(value) => serde_json::Value::Number((*value).into()),
            Node::Number(value) => serde_json::Number::from_f64(*value)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            Node::String(value) => serde_json::Value::String(value.clone()),
            Node::Array(items) => {
                serde_json::Value::Array(items.iter().map(Node::to_value).collect())
            }
            Node::Object(entries) => serde_json::Value::Object(
                entries
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_value()))
                    .collect(),
            ),
        }
    }
}

impl fmt::Display for Node {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.to_value())
    }
}

impl<'de> Deserialize<'de> for Node {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(NodeVisitor)
    }
}

struct NodeVisitor;

impl<'de> Visitor<'de> for NodeVisitor {
    type Value = Node;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("any JSON value")
    }

    fn visit_unit<E: de::Error>(self) -> Result<Node, E> {
        Ok(Node::Null)
    }

    fn visit_none<E: de::Error>(self) -> Result<Node, E> {
        Ok(Node::Null)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Node, D::Error> {
        Node::deserialize(deserializer)
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Node, E> {
        Ok(Node::Bool(value))
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Node, E> {
        Ok(Node::Integer(value))
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Node, E> {
        i64::try_from(value)
            .map(Node::Integer)
            .or(Ok(Node::Number(value as f64)))
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Node, E> {
        Ok(Node::Number(value))
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Node, E> {
        Ok(Node::String(value.to_owned()))
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Node, E> {
        Ok(Node::String(value))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> Result<Node, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = access.next_element()? {
            items.push(item);
        }
        Ok(Node::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Node, A::Error> {
        let mut entries: Vec<(String, Node)> = Vec::new();
        while let Some((key, value)) = access.next_entry::<String, Node>()? {
            if entries.iter().any(|(existing, _)| existing == &key) {
                return Err(de::Error::custom(format!("duplicate JSON object key '{key}'")));
            }
            entries.push((key, value));
        }
        Ok(Node::Object(entries))
    }
}

#[cfg(test)]
mod tests {
    use super::Node;

    #[test]
    fn object_keys_keep_document_order() {
        let node = Node::parse(br#"{"zulu": 1, "alpha": 2, "mike": 3}"#).unwrap();
        let names: Vec<&str> = node
            .entries()
            .iter()
            .map(|(key, _)| key.as_str())
            .collect();
        assert_eq!(names, vec!["zulu", "alpha", "mike"]);
    }

    #[test]
    fn duplicate_object_keys_refuse_rather_than_last_wins() {
        let error = Node::parse(br#"{"a": 1, "a": 2}"#).unwrap_err();
        assert!(error.contains("duplicate JSON object key 'a'"), "{error}");
    }

    #[test]
    fn duplicate_keys_are_refused_at_any_depth() {
        let error = Node::parse(br#"{"outer": {"b": 1, "b": 2}}"#).unwrap_err();
        assert!(error.contains("duplicate JSON object key 'b'"), "{error}");
    }

    #[test]
    fn trailing_content_after_the_document_refuses() {
        assert!(Node::parse(br#"{"a": 1} {"b": 2}"#).is_err());
    }

    #[test]
    fn replacing_an_entry_keeps_its_declared_position() {
        let node = Node::parse(br#"{"zulu": 1, "alpha": 2, "mike": 3}"#).unwrap();
        let replaced = node.with_entry("alpha", Node::Integer(9));
        let names: Vec<&str> = replaced
            .entries()
            .iter()
            .map(|(key, _)| key.as_str())
            .collect();
        assert_eq!(names, vec!["zulu", "alpha", "mike"]);
        assert_eq!(replaced.get("alpha"), Some(&Node::Integer(9)));
    }

    #[test]
    fn a_new_entry_is_appended_rather_than_sorted_in() {
        let node = Node::parse(br#"{"zulu": 1}"#).unwrap();
        let grown = node.with_entry("alpha", Node::Integer(2));
        let names: Vec<&str> = grown
            .entries()
            .iter()
            .map(|(key, _)| key.as_str())
            .collect();
        assert_eq!(names, vec!["zulu", "alpha"]);
    }

    #[test]
    fn filtering_an_object_keeps_the_survivors_in_declared_order() {
        let node = Node::parse(br#"{"zulu": 1, "alpha": 2, "mike": 3}"#).unwrap();
        let kept = node.object_filtered(|name, _| name != "alpha");
        let names: Vec<&str> = kept
            .entries()
            .iter()
            .map(|(key, _)| key.as_str())
            .collect();
        assert_eq!(names, vec!["zulu", "mike"]);
    }

    #[test]
    fn scalar_or_list_name_specs_flatten_the_same_way() {
        let node = Node::parse(br#"{"one": "T", "many": ["T", "t"]}"#).unwrap();
        assert_eq!(
            node.get("one").unwrap().as_string_list().unwrap(),
            vec!["T".to_owned()]
        );
        assert_eq!(
            node.get("many").unwrap().as_string_list().unwrap(),
            vec!["T".to_owned(), "t".to_owned()]
        );
    }
}
