//! Small strict JSON value reader for the offline bridge, without new dependencies.
use std::collections::BTreeMap;
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Array(Vec<Value>),
    Object(BTreeMap<String, Value>),
}
impl Value {
    pub fn get(&self, k: &str) -> Result<&Value, String> {
        if let Self::Object(o) = self {
            o.get(k).ok_or_else(|| format!("SPEC missing {k}"))
        } else {
            Err("expected JSON object".into())
        }
    }
    pub fn string(&self) -> Result<&str, String> {
        if let Self::Str(s) = self {
            Ok(s)
        } else {
            Err("expected JSON string".into())
        }
    }
    pub fn array(&self) -> Result<&[Value], String> {
        if let Self::Array(a) = self {
            Ok(a)
        } else {
            Err("expected JSON array".into())
        }
    }
    pub fn number(&self) -> Result<f64, String> {
        if let Self::Num(n) = self {
            Ok(*n)
        } else {
            Err("expected JSON number".into())
        }
    }
    pub fn render(&self) -> String {
        match self {
            Self::Null => "null".into(),
            Self::Bool(b) => b.to_string(),
            Self::Num(n) => n.to_string(),
            Self::Str(s) => quote(s),
            Self::Array(a) => format!(
                "[{}]",
                a.iter().map(Self::render).collect::<Vec<_>>().join(",")
            ),
            Self::Object(o) => format!(
                "{{{}}}",
                o.iter()
                    .map(|(k, v)| format!("{}:{}", quote(k), v.render()))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}
pub fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
pub fn parse(s: &str) -> Result<Value, String> {
    let mut p = Parser {
        s: s.as_bytes(),
        i: 0,
    };
    let v = p.value(0)?;
    p.space();
    if p.i != p.s.len() {
        return Err("trailing JSON bytes".into());
    }
    Ok(v)
}
struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}
impl Parser<'_> {
    fn space(&mut self) {
        while self
            .s
            .get(self.i)
            .map_or(false, |c| matches!(c, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.i += 1
        }
    }
    fn take(&mut self, c: u8) -> Result<(), String> {
        self.space();
        if self.s.get(self.i) != Some(&c) {
            return Err(format!("expected JSON byte {} at {}", c, self.i));
        }
        self.i += 1;
        Ok(())
    }
    fn hex(&mut self) -> Result<u16, String> {
        let end = self.i + 4;
        let t = self.s.get(self.i..end).ok_or("short unicode escape")?;
        self.i = end;
        u16::from_str_radix(
            std::str::from_utf8(t).map_err(|_| "invalid unicode escape")?,
            16,
        )
        .map_err(|_| "invalid unicode escape".into())
    }
    fn string(&mut self) -> Result<String, String> {
        self.take(b'"')?;
        let mut out = Vec::new();
        loop {
            let c = *self.s.get(self.i).ok_or("unterminated JSON string")?;
            self.i += 1;
            match c {
                b'"' => return String::from_utf8(out).map_err(|_| "invalid UTF-8".into()),
                b'\\' => {
                    let e = *self.s.get(self.i).ok_or("short JSON escape")?;
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'b' => out.push(8),
                        b'f' => out.push(12),
                        b'n' => out.push(10),
                        b'r' => out.push(13),
                        b't' => out.push(9),
                        b'u' => {
                            let first = self.hex()?;
                            let cp = if (0xd800..=0xdbff).contains(&first) {
                                if self.s.get(self.i..self.i + 2) != Some(b"\\u") {
                                    return Err("missing low surrogate".into());
                                }
                                self.i += 2;
                                let low = self.hex()?;
                                if !(0xdc00..=0xdfff).contains(&low) {
                                    return Err("invalid low surrogate".into());
                                }
                                0x10000 + ((first as u32 - 0xd800) << 10) + (low as u32 - 0xdc00)
                            } else {
                                first as u32
                            };
                            let ch = char::from_u32(cp).ok_or("invalid unicode scalar")?;
                            let mut bytes = [0; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut bytes).as_bytes());
                        }
                        _ => return Err("invalid JSON escape".into()),
                    }
                }
                0..=31 => return Err("control byte in JSON string".into()),
                _ => out.push(c),
            }
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > 64 {
            return Err("JSON nesting exceeds 64".into());
        }
        self.space();
        let c = *self.s.get(self.i).ok_or("missing JSON value")?;
        match c {
            b'"' => Ok(Value::Str(self.string()?)),
            b'[' => {
                self.i += 1;
                let mut a = Vec::new();
                self.space();
                if self.s.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Value::Array(a));
                }
                loop {
                    a.push(self.value(depth + 1)?);
                    self.space();
                    if self.s.get(self.i) == Some(&b']') {
                        self.i += 1;
                        break;
                    }
                    self.take(b',')?;
                }
                Ok(Value::Array(a))
            }
            b'{' => {
                self.i += 1;
                let mut o = BTreeMap::new();
                self.space();
                if self.s.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Value::Object(o));
                }
                loop {
                    let k = self.string()?;
                    self.take(b':')?;
                    let v = self.value(depth + 1)?;
                    if o.insert(k, v).is_some() {
                        return Err("duplicate JSON key".into());
                    }
                    self.space();
                    if self.s.get(self.i) == Some(&b'}') {
                        self.i += 1;
                        break;
                    }
                    self.take(b',')?;
                }
                Ok(Value::Object(o))
            }
            b'n' | b't' | b'f' => {
                let (word, v) = match c {
                    b'n' => ("null", Value::Null),
                    b't' => ("true", Value::Bool(true)),
                    _ => ("false", Value::Bool(false)),
                };
                if self.s.get(self.i..self.i + word.len()) != Some(word.as_bytes()) {
                    return Err("invalid JSON literal".into());
                }
                self.i += word.len();
                Ok(v)
            }
            b'-' | b'0'..=b'9' => {
                let start = self.i;
                if c == b'-' {
                    self.i += 1
                }
                match self.s.get(self.i) {
                    Some(b'0') => self.i += 1,
                    Some(b'1'..=b'9') => {
                        while self.s.get(self.i).map_or(false, u8::is_ascii_digit) {
                            self.i += 1
                        }
                    }
                    _ => return Err("invalid JSON number".into()),
                };
                if self.s.get(self.i) == Some(&b'.') {
                    self.i += 1;
                    let start = self.i;
                    while self.s.get(self.i).map_or(false, u8::is_ascii_digit) {
                        self.i += 1
                    }
                    if start == self.i {
                        return Err("missing fractional digits".into());
                    }
                }
                if matches!(self.s.get(self.i), Some(b'e' | b'E')) {
                    self.i += 1;
                    if matches!(self.s.get(self.i), Some(b'+' | b'-')) {
                        self.i += 1
                    }
                    let start = self.i;
                    while self.s.get(self.i).map_or(false, u8::is_ascii_digit) {
                        self.i += 1
                    }
                    if start == self.i {
                        return Err("missing exponent digits".into());
                    }
                }
                let n: f64 = std::str::from_utf8(&self.s[start..self.i])
                    .unwrap()
                    .parse()
                    .map_err(|_| "invalid JSON number")?;
                if !n.is_finite() {
                    return Err("non-finite JSON number".into());
                }
                Ok(Value::Num(n))
            }
            _ => Err(format!("invalid JSON value at {}", self.i)),
        }
    }
}
