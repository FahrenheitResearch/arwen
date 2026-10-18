//! MPAS-A v8.4.2 sounding text I/O. Numerical profiles remain on CUDA.
use std::fs;
use std::io::{self, Write};
use std::path::Path;

pub const STR_KIND: usize = 512;
#[derive(Clone, Debug, PartialEq)]
pub struct Location { pub latitude: f32, pub longitude: f32, pub name: Vec<u8> }

#[derive(Clone, Debug)]
enum Item { Value(Vec<u8>), Null, End }

// mpas_soundings.F:148-156,187-191: list-directed separators, quoted character
// fields, repeat counts and slash termination in GNU formatted input.
fn items(text: &[u8], end_of_input: bool) -> Result<Vec<Item>, String> {
    let b = text;
    let mut result = Vec::new();
    let mut i = 0;
    let mut after_comma = true;
    while i < b.len() {
        // GNU list READ skips NUL before an unquoted token, but retains it
        // inside a token or quoted CHARACTER: mpas_soundings.F:148-156,187-191.
        if b[i].is_ascii_whitespace() || b[i]==0 { i += 1; continue; }
        if b[i] == b'/' { result.push(Item::End); break; }
        if b[i] == b',' {
            if after_comma { result.push(Item::Null); }
            after_comma = true; i += 1; continue;
        }
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() { i += 1; }
        let repeat = if i > start && i < b.len() && b[i] == b'*' {
            let count = std::str::from_utf8(&text[start..i]).map_err(|e| e.to_string())?.parse::<usize>().map_err(|e| e.to_string())?;
            if count == 0 { return Err("zero list-directed repeat count".into()); }
            i += 1; count
        } else { i = start; 1 };
        let value;
        if i == b.len() || b[i].is_ascii_whitespace() || b[i] == b',' || b[i] == b'/' {
            value = Item::Null;
        } else if b[i] == b'\'' || b[i] == b'"' {
            let quote = b[i]; i += 1;
            let mut v = Vec::new();
            loop {
                if i == b.len() { if end_of_input {break} else {return Err("incomplete quoted character value".into())} }
                if b[i] == quote {
                    i += 1;
                    if i < b.len() && b[i] == quote { v.push(quote); i += 1; }
                    else { break; }
                } else { if !matches!(b[i],b'\n'|b'\r'){v.push(b[i]);} i += 1; }
            }
            value = Item::Value(v);
        } else {
            let start = i;
            while i < b.len() && !b[i].is_ascii_whitespace() && !matches!(b[i], b',' | b'/') { i += 1; }
            value = Item::Value(text[start..i].to_vec());
        }
        // A READ has only three destinations; expansion past those is unused.
        for _ in 0..repeat.min(3) { result.push(value.clone()); }
        after_comma = false;
        if result.len() >= 3 { break; }
    }
    Ok(result)
}

fn real(value: &str) -> Result<f32, String> {
    let mut s = value.replace(['d', 'D'], "e");
    // GNU accepts a signed exponent without the E letter.
    if !s.contains(['e', 'E']) {
        if let Some(i) = s.char_indices().skip(1).find_map(|(i,c)| matches!(c,'+'|'-').then_some(i)) { s.insert(i, 'e'); }
    }
    s.parse::<f32>().map_err(|_| format!("invalid list-directed REAL: {value}"))
}

fn record(text: &[u8], previous: Option<&Location>, end_of_input: bool) -> Result<Option<Location>, String> {
    let fields = items(text,end_of_input)?;
    if fields.len() < 3 && !fields.iter().any(|v| matches!(v, Item::End)) { return Ok(None); }
    let mut lat = previous.map(|l| l.latitude);
    let mut lon = previous.map(|l| l.longitude);
    let mut name = previous.map(|l| l.name.clone());
    for (i, item) in fields.iter().take(3).enumerate() {
        match item {
            Item::End => break,
            Item::Null => (),
            Item::Value(v) => match i {
                0 => lat = Some(real(std::str::from_utf8(v).map_err(|e| e.to_string())?)?),
                1 => lon = Some(real(std::str::from_utf8(v).map_err(|e| e.to_string())?)?),
                _ => name = Some(v[..v.len().min(STR_KIND)].to_vec()),
            },
        }
    }
    match (lat, lon, name) {
        (Some(latitude), Some(longitude), Some(name)) => Ok(Some(Location { latitude, longitude, name })),
        _ => Err("location READ leaves an undefined native destination".into()),
    }
}

/// Two native passes: count successful external READs, rewind, read precisely
/// that many physical records into character(len=StrKIND), then internal READ.
/// mpas_soundings.F:148-203. A malformed count record terminates the count.
pub fn read_locations(text: &str) -> Result<Vec<Location>, String> {read_locations_bytes(text.as_bytes())}

/// Fortran CHARACTER is bytes; preserve non-UTF8 station names and truncation.
/// mpas_soundings.F:174-191,274-289.
pub fn read_locations_bytes(text: &[u8]) -> Result<Vec<Location>, String> {
    // GNU formatted input treats 0xff as an end-of-record character, then
    // continues at the next physical record. Preserve byte names verbatim.
    let lines: Vec<&[u8]> = text.split(|c| *c==b'\n').collect();
    let mut cursor = 0;
    let mut count = 0;
    // First-pass destinations retain values on null fields. No first-pass
    // coordinate/name is used by diagnostics; the second pass owns storage.
    let mut previous: Option<Location> = Some(Location {latitude:0.0,longitude:0.0,name:Vec::new()});
    while cursor < lines.len() {
        let mut external = Vec::new();
        let parsed = loop {
            if cursor == lines.len() { break None; }
            let physical=lines[cursor];
            let stop=physical.iter().position(|b|*b==0xff);
            external.extend_from_slice(&physical[..stop.unwrap_or(physical.len())]);
            external.push(b'\n'); cursor += 1;
            match record(&external, previous.as_ref(),stop.is_some() || cursor==lines.len()) {
                Ok(Some(l)) => break Some(l),
                Ok(None) => continue,
                Err(e) if e=="incomplete quoted character value" => continue,
                Err(_) => break None,
            }
        };
        let Some(location) = parsed else { break; };
        previous = Some(location); count += 1;
    }
    let mut locations = Vec::with_capacity(count);
    for line in lines.iter().take(count) {
        let bytes = line.strip_suffix(b"\r").unwrap_or(line);
        let end=bytes.iter().position(|b|*b==0xff).unwrap_or(bytes.len());
        let line = &bytes[..end.min(STR_KIND)];
        // mpas_soundings.F:174-191 leaves null destinations unassigned. Real
        // strict-native repeat/host/rank runs vary for exactly these stations;
        // zero supplies explicit initial storage for the retained destination.
        // DECLARED NATIVE UNDEFINED: unassigned coordinates and dependent
        // nearest-cell ownership/profile words. Valid stations stay bitwise.
        // Unassigned names reproduce fresh standalone NUL storage; real-native
        // character-storage acceptance remains a separately witnessed item.
        // receipts/native-null-list-policy.json and native-list-snan.json.
        // Standalone fresh allocation stores zero, which this also reproduces.
        let initial=Location {latitude:0.0,longitude:0.0,name:vec![0;STR_KIND]};
        let location = record(line, Some(&initial),true)?.ok_or("end of internal location record")?;
        locations.push(location);
    }
    Ok(locations)
}

/// Exact binary32 -> decimal Fw.2, GNU's default nearest/even mode. Integer
/// arithmetic avoids a double conversion/rounding of the native REAL word.
/// mpas_soundings.F:288,324-331 (formatted values; no meteorological maths).
pub fn fixed(value: f32, width: usize) -> String {
    let pad = |s: &str| if s.len() > width { "*".repeat(width) } else { format!("{s:>width$}") };
    if value.is_nan() { return pad("NaN"); }
    if value.is_infinite() {
        let word = if width >= 8 + usize::from(value.is_sign_negative()) { "Infinity" } else { "Inf" };
        return pad(&format!("{}{word}", if value.is_sign_negative() { "-" } else { "" }));
    }
    if value.abs() >= 1.0e10 { return "*".repeat(width); }
    let bits = value.to_bits();
    let e = ((bits >> 23) & 255) as i32;
    let mantissa = if e == 0 { bits & 0x7fffff } else { (bits & 0x7fffff) | 0x800000 };
    let power = if e == 0 { -149 } else { e - 150 };
    let scaled = u128::from(mantissa) * 100;
    let cents = if power >= 0 { scaled << power } else {
        let shift = (-power) as u32;
        if shift >= 128 { 0 } else {
            let integer = scaled >> shift;
            let rem = scaled - (integer << shift);
            let half = 1u128 << (shift - 1);
            integer + u128::from(rem > half || (rem == half && integer & 1 != 0))
        }
    };
    let sign = if bits >> 31 != 0 { "-" } else { "" };
    let mut s = format!("{sign}{}.{:02}", cents / 100, cents % 100);
    if s.len() > width && cents < 100 { s.remove(sign.len()); }
    pad(&s)
}

/// Write every owned station in original order, replacing an existing file.
/// mpas_soundings.F:261-338. Rows are (6,levels,stations) in Fortran order.
pub fn write_soundings(directory: &Path, timestamp: &str, locations: &[Location], owned: &[i32], levels: usize, rows: &[f32]) -> io::Result<Vec<String>> {
    let t = crate::history::Timestamp::parse(timestamp).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput,e.to_string()))?;
    if owned.len() != locations.len() || rows.len() != locations.len() * levels * 6 { return Err(io::Error::new(io::ErrorKind::InvalidInput,"sounding buffer extents disagree")); }
    let mut names = Vec::new();
    for (i, l) in locations.iter().enumerate() {
        if owned[i] == 0 { continue; }
        let end=l.name.iter().rposition(|b|*b!=b' ').map_or(0,|i|i+1);
        let name=&l.name[..end];
        let mut fname=name.to_vec();fname.extend_from_slice(format!(".{:04}{:02}{:02}{:02}{:02}.snd",t.year,t.month,t.day,t.hour,t.minute).as_bytes());
        // mpas_soundings.F:275: an oversized internal filename WRITE fails
        // before OPEN; silently truncating would create a file native lacks.
        if fname.len()>STR_KIND {return Err(io::Error::new(io::ErrorKind::InvalidInput,"native filename internal record exceeds StrKIND bytes"));}
        // mpas_soundings.F:277: GNU OPEN passes its CHARACTER filename to
        // the C filesystem. Embedded NUL ends the path, but remains in STID.
        // The internal record extent check above still counts every byte.
        let path_end=fname.iter().position(|b|*b==0).unwrap_or(fname.len());
        let path_bytes=&fname[..path_end];
        // GNU OPEN on a quoted leading NUL fails with an empty path, rather
        // than opening the output directory: mpas_soundings.F:277.
        let path=if path_bytes.is_empty() {std::path::PathBuf::new()}
                 else {directory.join(native_path_component(path_bytes)?)};
        let mut file = fs::File::create(path)?;
        // Internal WRITE(stid,'(a)') signals end-of-record for names >10 bytes.
        if name.len() > 10 { return Err(io::Error::new(io::ErrorKind::InvalidInput,"native STID internal record exceeds ten characters")); }
        writeln!(file,"  SNPARM = PRES;HGHT;TMPC;DWPC;DRCT;SPED;")?;
        writeln!(file)?;
        file.write_all(b" STID = ")?;file.write_all(name)?;file.write_all(&vec![b' ';10-name.len()])?;
        writeln!(file,"  STNM = 99999       TIME = {:02}{:02}{:02}/{:02}{:02}",t.year % 100,t.month,t.day,t.hour,t.minute)?;
        writeln!(file," SLAT = {}      SLON = {}     SELV =    -999",fixed(l.latitude,6),fixed(l.longitude,7))?;
        writeln!(file)?;
        writeln!(file,"      PRES      HGHT     TMPC     DWPC     DRCT     SPED")?;
        for row in rows[i*levels*6..(i+1)*levels*6].chunks_exact(6) {
            writeln!(file,"{}{}{}{}{}{}",fixed(row[0],10),fixed(row[1],10),fixed(row[2],9),fixed(row[3],9),fixed(row[4],9),fixed(row[5],9))?;
        }
        names.push(String::from_utf8_lossy(path_bytes).into_owned());
    }
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn decimal_ties_signed_zero_and_overflow() {
        assert_eq!(fixed(1.125,6),"  1.12"); assert_eq!(fixed(1.375,6),"  1.38");
        assert_eq!(fixed(-0.0,6)," -0.00"); assert_eq!(fixed(-0.0,4),"-.00");
        assert_eq!(fixed(9999.999,6),"******");
    }
    #[test] fn two_pass_and_quoted_names() {
        let l = read_locations("45,-1.2D2,'A B'\n2*0,\"quoted\"\nbad line\n1 2 later\n").unwrap();
        assert_eq!(l.len(),2); assert_eq!(l[0].longitude,-120.0); assert_eq!(l[0].name,b"A B");
        assert_eq!(l[1].latitude,0.0); assert_eq!(l[1].name,b"quoted");
        assert!(read_locations("! comment\n1 2 name\n").unwrap().is_empty());
    }
}

/// Gregorian MPAS sounding alarm arithmetic. The driver supplies its current
/// clock string; this bridge owns interval parsing and recurring alarm reset.
pub mod clock {
    const NS: i128 = 1_000_000_000;
    #[derive(Clone, Copy)]
    struct Instant { year: i32, month: u32, day: u32, time: i128 }
    #[derive(Clone, Copy)]
    struct Interval { years: i32, months: i32, nanos: i128 }
    fn leap(y: i32) -> bool { y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) }
    fn month_days(y: i32,m: u32) -> u32 { match m { 2=>if leap(y){29}else{28},4|6|9|11=>30,_=>31 } }
    fn year_days(y: i32) -> i128 { let y=i128::from(y);365*y+(y+3).div_euclid(4)-(y+99).div_euclid(100)+(y+399).div_euclid(400) }
    fn fraction(text: &str) -> Result<i128,String> {
        // mpas_timekeeping.F:1477-1495: integerMaxDigits=8, not nine.
        let s=text.trim_end();let s=&s[..s.len().min(8)];
        if s.is_empty(){return Ok(0)}
        let n=s.parse::<i128>().map_err(|e|e.to_string())?;
        Ok(n*NS/10i128.pow(s.len() as u32))
    }
    fn instant(text: &str) -> Result<Instant,String> {
        let t=crate::history::Timestamp::parse(text).map_err(|e|e.to_string())?;
        let frac=if text.len()>19 && text.as_bytes()[19]==b'.' { fraction(&text[20..])? } else {0};
        Ok(Instant {year:t.year,month:t.month,day:t.day,time:(i128::from(t.hour)*3600+i128::from(t.minute)*60+i128::from(t.second))*NS+frac})
    }
    impl Instant {
        fn ticks(self)->i128 { let mut d=year_days(self.year);for m in 1..self.month {d+=i128::from(month_days(self.year,m));} (d+i128::from(self.day)-1)*86400*NS+self.time }
        fn from_ticks(t:i128)->Self {
            let d=t.div_euclid(86400*NS);let mut y=(d/365) as i32;
            while year_days(y)>d {y-=1} while year_days(y+1)<=d {y+=1}
            let mut day=d-year_days(y);let mut m=1;
            while day>=i128::from(month_days(y,m)){day-=i128::from(month_days(y,m));m+=1;}
            Self {year:y,month:m,day:day as u32+1,time:t.rem_euclid(86400*NS)}
        }
        // The calendar rule MPAS's clock applies to time + interval
        // (mpas_timekeeping.F, operator(+) add_t_ti): years and months first,
        // the day clamped to the target month's length, then the seconds.
        // Implemented here from that rule; it does not copy the time
        // manager code.
        fn add(mut self,i:Interval)->Self {
            if i.years!=0 || i.months!=0 {
                let months=(self.year+i.years)*12+self.month as i32-1+i.months;
                self.year=months.div_euclid(12);self.month=months.rem_euclid(12) as u32+1;
                self.day=self.day.min(month_days(self.year,self.month));
                self.time=self.time/NS*NS; // native sets the month-shifted time in whole seconds.
            }
            Self::from_ticks(self.ticks()+i.nanos)
        }
        fn string(self)->String {
            let seconds=self.time/NS;let mut s=format!("{:04}-{:02}-{:02}_{:02}:{:02}:{:02}",self.year,self.month,self.day,seconds/3600,seconds/60%60,seconds%60);
            let nanos=self.time%NS;if nanos!=0 {s.push('.');s.push_str(format!("{nanos:09}").trim_end_matches('0'));}s
        }
    }
    // mpas_timekeeping.F:1477-1573: date width 1/2/3, time width 1/2/3.
    fn interval(text: &str)->Result<Interval,String> {
        let pieces:Vec<&str>=text.trim_end().split('.').collect();
        if pieces.len()>2{return Err("invalid TimeInterval decimals".into())}
        let nanos=if pieces.len()==2 {fraction(pieces[1])?}else{0};
        let parts:Vec<&str>=pieces[0].split('_').collect();
        let (date,time)=match parts.as_slice(){[time] => (None,*time),[date,time] => (Some(*date),*time),_=>return Err("invalid TimeInterval date/time".into())};
        let parse=|s:&str|s.trim().parse::<i128>().map_err(|e|e.to_string());
        let (years,months,days)=if let Some(date)=date {
            let p:Vec<i128>=date.split('-').map(parse).collect::<Result<_,_>>()?;
            match p.as_slice(){[d]=>(0,0,*d),[m,d]=>(0,*m,*d),[y,m,d]=>(*y,*m,*d),_=>(0,0,0)}
        }else{(0,0,0)};
        let p:Vec<i128>=time.split(':').map(parse).collect::<Result<_,_>>()?;
        let seconds=match p.as_slice(){[s]=>*s,[m,s]=>m*60+s,[h,m,s]=>h*3600+m*60+s,_=>return Err("invalid TimeInterval time".into())};
        Ok(Interval {years:years as i32,months:months as i32,nanos:(days*86400+seconds)*NS+nanos})
    }
    impl Interval {fn mul(self,n:i64)->Self {Self {years:self.years*n as i32,months:self.months*n as i32,nanos:self.nanos*i128::from(n)}}}
    #[derive(serde::Serialize,serde::Deserialize,Debug)]
    pub struct Request {pub now:String,pub interval:String,#[serde(default)] pub previous:Option<String>,#[serde(default)]pub reset:bool}
    #[derive(serde::Serialize,serde::Deserialize,Debug)]
    pub struct Response {pub ringing:bool,pub previous:String,pub next:String}
    /// Initial prevRingTime=start-interval, envelope <=now and native binary
    /// interval division on reset. mpas_timekeeping.F:490-566,876-985,1849-1932.
    pub fn query(request:Request)->Result<Response,String> {
        let now=instant(&request.now)?;let i=interval(&request.interval)?;
        let mut previous=if let Some(s)=request.previous {instant(&s)?}else{now.add(i.mul(-1))};
        let ringing=previous.add(i).ticks()<=now.ticks();
        if ringing && request.reset {
            if i.years==0 && i.months==0 && i.nanos==0 {previous=now;}
            else {
                let mut updated=previous.add(i);let mut n=1i64;
                if updated.ticks()<=previous.ticks(){return Err("native interval division does not advance for this interval".into())}
                if updated.ticks()==now.ticks(){previous=now;}
                else if updated.ticks()<=now.ticks(){
                    while updated.ticks()<now.ticks(){n=n.checked_mul(2).ok_or("native interval division overflow")?;updated=previous.add(i.mul(n));}
                    n/=2;let mut m=n;updated=previous.add(i.mul(n));
                    while m>0 {let candidate=updated.add(i.mul(m));if candidate.ticks()<=now.ticks(){updated=candidate;n+=m;}m/=2;}
                    previous=updated;
                }
            }
        }
        Ok(Response {ringing,previous:previous.string(),next:previous.add(i).string()})
    }
}

/// The station file name GNU OPEN receives, as an OS path component.  On
/// Unix the bytes are the name (mpas_soundings.F:277); elsewhere the name
/// must be UTF-8, and a name that is not is refused rather than altered.
/// The Unix-only API sits under its cfg so the Windows build compiles
/// (the cargo gate refuses std::os::unix outside a cfg).
#[cfg(unix)]
fn native_path_component(bytes: &[u8]) -> io::Result<std::path::PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    Ok(std::path::PathBuf::from(std::ffi::OsStr::from_bytes(bytes)))
}

#[cfg(not(unix))]
fn native_path_component(bytes: &[u8]) -> io::Result<std::path::PathBuf> {
    std::str::from_utf8(bytes)
        .map(std::path::PathBuf::from)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "native station file name is not UTF-8"))
}
