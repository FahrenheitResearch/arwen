//! Evidence pictures of a fabric (not weather fields): catchments filled in
//! muted colours with darker borders, flowpaths in blue with width by
//! Strahler order, lakes as cyan discs, domain outlets as red discs, and an
//! optional reference fabric drawn in orange on top. Pure Rust PNG writer.
use crate::compare::albers5070_inverse;
use crate::contract::Fabric;
use crate::geo;
use crate::Result;
use rayon::prelude::*;
use std::io::Write;

pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub px: Vec<[u8; 3]>,
    pub x0: f64,
    pub y1: f64,
    pub scale: f64,
}

impl Canvas {
    pub fn new(extent: [f64; 4], width: usize) -> Canvas {
        let (x0, y0, x1, y1) = (extent[0], extent[1], extent[2], extent[3]);
        let scale = width as f64 / (x1 - x0);
        let h = ((y1 - y0) * scale).ceil().max(1.0) as usize;
        Canvas { w: width, h, px: vec![[250, 250, 247]; width * h], x0, y1, scale }
    }
    fn to_px(&self, p: (f64, f64)) -> (f64, f64) {
        ((p.0 - self.x0) * self.scale, (self.y1 - p.1) * self.scale)
    }
    fn put(&mut self, x: i64, y: i64, c: [u8; 3]) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            self.px[y as usize * self.w + x as usize] = c;
        }
    }
    pub fn disc(&mut self, p: (f64, f64), r: f64, c: [u8; 3]) {
        let (cx, cy) = self.to_px(p);
        let ri = r.ceil() as i64;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                if (dx * dx + dy * dy) as f64 <= r * r {
                    self.put(cx as i64 + dx, cy as i64 + dy, c);
                }
            }
        }
    }
    pub fn line(&mut self, pts: &[(f64, f64)], width: f64, c: [u8; 3]) {
        for w in pts.windows(2) {
            let (a, b) = (self.to_px(w[0]), self.to_px(w[1]));
            let n = ((b.0 - a.0).abs().max((b.1 - a.1).abs()).ceil() as usize).max(1);
            for s in 0..=n {
                let t = s as f64 / n as f64;
                let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
                if width <= 1.0 {
                    self.put(x as i64, y as i64, c);
                } else {
                    let r = width / 2.0;
                    let ri = r.ceil() as i64;
                    for dy in -ri..=ri {
                        for dx in -ri..=ri {
                            if (dx * dx + dy * dy) as f64 <= r * r {
                                self.put(x as i64 + dx, y as i64 + dy, c);
                            }
                        }
                    }
                }
            }
        }
    }
    /// Even-odd scanline fill of polygons (rings in map units), one task per pixel row.
    pub fn fill_polygons(&mut self, polys: &[(Vec<Vec<(f64, f64)>>, [u8; 3])]) {
        let px: Vec<Vec<Vec<(f64, f64)>>> = polys.iter().map(|(rings, _)| rings.iter().map(|r| r.iter().map(|&p| self.to_px(p)).collect()).collect()).collect();
        let spans: Vec<(f64, f64)> = px.iter().map(|rings| {
            let ys = rings.iter().flatten().map(|p| p.1);
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            for y in ys {
                lo = lo.min(y);
                hi = hi.max(y);
            }
            (lo, hi)
        }).collect();
        let w = self.w;
        self.px.par_chunks_mut(w).enumerate().for_each(|(row, line)| {
            let y = row as f64 + 0.5;
            for (k, rings) in px.iter().enumerate() {
                if y < spans[k].0 || y > spans[k].1 {
                    continue;
                }
                let mut xs = Vec::new();
                for r in rings {
                    let n = r.len();
                    for i in 0..n {
                        let (a, b) = (r[i], r[(i + 1) % n]);
                        if (a.1 > y) != (b.1 > y) {
                            xs.push(a.0 + (y - a.1) * (b.0 - a.0) / (b.1 - a.1));
                        }
                    }
                }
                xs.sort_by(|a, b| a.total_cmp(b));
                for pair in xs.chunks(2) {
                    if pair.len() == 2 {
                        let (x0, x1) = ((pair[0] - 0.5).ceil().max(0.0) as usize, ((pair[1] - 0.5).floor().min(w as f64 - 1.0)).max(-1.0) as i64);
                        for x in x0 as i64..=x1 {
                            line[x as usize] = polys[k].1;
                        }
                    }
                }
            }
        });
    }
    pub fn write_png(&self, path: &str) -> Result<()> {
        let mut raw = Vec::with_capacity((self.w * 3 + 1) * self.h);
        for r in 0..self.h {
            raw.push(0u8);
            for p in &self.px[r * self.w..(r + 1) * self.w] {
                raw.extend_from_slice(p);
            }
        }
        let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        z.write_all(&raw)?;
        let idat = z.finish()?;
        let mut out = Vec::new();
        out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
        let chunk = |out: &mut Vec<u8>, ty: &[u8], data: &[u8]| {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            let mut c = crc32fast::Hasher::new();
            c.update(ty);
            c.update(data);
            out.extend_from_slice(ty);
            out.extend_from_slice(data);
            out.extend_from_slice(&c.finalize().to_be_bytes());
        };
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&(self.w as u32).to_be_bytes());
        ihdr.extend_from_slice(&(self.h as u32).to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        chunk(&mut out, b"IHDR", &ihdr);
        chunk(&mut out, b"IDAT", &idat);
        chunk(&mut out, b"IEND", &[]);
        std::fs::write(path, out)?;
        Ok(())
    }
}

fn pastel(n: u64) -> [u8; 3] {
    let mut x = n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    let h = (x % 360) as f64;
    let (s, v) = (0.28, 0.93);
    let c = v * s;
    let hp = h / 60.0;
    let xx = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, xx, 0.0),
        1 => (xx, c, 0.0),
        2 => (0.0, c, xx),
        3 => (0.0, xx, c),
        4 => (xx, 0.0, c),
        _ => (c, 0.0, xx),
    };
    let m = v - c;
    [((r + m) * 255.0) as u8, ((g + m) * 255.0) as u8, ((b + m) * 255.0) as u8]
}

/// Render `f` (its own projected CRS). `extent` in that CRS, or None for the
/// divides' envelope. `reference` (EPSG:5070) is reprojected through
/// lon/lat with `to_ours`.
pub fn render(f: &Fabric, extent: Option<[f64; 4]>, width: usize, reference: Option<(&Fabric, &dyn Fn(f64, f64) -> (f64, f64))>, path: &str) -> Result<serde_json::Value> {
    let polys: Vec<(String, Vec<Vec<(f64, f64)>>)> = f.divides.iter().filter_map(|d| Some((d.divide_id.clone()?, geo::gpkg_rings(&d.geom.as_ref()?.0)?))).collect();
    let ext = extent.unwrap_or_else(|| {
        let mut e = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for (_, rings) in &polys {
            for &(x, y) in rings.iter().flatten() {
                e[0] = e[0].min(x);
                e[1] = e[1].min(y);
                e[2] = e[2].max(x);
                e[3] = e[3].max(y);
            }
        }
        e
    });
    let mut cv = Canvas::new(ext, width);
    let num = |s: &str| s.rsplit('-').next().and_then(|n| n.parse::<u64>().ok()).unwrap_or(0);
    let fills: Vec<(Vec<Vec<(f64, f64)>>, [u8; 3])> = polys.iter().map(|(id, r)| (r.clone(), pastel(num(id)))).collect();
    cv.fill_polygons(&fills);
    let scale_px_per_km = cv.scale * 1000.0;
    if scale_px_per_km > 0.08 {
        for (_, rings) in &polys {
            for r in rings {
                let mut rr = r.clone();
                if rr.first() != rr.last() {
                    rr.push(rr[0]);
                }
                cv.line(&rr, 1.0, [170, 170, 165]);
            }
        }
    }
    let mut drawn = 0usize;
    let mut fps: Vec<(f64, Vec<(f64, f64)>)> = f.flowpaths.iter().filter_map(|x| Some((x.order?, geo::gpkg_rings(&x.geom.as_ref()?.0)?.into_iter().next()?))).collect();
    fps.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (o, l) in &fps {
        let w = (0.6 + 0.55 * (o - 1.0)).clamp(1.0, 6.0) * (width as f64 / 1600.0).max(0.6);
        cv.line(l, w, [30, 90, 200]);
        drawn += 1;
    }
    if let Some((rf, to_ours)) = reference {
        for x in &rf.flowpaths {
            if let Some(l) = x.geom.as_ref().and_then(|g| geo::gpkg_rings(&g.0)).and_then(|r| r.into_iter().next()) {
                let pts: Vec<(f64, f64)> = l.iter().map(|&(a, b)| {
                    let (lo, la) = albers5070_inverse(a, b);
                    to_ours(lo, la)
                }).collect();
                let w = (0.4 + 0.45 * (x.order.unwrap_or(1.0) - 1.0)).clamp(1.0, 4.0);
                cv.line(&pts, w, [235, 120, 20]);
            }
        }
    }
    let r_disc = (width as f64 / 400.0).max(2.5);
    for l in &f.lakes {
        if let Some(p) = l.geom.as_ref().and_then(|g| geo::gpkg_rings(&g.0)).and_then(|r| r.first()?.first().copied()) {
            cv.disc(p, r_disc, [0, 190, 210]);
        }
    }
    let mut outlets = 0;
    for n in &f.nexus {
        if n.id.as_deref().is_some_and(|s| s.starts_with("tnx-")) {
            if let Some(p) = n.geom.as_ref().and_then(|g| geo::gpkg_rings(&g.0)).and_then(|r| r.first()?.first().copied()) {
                cv.disc(p, r_disc * 0.6, [210, 30, 30]);
                outlets += 1;
            }
        }
    }
    cv.write_png(path)?;
    Ok(serde_json::json!({"png": path, "width": cv.w, "height": cv.h, "extent": ext, "divides": polys.len(), "flowpaths": drawn, "lakes": f.lakes.len(), "outlets": outlets}))
}
