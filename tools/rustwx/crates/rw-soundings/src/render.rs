//! Headless rendering of the sharppyrs SPC sounding window (and the
//! skew-T-only widget) through egui_kittest's wgpu renderer, plus the
//! ensemble overlays sharppyrs does not have yet (member spaghetti and a
//! percentile envelope on the skew-T) and a time-height panel.
//!
//! Each render thread owns one Harness (egui contexts are not Send).

use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;

use egui::{Align2, Color32, FontId, Pos2, Rect, Shape, Stroke, Vec2};
use sharppyrs::{
    CornerPanel, DiagnosticTablePanelKind, NativeDiagnosticPatchBoard, ParcelType, SkewTStyle,
    SoundingLayout,
};

use crate::analysis::Analysis;

#[derive(Clone, Debug)]
pub enum ViewKind {
    Window {
        omega: Option<bool>,
        interp_winds: bool,
    },
    SkewT {
        omega: Option<bool>,
        interp_winds: bool,
    },
}

#[derive(Clone, Debug, Default)]
pub struct MemberTrace {
    pub label: String,
    pub p: Vec<f64>,
    pub t: Vec<f64>,
    pub td: Vec<f64>,
    /// hodograph (u, v) kt, 0-10 km AGL
    pub hodo: Vec<(f64, f64)>,
}

/// Member percentiles per level: index 0..5 = p10, p25, p50, p75, p90.
#[derive(Clone, Debug, Default)]
pub struct Bands {
    pub p: Vec<f64>,
    pub t: [Vec<f64>; 5],
    pub td: [Vec<f64>; 5],
    /// median hodograph (u, v) kt, 0-10 km AGL
    pub hodo_med: Vec<(f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct Overlay {
    pub members: Vec<MemberTrace>,
    pub bands: Option<Bands>,
    /// envelope mode: p10-p90 and p25-p75 bands; otherwise only p10-p90
    pub inner_band: bool,
    pub show_t: bool,
    pub show_td: bool,
    pub alpha: u8,
    pub legend: String,
}

pub struct SoundingScene {
    pub analysis: Arc<Analysis>,
    pub view: ViewKind,
    pub layout: SoundingLayout,
    pub corner: CornerPanel,
    pub parcel: ParcelType,
    pub title: String,
    pub brand: String,
    pub style: SkewTStyle,
    pub patches: NativeDiagnosticPatchBoard,
    pub overlay: Option<Overlay>,
}

pub struct TimeHeightScene {
    pub title: String,
    pub subtitle: String,
    pub leads_min: Vec<u32>,
    pub z_km: Vec<f64>,
    /// [lead][z]
    pub field: Vec<Vec<f64>>,
    pub field_label: String,
    pub ramp: Ramp,
    /// (lead index, z km, wdir, wspd kt)
    pub barbs: Vec<(usize, f64, f64, f64)>,
    /// named height lines per lead (km AGL), e.g. ML LCL, MU LFC, effective inflow
    pub lines: Vec<(String, Color32, Vec<f64>)>,
    /// index strip rows: (label, per-lead values)
    pub strip: Vec<(String, Vec<f64>)>,
    pub style: SkewTStyle,
}

#[derive(Clone, Debug)]
pub struct Ramp {
    pub stops: Vec<(f64, Color32)>,
}

impl Ramp {
    pub fn color(&self, v: f64) -> Color32 {
        if !v.is_finite() || self.stops.is_empty() {
            return Color32::TRANSPARENT;
        }
        let s = &self.stops;
        if v <= s[0].0 {
            return s[0].1;
        }
        for w in s.windows(2) {
            if v <= w[1].0 {
                let a = ((v - w[0].0) / (w[1].0 - w[0].0)).clamp(0.0, 1.0) as f32;
                let (c0, c1) = (w[0].1, w[1].1);
                let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * a).round() as u8;
                return Color32::from_rgb(l(c0.r(), c1.r()), l(c0.g(), c1.g()), l(c0.b(), c1.b()));
            }
        }
        s[s.len() - 1].1
    }
}

pub enum Job {
    Sounding(SoundingScene),
    TimeHeight(TimeHeightScene),
}

pub struct RenderRequest {
    pub job: Arc<Job>,
    pub size: Vec2,
    pub ppp: f32,
    pub reply: tokio::sync::oneshot::Sender<Result<Rendered, String>>,
}

pub struct Rendered {
    pub png: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub frame_ms: f64,
    pub encode_ms: f64,
}

struct RState {
    fonts: bool,
    job: Option<Arc<Job>>,
}

/// Own the render workers until their GPU contexts have shut down. Detached
/// workers raced NVIDIA's process teardown after the last PNG was written.
pub struct RenderPool {
    sender: Option<Sender<RenderRequest>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl RenderPool {
    pub fn send(
        &self,
        request: RenderRequest,
    ) -> Result<(), std::sync::mpsc::SendError<RenderRequest>> {
        self.sender.as_ref().unwrap().send(request)
    }
}

impl Drop for RenderPool {
    fn drop(&mut self) {
        self.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Spawn `n` render threads sharing one queue.
pub fn spawn_pool(n: usize) -> RenderPool {
    let (tx, rx) = channel::<RenderRequest>();
    let rx = Arc::new(std::sync::Mutex::new(rx));
    let mut workers = Vec::new();
    for i in 0..n.max(1) {
        let rx = rx.clone();
        workers.push(
            std::thread::Builder::new()
                .name(format!("sharppyrs-render-{i}"))
                .spawn(move || render_thread(rx))
                .expect("spawn render thread"),
        );
    }
    RenderPool {
        sender: Some(tx),
        workers,
    }
}

fn build_harness(ppp: f32) -> egui_kittest::Harness<'static, RState> {
    let mut h = egui_kittest::Harness::builder()
        .with_size(Vec2::new(1630.0, 1100.0))
        .with_pixels_per_point(ppp)
        .with_max_steps(8)
        // The snapshot harness defaults to preferring a CPU rasterizer. This
        // public renderer uses wgpu's production adapter selection instead:
        // high-performance GPU first, software when that is all that exists.
        .wgpu_setup(egui_wgpu::WgpuSetup::CreateNew(
            egui_wgpu::WgpuSetupCreateNew::without_display_handle(),
        ))
        .build_ui_state(
            |ui: &mut egui::Ui, st: &mut RState| {
                if !st.fonts {
                    sharppyrs::install_fonts(ui.ctx());
                    st.fonts = true;
                    return;
                }
                if let Some(job) = st.job.clone() {
                    draw_job(ui, &job);
                }
            },
            RState {
                fonts: false,
                job: None,
            },
        );
    h.run_steps(2);
    h
}

/// One Harness per pixels-per-point value (switching ppp on a live context
/// only lands a frame late, which produced wrong-size images).
fn render_thread(rx: Arc<std::sync::Mutex<Receiver<RenderRequest>>>) {
    let mut harnesses: std::collections::HashMap<u32, egui_kittest::Harness<'static, RState>> =
        std::collections::HashMap::new();
    harnesses.insert(100, build_harness(1.0));
    loop {
        let req = {
            let guard = rx.lock().unwrap();
            match guard.recv() {
                Ok(r) => r,
                Err(_) => return,
            }
        };
        let t0 = Instant::now();
        let key = ((req.ppp * 4.0).round() * 25.0) as u32;
        let h = harnesses
            .entry(key)
            .or_insert_with(|| build_harness(key as f32 / 100.0));
        h.state_mut().job = Some(req.job.clone());
        h.set_size(req.size);
        h.run_steps(2);
        let out = h.render();
        h.state_mut().job = None;
        let frame_ms = t0.elapsed().as_secs_f64() * 1e3;
        let res = out.and_then(|img| {
            let t1 = Instant::now();
            let (w, hh) = (img.width(), img.height());
            let mut buf = std::io::Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)
                .map_err(|e| e.to_string())?;
            Ok(Rendered {
                png: buf.into_inner(),
                width: w,
                height: hh,
                frame_ms,
                encode_ms: t1.elapsed().as_secs_f64() * 1e3,
            })
        });
        let _ = req.reply.send(res);
    }
}

fn draw_job(ui: &mut egui::Ui, job: &Job) {
    // cover the harness panel margin with the sounding background
    let bg = match job {
        Job::Sounding(s) => s.style.bg_color,
        Job::TimeHeight(t) => t.style.bg_color,
    };
    let screen = ui.ctx().content_rect();
    ui.ctx()
        .layer_painter(egui::LayerId::background())
        .rect_filled(screen, 0.0, bg);
    match job {
        Job::Sounding(s) => draw_sounding(ui, s),
        Job::TimeHeight(t) => draw_time_height(ui, t),
    }
}

fn draw_sounding(ui: &mut egui::Ui, s: &SoundingScene) {
    let size = ui.available_size();
    let a = &s.analysis;
    let layout_id = egui::Id::new("gpuwm-sharppyrs-layout");
    let resp = egui::Frame::new()
        .fill(s.style.bg_color)
        .show(ui, |ui| {
            ui.set_min_size(size);
            match &s.view {
                ViewKind::Window {
                    omega,
                    interp_winds,
                } => {
                    sharppyrs::store_layout(ui.ctx(), layout_id, &s.layout);
                    let mut view = sharppyrs::SoundingView::new(&a.prof, &a.derived)
                        .title(s.title.clone())
                        .brand(s.brand.clone())
                        .style(s.style.clone())
                        .parcel(s.parcel)
                        .corner_panel(s.corner)
                        .interactive(false)
                        .layout_memory_id(layout_id)
                        .interp_winds(*interp_winds)
                        .size(size);
                    if let Some(o) = omega {
                        view = view.plot_omega(*o);
                    }
                    let view = if s.patches.patches.is_empty() {
                        view
                    } else {
                        view.native_diagnostic_patches(&s.patches)
                    };
                    ui.add(view)
                }
                ViewKind::SkewT {
                    omega,
                    interp_winds,
                } => {
                    let mut w = sharppyrs::SkewT::new(&a.prof)
                        .title(s.title.clone())
                        .brand(s.brand.clone())
                        .style(s.style.clone())
                        .parcel(s.parcel)
                        .cursor_readout(false)
                        .interp_winds(*interp_winds)
                        .size(size);
                    if let Some(o) = omega {
                        w = w.plot_omega(*o);
                    }
                    ui.add(w)
                }
            }
        })
        .inner;
    if let Some(ov) = &s.overlay {
        let rect = resp.rect;
        let (skew, hodo) = match s.view {
            ViewKind::Window { .. } => {
                let top = s.layout.top_height_fraction.clamp(0.40, 0.85);
                let sw = s.layout.skew_width_fraction.clamp(0.30, 0.70);
                (
                    Rect::from_min_max(
                        rect.min,
                        Pos2::new(
                            rect.min.x + rect.width() * sw,
                            rect.min.y + rect.height() * top,
                        ),
                    ),
                    hodo_rect(&s.layout, rect),
                )
            }
            ViewKind::SkewT { .. } => (rect, None),
        };
        draw_overlay(ui, skew, ov, &s.style);
        if let Some(h) = hodo {
            draw_hodo_overlay(ui, h, s.layout.hodo_zoom_kts, ov);
        }
    }
}

/// Port of the sharppyrs skew-T `Geom` transform (SHARPpy backgroundSkewT):
/// the same padding, pressure range and 100/3-degree skew, so overlays land
/// on the widget's own coordinates.
struct Geom {
    rect: Rect,
    lpad: f64,
    tpad: f64,
    brx: f64,
    bry: f64,
    yrange: f64,
}

impl Geom {
    fn new(rect: Rect) -> Self {
        let (w, h) = (rect.width() as f64, rect.height() as f64);
        Geom {
            rect,
            lpad: 46.0,
            tpad: 20.0,
            brx: w - 65.0,
            bry: h - 20.0,
            yrange: (100.0f64 / 3.0).to_radians().tan() * 100.0,
        }
    }
    fn py(&self, p: f64) -> f64 {
        let (lmax, lmin) = (1050.0f64.ln(), 100.0f64.ln());
        self.bry - ((lmax - p.ln()) / (lmax - lmin)) * (self.bry - self.tpad)
    }
    fn tx(&self, t: f64, p: f64) -> f64 {
        let scl1 = 50.0 - ((self.bry - self.py(p)) / (self.bry - self.tpad)) * self.yrange;
        self.brx - ((scl1 - t) / 100.0) * (self.brx - self.lpad)
    }
    fn pt(&self, t: f64, p: f64) -> Pos2 {
        Pos2::new(
            self.rect.min.x + self.tx(t, p) as f32,
            self.rect.min.y + self.py(p) as f32,
        )
    }
    fn clip(&self) -> Rect {
        Rect::from_min_max(
            Pos2::new(
                self.rect.min.x + self.lpad as f32,
                self.rect.min.y + self.tpad as f32,
            ),
            Pos2::new(
                self.rect.min.x + self.brx as f32,
                self.rect.min.y + self.bry as f32,
            ),
        )
    }
}

fn norm<const N: usize>(f: &[f32; N]) -> [f32; N] {
    let t: f32 = f.iter().map(|x| x.max(0.0)).sum();
    if t <= f32::EPSILON {
        [1.0 / N as f32; N]
    } else {
        std::array::from_fn(|i| f[i].max(0.0) / t)
    }
}

fn split_h<const N: usize>(band: Rect, f: &[f32; N]) -> [Rect; N] {
    let f = norm(f);
    let mut x = band.min.x;
    std::array::from_fn(|i| {
        let x0 = x;
        x = if i == N - 1 {
            band.max.x
        } else {
            x + band.width() * f[i]
        };
        Rect::from_min_max(Pos2::new(x0, band.min.y), Pos2::new(x, band.max.y))
    })
}

/// Where sharppyrs puts the hodograph cell (same track arithmetic as
/// `SoundingView`: 16 px brand strip, right columns, main/inset split).
/// Only the upper-right cells are mapped; a hodograph moved into the bottom
/// band gets no member overlay.
fn hodo_rect(l: &SoundingLayout, rect: Rect) -> Option<Rect> {
    use sharppyrs::PanelKind::Hodograph;
    let band_top = rect.min.y + rect.height() * l.top_height_fraction.clamp(0.40, 0.85);
    let skew_right = rect.min.x + rect.width() * l.skew_width_fraction.clamp(0.30, 0.70);
    let g2 = Rect::from_min_max(
        Pos2::new(skew_right, rect.min.y + 16.0),
        Pos2::new(rect.max.x, band_top),
    );
    let cols = split_h(g2, &l.right_column_fractions);
    let rc = cols[2];
    let main_bottom = rc.min.y + rc.height() * l.right_main_height_fraction.clamp(0.35, 0.85);
    let main = Rect::from_min_max(rc.min, Pos2::new(rc.max.x, main_bottom));
    let insets = split_h(
        Rect::from_min_max(Pos2::new(rc.min.x, main_bottom), rc.max),
        &l.inset_column_fractions,
    );
    if l.main == Hodograph {
        return Some(main);
    }
    for i in 0..4 {
        if l.insets[i] == Hodograph {
            return Some(insets[i]);
        }
    }
    for i in 0..2 {
        if l.strips[i] == Hodograph {
            return Some(cols[i]);
        }
    }
    None
}

fn draw_hodo_overlay(ui: &mut egui::Ui, r: Rect, zoom_kts: f64, ov: &Overlay) {
    let painter = ui.painter().with_clip_rect(r);
    let scale = r.width() as f64 / zoom_kts.clamp(80.0, 500.0);
    let (cx, cy) = (r.width() as f64 / 2.0, r.height() as f64 / 2.0);
    let pt = |u: f64, v: f64| {
        Pos2::new(
            r.min.x + (cx + u * scale) as f32,
            r.min.y + (cy - v * scale) as f32,
        )
    };
    let line = |pts: &Vec<(f64, f64)>, c: Color32, w: f32| {
        let v: Vec<Pos2> = pts
            .iter()
            .filter(|(u, v)| u.is_finite() && v.is_finite())
            .map(|(u, v)| pt(*u, *v))
            .collect();
        if v.len() > 1 {
            painter.add(Shape::line(v, Stroke::new(w, c)));
        }
    };
    for m in &ov.members {
        line(
            &m.hodo,
            Color32::from_rgba_unmultiplied(220, 220, 220, ov.alpha),
            1.0,
        );
    }
    if let Some(b) = &ov.bands
        && ov.members.is_empty()
    {
        line(
            &b.hodo_med,
            Color32::from_rgba_unmultiplied(255, 255, 255, 200),
            2.0,
        );
    }
}

fn draw_overlay(ui: &mut egui::Ui, skew: Rect, ov: &Overlay, st: &SkewTStyle) {
    let g = Geom::new(skew);
    let painter = ui.painter().with_clip_rect(g.clip());
    let ok = |p: f64, v: f64| p.is_finite() && v.is_finite() && p > 100.0;
    if let Some(b) = &ov.bands {
        let band = |lo: &Vec<f64>, hi: &Vec<f64>, c: Color32| {
            for k in 0..b.p.len().saturating_sub(1) {
                let (p0, p1) = (b.p[k], b.p[k + 1]);
                if !(ok(p0, lo[k])
                    && ok(p1, lo[k + 1])
                    && hi[k].is_finite()
                    && hi[k + 1].is_finite())
                {
                    continue;
                }
                let quad = vec![
                    g.pt(lo[k], p0),
                    g.pt(lo[k + 1], p1),
                    g.pt(hi[k + 1], p1),
                    g.pt(hi[k], p0),
                ];
                painter.add(Shape::convex_polygon(quad, c, Stroke::NONE));
            }
        };
        if ov.show_t {
            band(
                &b.t[0],
                &b.t[4],
                Color32::from_rgba_unmultiplied(255, 70, 70, 55),
            );
            if ov.inner_band {
                band(
                    &b.t[1],
                    &b.t[3],
                    Color32::from_rgba_unmultiplied(255, 70, 70, 75),
                );
            }
        }
        if ov.show_td {
            band(
                &b.td[0],
                &b.td[4],
                Color32::from_rgba_unmultiplied(70, 230, 90, 55),
            );
            if ov.inner_band {
                band(
                    &b.td[1],
                    &b.td[3],
                    Color32::from_rgba_unmultiplied(70, 230, 90, 75),
                );
            }
        }
    }
    let line = |p: &Vec<f64>, v: &Vec<f64>, c: Color32, w: f32| {
        let pts: Vec<Pos2> = p
            .iter()
            .zip(v)
            .filter(|(p, v)| ok(**p, **v))
            .map(|(p, v)| g.pt(*v, *p))
            .collect();
        if pts.len() > 1 {
            painter.add(Shape::line(pts, Stroke::new(w, c)));
        }
    };
    for m in &ov.members {
        if ov.show_t {
            line(
                &m.p,
                &m.t,
                Color32::from_rgba_unmultiplied(255, 90, 90, ov.alpha),
                1.0,
            );
        }
        if ov.show_td {
            line(
                &m.p,
                &m.td,
                Color32::from_rgba_unmultiplied(90, 235, 110, ov.alpha),
                1.0,
            );
        }
    }
    if let Some(b) = &ov.bands {
        if ov.show_t {
            line(
                &b.p,
                &b.t[2],
                Color32::from_rgba_unmultiplied(255, 210, 210, 170),
                1.0,
            );
        }
        if ov.show_td {
            line(
                &b.p,
                &b.td[2],
                Color32::from_rgba_unmultiplied(210, 255, 210, 170),
                1.0,
            );
        }
    }
    if !ov.legend.is_empty() {
        let c = g.clip();
        ui.painter().text(
            Pos2::new(c.max.x - 6.0, c.min.y + 4.0),
            Align2::RIGHT_TOP,
            &ov.legend,
            st.regular_font(12.0),
            st.fg_color,
        );
    }
}

fn draw_time_height(ui: &mut egui::Ui, s: &TimeHeightScene) {
    let size = ui.available_size();
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let p = ui.painter_at(rect);
    let st = &s.style;
    p.rect_filled(rect, 0.0, st.bg_color);
    let font = |sz: f32| st.regular_font(sz);
    let strip_h = 18.0 * s.strip.len() as f32;
    let plot = Rect::from_min_max(
        Pos2::new(rect.min.x + 70.0, rect.min.y + 56.0),
        Pos2::new(rect.max.x - 110.0, rect.max.y - 46.0 - strip_h),
    );
    p.text(
        Pos2::new(rect.min.x + 12.0, rect.min.y + 8.0),
        Align2::LEFT_TOP,
        &s.title,
        st.bold_font(18.0),
        st.fg_color,
    );
    p.text(
        Pos2::new(rect.min.x + 12.0, rect.min.y + 32.0),
        Align2::LEFT_TOP,
        &s.subtitle,
        font(13.0),
        st.fg_color,
    );
    let nl = s.leads_min.len();
    if nl == 0 || s.z_km.len() < 2 {
        return;
    }
    let zmax = *s.z_km.last().unwrap();
    let col_w = plot.width() / nl as f32;
    let xc = |i: usize| plot.min.x + col_w * (i as f32 + 0.5);
    let yz = |z: f64| plot.max.y - (z / zmax) as f32 * plot.height();
    // shaded cells
    for (i, col) in s.field.iter().enumerate() {
        for k in 0..s.z_km.len().saturating_sub(1) {
            let c = s.ramp.color(col[k]);
            if c == Color32::TRANSPARENT {
                continue;
            }
            let r = Rect::from_min_max(
                Pos2::new(plot.min.x + col_w * i as f32, yz(s.z_km[k + 1])),
                Pos2::new(plot.min.x + col_w * (i + 1) as f32, yz(s.z_km[k])),
            );
            p.rect_filled(r.expand(0.3), 0.0, c);
        }
    }
    // height lines
    for (name, c, vals) in &s.lines {
        let pts: Vec<Pos2> = vals
            .iter()
            .enumerate()
            .filter(|(_, z)| z.is_finite() && **z <= zmax)
            .map(|(i, z)| Pos2::new(xc(i), yz(*z)))
            .collect();
        if pts.len() > 1 {
            p.add(Shape::line(pts.clone(), Stroke::new(2.0, *c)));
        }
        for q in &pts {
            p.circle_filled(*q, 3.0, *c);
        }
        if let Some(last) = pts.last() {
            p.text(
                Pos2::new(last.x + 6.0, last.y),
                Align2::LEFT_CENTER,
                name,
                font(11.0),
                *c,
            );
        }
    }
    // barbs
    for (i, z, d, sp) in &s.barbs {
        if *z <= zmax {
            sharppyrs::barbs::draw_barb(&p, Pos2::new(xc(*i), yz(*z)), *d, *sp, false);
        }
    }
    // axes
    let ax = Stroke::new(1.0, st.fg_color);
    p.rect_stroke(plot, 0.0, ax, egui::StrokeKind::Inside);
    let mut z = 0.0;
    let step = if zmax > 8.0 { 1.0 } else { 0.5 };
    while z <= zmax + 1e-9 {
        let y = yz(z);
        p.line_segment(
            [Pos2::new(plot.min.x - 5.0, y), Pos2::new(plot.min.x, y)],
            ax,
        );
        p.text(
            Pos2::new(plot.min.x - 8.0, y),
            Align2::RIGHT_CENTER,
            format!("{z:.1}"),
            font(12.0),
            st.fg_color,
        );
        z += step;
    }
    p.text(
        Pos2::new(rect.min.x + 10.0, plot.center().y),
        Align2::LEFT_CENTER,
        "km\nAGL",
        font(12.0),
        st.fg_color,
    );
    for (i, l) in s.leads_min.iter().enumerate() {
        p.text(
            Pos2::new(xc(i), plot.max.y + 6.0),
            Align2::CENTER_TOP,
            format!("+{l}"),
            font(12.0),
            st.fg_color,
        );
    }
    p.text(
        Pos2::new(plot.center().x, plot.max.y + 24.0),
        Align2::CENTER_TOP,
        "lead (min)",
        font(12.0),
        st.fg_color,
    );
    // index strip
    for (r, (label, vals)) in s.strip.iter().enumerate() {
        let y = plot.max.y + 46.0 + 18.0 * r as f32;
        p.text(
            Pos2::new(plot.min.x - 8.0, y),
            Align2::RIGHT_TOP,
            label,
            font(12.0),
            st.fg_color,
        );
        for (i, v) in vals.iter().enumerate() {
            let txt = if !v.is_finite() {
                "--".to_string()
            } else if v.abs() >= 100.0 {
                format!("{v:.0}")
            } else {
                format!("{v:.1}")
            };
            p.text(
                Pos2::new(xc(i), y),
                Align2::CENTER_TOP,
                txt,
                font(12.0),
                st.fg_color,
            );
        }
    }
    // colour bar
    let cb = Rect::from_min_max(
        Pos2::new(plot.max.x + 40.0, plot.min.y),
        Pos2::new(plot.max.x + 58.0, plot.max.y),
    );
    let (v0, v1) = (
        s.ramp.stops.first().map(|x| x.0).unwrap_or(0.0),
        s.ramp.stops.last().map(|x| x.0).unwrap_or(1.0),
    );
    let n = 100;
    for j in 0..n {
        let a = j as f64 / n as f64;
        let v = v0 + (v1 - v0) * a;
        let y0 = cb.max.y - cb.height() * (j as f32 / n as f32);
        let y1 = cb.max.y - cb.height() * ((j + 1) as f32 / n as f32);
        p.rect_filled(
            Rect::from_min_max(Pos2::new(cb.min.x, y1), Pos2::new(cb.max.x, y0 + 0.5)),
            0.0,
            s.ramp.color(v),
        );
    }
    p.rect_stroke(cb, 0.0, ax, egui::StrokeKind::Inside);
    for (v, _) in &s.ramp.stops {
        let y = cb.max.y - cb.height() * ((v - v0) / (v1 - v0)) as f32;
        p.text(
            Pos2::new(cb.max.x + 4.0, y),
            Align2::LEFT_CENTER,
            format!("{v:.0}"),
            font(11.0),
            st.fg_color,
        );
    }
    p.text(
        Pos2::new(cb.center().x, cb.min.y - 6.0),
        Align2::CENTER_BOTTOM,
        &s.field_label,
        font(12.0),
        st.fg_color,
    );
    let _ = FontId::default();
}

/// Patches every rendered window carries: the two ECAPE cells are blanked,
/// always (Drew: never ECAPE).
pub fn base_patches() -> Vec<sharppyrs::NativeDiagnosticPatch> {
    vec![
        sharppyrs::NativeDiagnosticPatch::blank(DiagnosticTablePanelKind::Severe, "severe.ecape"),
        sharppyrs::NativeDiagnosticPatch::blank(DiagnosticTablePanelKind::Severe, "severe.ncape"),
    ]
}
