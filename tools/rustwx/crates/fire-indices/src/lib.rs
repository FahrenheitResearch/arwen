//! Daily Canadian FWI and KBDI equations, independently implemented in Rust.
//! References: Canadian Forestry Service FTR 35 (1987), FTR 33 (1985);
//! USDA Forest Service Research Paper SE-38 (1968).
//! Boundary units: Celsius, percent RH, km/hour and mm precipitation.
pub mod nfdrs;
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct FwiState {
    pub ffmc: f64,
    pub dmc: f64,
    pub dc: f64,
}
impl Default for FwiState {
    fn default() -> Self {
        Self {
            ffmc: 85.,
            dmc: 6.,
            dc: 15.,
        }
    }
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Fwi {
    pub ffmc: f64,
    pub dmc: f64,
    pub dc: f64,
    pub isi: f64,
    pub bui: f64,
    pub fwi: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct Weather {
    pub temperature: f64,
    pub humidity: f64,
    pub wind: f64,
    pub rain: f64,
    pub latitude: f64,
    pub month: u32,
}
fn day_length(lat: f64, month: u32) -> f64 {
    let n = [6.5, 7.5, 9., 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8., 7., 6.];
    let nl = [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8];
    let sl = [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2];
    let s = [
        11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10., 11.2, 11.8,
    ];
    let m = month as usize - 1;
    if lat >= 30. {
        n[m]
    } else if lat >= 10. {
        nl[m]
    } else if lat > -10. {
        9.
    } else if lat > -30. {
        sl[m]
    } else {
        s[m]
    }
}
fn dc_factor(lat: f64, month: u32) -> f64 {
    let n = [
        -1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5., 2.4, 0.4, -1.6, -1.6,
    ];
    let s = [
        6.4, 5., 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8,
    ];
    if lat > 20. {
        n[month as usize - 1]
    } else if lat < -20. {
        s[month as usize - 1]
    } else {
        1.4
    }
}
impl FwiState {
    pub fn validate(&self) -> Result<(), String> {
        if !self.ffmc.is_finite()
            || !(0. ..=101.).contains(&self.ffmc)
            || !self.dmc.is_finite()
            || self.dmc < 0.
            || !self.dc.is_finite()
            || self.dc < 0.
        {
            Err("invalid FFMC/DMC/DC carried state".into())
        } else {
            Ok(())
        }
    }
    pub fn step(&mut self, w: Weather) -> Result<Fwi, String> {
        self.validate()?;
        if ![w.temperature, w.humidity, w.wind, w.rain, w.latitude]
            .iter()
            .all(|v| v.is_finite())
            || !(-90. ..=65.).contains(&w.temperature)
            || !(0. ..=100.).contains(&w.humidity)
            || !(0. ..=540.).contains(&w.wind)
            || w.rain < 0.
            || w.latitude.abs() > 90.
            || !(1..=12).contains(&w.month)
        {
            return Err("invalid FWI weather or calendar".into());
        }
        let t = w.temperature;
        let h = w.humidity;
        let wind = w.wind;
        let mut m = 147.2 * (101. - self.ffmc) / (59.5 + self.ffmc);
        if w.rain > 0.5 {
            let r = w.rain - 0.5;
            let extra = if m > 150. {
                0.0015 * (m - 150.).powi(2) * r.sqrt()
            } else {
                0.
            };
            m = (m + 42.5 * r * (-100. / (251. - m)).exp() * (1. - (-6.93 / r).exp()) + extra)
                .min(250.);
        }
        let ed = 0.942 * h.powf(0.679)
            + 11. * ((h - 100.) / 10.).exp()
            + 0.18 * (21.1 - t) * (1. - (-0.115 * h).exp());
        let ew = 0.618 * h.powf(0.753)
            + 10. * ((h - 100.) / 10.).exp()
            + 0.18 * (21.1 - t) * (1. - (-0.115 * h).exp());
        if m > ed {
            let k = (0.424 * (1. - (h / 100.).powf(1.7))
                + 0.0694 * wind.sqrt() * (1. - (h / 100.).powi(8)))
                * 0.581
                * (0.0365 * t).exp();
            m = ed + (m - ed) * 10f64.powf(-k);
        } else if m < ew {
            let k = (0.424 * (1. - ((100. - h) / 100.).powf(1.7))
                + 0.0694 * wind.sqrt() * (1. - ((100. - h) / 100.).powi(8)))
                * 0.581
                * (0.0365 * t).exp();
            m = ew - (ew - m) * 10f64.powf(-k);
        }
        self.ffmc = (59.5 * (250. - m) / (147.2 + m)).clamp(0., 101.);
        if w.rain > 1.5 {
            let r = 0.92 * w.rain - 1.27;
            let mo = 20. + (5.6348 - self.dmc / 43.43).exp();
            let b = if self.dmc <= 33. {
                100. / (0.5 + 0.3 * self.dmc)
            } else if self.dmc <= 65. {
                14. - 1.3 * self.dmc.ln()
            } else {
                6.2 * self.dmc.ln() - 17.2
            };
            let mr = mo + 1000. * r / (48.77 + b * r);
            self.dmc = (244.72 - 43.43 * (mr - 20.).ln()).max(0.);
        }
        self.dmc +=
            1.894 * (t.max(-1.1) + 1.1) * (100. - h) * day_length(w.latitude, w.month) * 0.0001;
        if w.rain > 2.8 {
            let r = 0.83 * w.rain - 1.27;
            let q = 800. * (-self.dc / 400.).exp() + 3.937 * r;
            self.dc = (400. * (800. / q).ln()).max(0.);
        }
        self.dc += 0.5 * (0.36 * (t.max(-2.8) + 2.8) + dc_factor(w.latitude, w.month)).max(0.);
        let fm = 147.2 * (101. - self.ffmc) / (59.5 + self.ffmc);
        let isi = 0.208
            * (0.05039 * wind).exp()
            * 91.9
            * (-0.1386 * fm).exp()
            * (1. + fm.powf(5.31) / 49_300_000.);
        let d = self.dmc + 0.4 * self.dc;
        let bui = if d == 0. {
            0.
        } else if self.dmc <= 0.4 * self.dc {
            0.8 * self.dmc * self.dc / d
        } else {
            self.dmc - (1. - 0.8 * self.dc / d) * (0.92 + (0.0114 * self.dmc).powf(1.7))
        }
        .max(0.);
        let fd = if bui <= 80. {
            0.626 * bui.powf(0.809) + 2.
        } else {
            1000. / (25. + 108.64 * (-0.023 * bui).exp())
        };
        let b = 0.1 * isi * fd;
        let fwi = if b <= 1. {
            b
        } else {
            (2.72 * (0.434 * b.ln()).powf(0.647)).exp()
        };
        Ok(Fwi {
            ffmc: self.ffmc,
            dmc: self.dmc,
            dc: self.dc,
            isi,
            bui,
            fwi,
        })
    }
}
/// KBDI in hundredths of an inch, 0..800. Carry the 0.2 inch interception
/// threshold across consecutive wet days, rather than applying it every day.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct KbdiState {
    pub deficit: f64,
    pub wet_event_mm: f64,
}
impl KbdiState {
    pub fn validate(&self) -> Result<(), String> {
        if !self.deficit.is_finite()
            || !(0. ..=800.).contains(&self.deficit)
            || !self.wet_event_mm.is_finite()
            || self.wet_event_mm < 0.
        {
            Err("invalid carried KBDI state".into())
        } else {
            Ok(())
        }
    }
    pub fn step(&mut self, tmax: f64, rain: f64, annual: f64) -> Result<f64, String> {
        if ![self.deficit, self.wet_event_mm, tmax, rain, annual]
            .iter()
            .all(|v| v.is_finite())
            || !(0. ..=800.).contains(&self.deficit)
            || self.wet_event_mm < 0.
            || !(-90. ..=65.).contains(&tmax)
            || rain < 0.
            || annual <= 0.
        {
            return Err("invalid KBDI state, weather or annual rainfall".into());
        }
        let before = (self.wet_event_mm - 5.08).max(0.);
        self.wet_event_mm = if rain > 0. {
            self.wet_event_mm + rain
        } else {
            0.
        };
        let effective = if rain > 0. {
            (self.wet_event_mm - 5.08).max(0.) - before
        } else {
            0.
        };
        self.deficit = (self.deficit - effective * 100. / 25.4).max(0.);
        let tf = tmax * 1.8 + 32.;
        let dq = (800. - self.deficit) * (0.968 * (0.0486 * tf).exp() - 8.30) * 0.001
            / (1. + 10.88 * (-0.0441 * annual / 25.4).exp());
        self.deficit = (self.deficit + dq.max(0.)).clamp(0., 800.);
        Ok(self.deficit)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standard_equation_reference() {
        let v = FwiState::default()
            .step(Weather {
                temperature: 17.,
                humidity: 42.,
                wind: 25.,
                rain: 0.,
                latitude: 45.,
                month: 4,
            })
            .unwrap();
        for (a, b) in [
            (v.ffmc, 87.69298),
            (v.dmc, 8.54505),
            (v.dc, 19.014),
            (v.isi, 10.85366),
            (v.bui, 8.49043),
            (v.fwi, 10.09637),
        ] {
            assert!((a - b).abs() < 0.0001, "{a} != {b}");
        }
    }
    #[test]
    fn interception_is_per_event() {
        let mut x = KbdiState {
            deficit: 400.,
            wet_event_mm: 0.,
        };
        x.step(-20., 3., 1000.).unwrap();
        assert_eq!(x.deficit, 400.);
        x.step(-20., 3., 1000.).unwrap();
        assert!((x.deficit - (400. - 0.92 * 100. / 25.4)).abs() < 1e-10);
        x.step(-20., 0., 1000.).unwrap();
        assert_eq!(x.wet_event_mm, 0.);
    }
    #[test]
    fn wet_weather_and_hemispheres() {
        let mut a = FwiState {
            ffmc: 96.,
            dmc: 150.,
            dc: 600.,
        };
        let old = a;
        let v = a
            .step(Weather {
                temperature: 20.,
                humidity: 90.,
                wind: 2.,
                rain: 60.,
                latitude: -45.,
                month: 1,
            })
            .unwrap();
        assert!(v.ffmc < old.ffmc && v.dmc < old.dmc && v.dc < old.dc);
        assert_eq!(day_length(45., 1), day_length(-45., 7));
        assert_eq!(dc_factor(45., 7), dc_factor(-45., 1));
    }
    #[test]
    fn reject_before_mutation() {
        let mut a = FwiState::default();
        let before = a;
        assert!(
            a.step(Weather {
                temperature: 20.,
                humidity: 101.,
                wind: 2.,
                rain: 0.,
                latitude: 45.,
                month: 7
            })
            .is_err()
        );
        assert_eq!(a, before);
    }
}
