//! Weather station lookup by ID, nearest point, or radius search.

// ── Public types ────────────────────────────────────────────────────────

/// Metadata for a single weather observation station.
#[derive(Debug, Clone)]
pub struct StationInfo {
    /// Station identifier (e.g. "KATL", "72219").
    pub id: String,
    /// Human-readable station name.
    pub name: String,
    /// State or province code (may be empty for non-US stations).
    pub state: String,
    /// Country code (e.g. "US").
    pub country: String,
    /// Latitude in decimal degrees.
    pub latitude: f64,
    /// Longitude in decimal degrees.
    pub longitude: f64,
    /// Elevation in metres above mean sea level.
    pub elevation: f64,
}

/// In-memory station database with spatial lookup.
#[derive(Debug, Clone)]
pub struct StationLookup {
    stations: Vec<StationInfo>,
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Approximate great-circle distance in kilometres using the Haversine
/// formula.
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6371.0; // Earth mean radius in km
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let a = (d_lat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (d_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    R * c
}

// ── Public API ──────────────────────────────────────────────────────────

impl StationLookup {
    /// Create an empty lookup table.
    pub fn new() -> Self {
        Self {
            stations: Vec::new(),
        }
    }

    /// Create a lookup table from an existing vector of station entries.
    pub fn from_entries(entries: Vec<StationInfo>) -> Self {
        Self { stations: entries }
    }

    /// Return the number of stations in the table.
    pub fn len(&self) -> usize {
        self.stations.len()
    }

    /// Return true if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.stations.is_empty()
    }

    /// Add a station to the table.
    pub fn add(&mut self, station: StationInfo) {
        self.stations.push(station);
    }

    /// Look up a station by its ID (case-insensitive).
    pub fn lookup(&self, id: &str) -> Option<&StationInfo> {
        let upper = id.to_uppercase();
        self.stations.iter().find(|s| s.id.to_uppercase() == upper)
    }

    /// Find the station nearest to a given latitude/longitude.
    pub fn nearest(&self, lat: f64, lon: f64) -> Option<&StationInfo> {
        self.stations
            .iter()
            .min_by(|a, b| {
                let da = haversine_km(lat, lon, a.latitude, a.longitude);
                let db = haversine_km(lat, lon, b.latitude, b.longitude);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Find all stations within `radius_km` of a given point.
    ///
    /// Results are sorted by distance (nearest first).
    pub fn within_radius(&self, lat: f64, lon: f64, radius_km: f64) -> Vec<&StationInfo> {
        let mut results: Vec<(&StationInfo, f64)> = self
            .stations
            .iter()
            .filter_map(|s| {
                let d = haversine_km(lat, lon, s.latitude, s.longitude);
                if d <= radius_km {
                    Some((s, d))
                } else {
                    None
                }
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().map(|(s, _)| s).collect()
    }
}

impl Default for StationLookup {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stations() -> Vec<StationInfo> {
        vec![
            StationInfo {
                id: "KATL".into(),
                name: "Hartsfield-Jackson Atlanta Intl".into(),
                state: "GA".into(),
                country: "US".into(),
                latitude: 33.6407,
                longitude: -84.4277,
                elevation: 313.0,
            },
            StationInfo {
                id: "KORD".into(),
                name: "Chicago O'Hare Intl".into(),
                state: "IL".into(),
                country: "US".into(),
                latitude: 41.9742,
                longitude: -87.9073,
                elevation: 205.0,
            },
            StationInfo {
                id: "KJFK".into(),
                name: "John F Kennedy Intl".into(),
                state: "NY".into(),
                country: "US".into(),
                latitude: 40.6413,
                longitude: -73.7781,
                elevation: 4.0,
            },
            StationInfo {
                id: "EGLL".into(),
                name: "London Heathrow".into(),
                state: "".into(),
                country: "UK".into(),
                latitude: 51.4700,
                longitude: -0.4543,
                elevation: 25.0,
            },
        ]
    }

    #[test]
    fn lookup_by_id() {
        let db = StationLookup::from_entries(sample_stations());
        let s = db.lookup("KATL").unwrap();
        assert_eq!(s.name, "Hartsfield-Jackson Atlanta Intl");

        // Case-insensitive
        let s2 = db.lookup("katl").unwrap();
        assert_eq!(s2.id, "KATL");

        assert!(db.lookup("XXXX").is_none());
    }

    #[test]
    fn nearest_station() {
        let db = StationLookup::from_entries(sample_stations());

        // Point near Atlanta
        let s = db.nearest(33.75, -84.39).unwrap();
        assert_eq!(s.id, "KATL");

        // Point near Chicago
        let s = db.nearest(42.0, -87.9).unwrap();
        assert_eq!(s.id, "KORD");
    }

    #[test]
    fn within_radius_search() {
        let db = StationLookup::from_entries(sample_stations());

        // 50 km around Atlanta — should only get KATL
        let results = db.within_radius(33.75, -84.39, 50.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "KATL");

        // Very large radius — should get all 4
        let results = db.within_radius(40.0, -80.0, 10000.0);
        assert_eq!(results.len(), 4);

        // 0 radius — nothing
        let results = db.within_radius(33.75, -84.39, 0.001);
        assert!(results.is_empty());
    }

    #[test]
    fn within_radius_sorted_by_distance() {
        let db = StationLookup::from_entries(sample_stations());

        // Center on JFK — results should be JFK first, then farther away
        let results = db.within_radius(40.6, -73.8, 2000.0);
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "KJFK");
    }

    #[test]
    fn empty_lookup() {
        let db = StationLookup::new();
        assert!(db.is_empty());
        assert_eq!(db.len(), 0);
        assert!(db.lookup("KATL").is_none());
        assert!(db.nearest(33.0, -84.0).is_none());
        assert!(db.within_radius(33.0, -84.0, 100.0).is_empty());
    }

    #[test]
    fn add_station() {
        let mut db = StationLookup::new();
        db.add(StationInfo {
            id: "KDEN".into(),
            name: "Denver Intl".into(),
            state: "CO".into(),
            country: "US".into(),
            latitude: 39.8561,
            longitude: -104.6737,
            elevation: 1656.0,
        });
        assert_eq!(db.len(), 1);
        assert!(db.lookup("KDEN").is_some());
    }

    #[test]
    fn default_is_empty() {
        let db = StationLookup::default();
        assert!(db.is_empty());
    }

    #[test]
    fn haversine_sanity() {
        // Same point -> 0
        assert!((haversine_km(40.0, -80.0, 40.0, -80.0)).abs() < 0.001);

        // ATL to ORD ~ 975 km (known value)
        let d = haversine_km(33.6407, -84.4277, 41.9742, -87.9073);
        assert!(d > 900.0 && d < 1050.0, "ATL-ORD distance: {} km", d);
    }
}
