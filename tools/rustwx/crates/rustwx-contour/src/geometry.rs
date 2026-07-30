#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LineSegment {
    pub start: Point2,
    pub end: Point2,
}

impl LineSegment {
    pub fn new(start: Point2, end: Point2) -> Self {
        Self { start, end }
    }

    pub fn length_squared(&self) -> f64 {
        let dx = self.end.x - self.start.x;
        let dy = self.end.y - self.start.y;
        dx * dx + dy * dy
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Ring {
    pub vertices: Vec<Point2>,
}

impl Ring {
    pub fn signed_area(&self) -> f64 {
        if self.vertices.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        for index in 0..self.vertices.len() {
            let current = self.vertices[index];
            let next = self.vertices[(index + 1) % self.vertices.len()];
            area += current.x * next.y - next.x * current.y;
        }
        area * 0.5
    }

    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Polygon {
    pub exterior: Ring,
    pub holes: Vec<Ring>,
}

impl Polygon {
    pub fn area(&self) -> f64 {
        let hole_area: f64 = self.holes.iter().map(Ring::area).sum();
        self.exterior.area() - hole_area
    }
}
