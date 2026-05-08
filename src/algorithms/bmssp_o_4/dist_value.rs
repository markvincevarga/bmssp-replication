use std::cmp::Ordering;

const SANITIZE_SCALE: f64 = 1e10;

#[derive(Debug, Clone, Copy)]
pub struct DistValue {
    pub dist: f64,
    pub hops: u32,
    pub node: u32,
}

impl DistValue {
    #[inline(always)]
    pub fn new(dist: f64, hops: u32, node: u32) -> Self {
        Self { dist, hops, node }
    }
}

pub fn sanitize(w: f64) -> f64 {
    (w * SANITIZE_SCALE).round() / SANITIZE_SCALE
}

pub fn safe_infinity() -> f64 {
    f64::INFINITY / 10.0
}

impl PartialEq for DistValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for DistValue {}

impl PartialOrd for DistValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistValue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.hops.cmp(&other.hops))
            .then_with(|| self.node.cmp(&other.node))
    }
}
