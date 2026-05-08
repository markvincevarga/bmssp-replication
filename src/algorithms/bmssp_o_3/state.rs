use super::dist_value::{safe_infinity, sanitize, DistValue};

#[repr(C)]
pub struct NodeRelax {
    pub dist: f64,
    pub hops: u32,
    pub pred: u32,
}

#[repr(C)]
pub struct NodePivot {
    pub root: u32,
    pub tree_size: u32,
}

pub struct BmsspState {
    pub relax_data: Vec<NodeRelax>,
    pub pivot_data: Vec<NodePivot>,
    pub k: usize,
    pub t: usize,
    pub last_complete_level: Vec<i16>,
}

impl BmsspState {
    pub fn new(n: usize, source: u32) -> Self {
        let log_n = if n > 1 { (n as f64).log2() } else { 1.0 };
        let k = (log_n.powf(1.0 / 3.0)).floor().max(1.0) as usize;
        let t = (log_n.powf(2.0 / 3.0)).floor().max(1.0) as usize;

        let mut relax_data: Vec<NodeRelax> = (0..n)
            .map(|_| NodeRelax { dist: safe_infinity(), hops: u32::MAX, pred: u32::MAX })
            .collect();
        relax_data[source as usize].dist = 0.0;
        relax_data[source as usize].hops = 0;

        let pivot_data: Vec<NodePivot> = (0..n)
            .map(|_| NodePivot { root: 0, tree_size: 0 })
            .collect();

        Self {
            relax_data,
            pivot_data,
            k,
            t,
            last_complete_level: vec![-1; n],
        }
    }

    #[inline(always)]
    pub fn dist_value(&self, u: u32) -> DistValue {
        let rd = &self.relax_data[u as usize];
        DistValue::new(rd.dist, rd.hops, u)
    }

    #[inline(always)]
    pub fn relax(&mut self, u: u32, v: u32, weight: f64) -> bool {
        let new_dist = sanitize(self.relax_data[u as usize].dist + weight);
        let new_hops = self.relax_data[u as usize].hops.saturating_add(1);
        let new_dv = DistValue::new(new_dist, new_hops, v);
        let old_dv = self.dist_value(v);
        if new_dv <= old_dv {
            let rd = &mut self.relax_data[v as usize];
            rd.dist = new_dist;
            rd.hops = new_hops;
            rd.pred = u;
            true
        } else {
            false
        }
    }
}
