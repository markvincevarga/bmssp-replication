use super::dist_value::{safe_infinity, sanitize, DistValue};

pub struct BmsspState {
    pub d_hat: Vec<f64>,
    pub pred: Vec<u32>,
    pub hop_count: Vec<u32>,
    pub k: usize,
    pub t: usize,
    pub root: Vec<u32>,
    pub tree_size: Vec<u32>,
    pub last_complete_level: Vec<i16>,
}

impl BmsspState {
    pub fn new(n: usize, source: u32) -> Self {
        let log_n = if n > 1 { (n as f64).log2() } else { 1.0 };
        let k = (log_n.powf(1.0 / 3.0)).floor().max(1.0) as usize;
        let t = (log_n.powf(2.0 / 3.0)).floor().max(1.0) as usize;

        let mut d_hat = vec![safe_infinity(); n];
        let mut hop_count = vec![u32::MAX; n];
        let pred = vec![u32::MAX; n];

        d_hat[source as usize] = 0.0;
        hop_count[source as usize] = 0;

        Self {
            d_hat,
            pred,
            hop_count,
            k,
            t,
            root: vec![0; n],
            tree_size: vec![0; n],
            last_complete_level: vec![-1; n],
        }
    }

    pub fn dist_value(&self, u: u32) -> DistValue {
        DistValue::new(self.d_hat[u as usize], self.hop_count[u as usize], u)
    }

    // Remark 3.4: <= so edges relaxed at lower levels can be re-used at upper levels.
    pub fn relax(&mut self, u: u32, v: u32, weight: f64) -> bool {
        let new_dist = sanitize(self.d_hat[u as usize] + weight);
        let new_hops = self.hop_count[u as usize].saturating_add(1);
        let new_dv = DistValue::new(new_dist, new_hops, v);
        let old_dv = self.dist_value(v);
        if new_dv <= old_dv {
            self.d_hat[v as usize] = new_dist;
            self.hop_count[v as usize] = new_hops;
            self.pred[v as usize] = u;
            true
        } else {
            false
        }
    }
}
