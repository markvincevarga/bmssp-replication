use super::dist_value::{safe_infinity, DistValue};
use rustc_hash::FxHashMap as HashMap;

pub struct BlockList {
    blocks: Vec<Vec<DistValue>>,
    block_count: usize,
    total_size: usize,
    current_block: usize,
    m: usize,
    best_dist: HashMap<u32, f64>,
    upper_bound: f64,
    lower_bound: f64,
    step: f64,
}

impl BlockList {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            block_count: 0,
            total_size: 0,
            current_block: 0,
            m: 1,
            best_dist: HashMap::default(),
            upper_bound: f64::INFINITY,
            lower_bound: 0.0,
            step: 0.0,
        }
    }

    pub fn initialize(&mut self, m: usize, base_bound: f64, upper_bound: f64) {
        self.total_size = 0;
        self.best_dist.clear();
        self.upper_bound = upper_bound;

        let m = m.max(1);
        self.m = m;
        let range = (upper_bound - base_bound).max(0.0);
        let step = if m > 0 { range / m as f64 } else { range };

        self.lower_bound = base_bound;
        self.step = step;

        self.blocks.clear();
        self.blocks.resize_with(m, Vec::new);
        self.block_count = m;
        self.current_block = 0;
    }

    fn block_index(&self, dist: f64) -> usize {
        if self.step <= 0.0 || self.block_count == 0 {
            return 0;
        }
        let raw = (dist - self.lower_bound) / self.step;
        if raw < 0.0 {
            return 0;
        }
        (raw as usize).min(self.block_count - 1)
    }

    pub fn insert(&mut self, dv: DistValue) {
        if dv.dist >= safe_infinity() {
            return;
        }
        if let Some(&best) = self.best_dist.get(&dv.node) {
            if best <= dv.dist {
                return;
            }
        }
        self.best_dist.insert(dv.node, dv.dist);
        let idx = self.block_index(dv.dist);
        self.blocks[idx].push(dv);
        self.total_size += 1;
    }

    pub fn batch_prepend(&mut self, entries: &[DistValue]) {
        let mut min_idx = self.current_block;
        for &dv in entries {
            if let Some(&best) = self.best_dist.get(&dv.node) {
                if best <= dv.dist {
                    continue;
                }
            }
            self.best_dist.insert(dv.node, dv.dist);
            let idx = self.block_index(dv.dist);
            self.blocks[idx].push(dv);
            self.total_size += 1;
            if idx < min_idx {
                min_idx = idx;
            }
        }
        self.current_block = min_idx;
    }

    pub fn pull(&mut self) -> (f64, Vec<u32>) {
        loop {
            if self.current_block >= self.block_count {
                return (safe_infinity(), Vec::new());
            }

            if self.blocks[self.current_block].is_empty() {
                self.current_block += 1;
                continue;
            }

            let block = std::mem::take(&mut self.blocks[self.current_block]);
            let mut valid: Vec<DistValue> = Vec::with_capacity(block.len());
            let mut stale_count = 0;

            for dv in block {
                if let Some(&best) = self.best_dist.get(&dv.node) {
                    if dv.dist == best {
                        valid.push(dv);
                        continue;
                    }
                }
                stale_count += 1;
            }

            self.total_size = self.total_size.saturating_sub(stale_count);

            if valid.is_empty() {
                self.current_block += 1;
                continue;
            }

            let mut nodes: Vec<u32> = Vec::new();
            let boundary;

            if valid.len() <= self.m {
                for dv in &valid {
                    nodes.push(dv.node);
                    self.best_dist.remove(&dv.node);
                }
                self.total_size = self.total_size.saturating_sub(valid.len());

                // Lemma 3.3 PULL: if no remaining values, x = B
                if self.total_size == 0 {
                    boundary = self.upper_bound;
                } else {
                    boundary = self.block_upper_bound(self.current_block);
                }
            } else {
                valid.select_nth_unstable(self.m);
                for dv in &valid[..self.m] {
                    nodes.push(dv.node);
                    self.best_dist.remove(&dv.node);
                }
                let candidate = valid[self.m].dist;
                let has_tie = valid[..self.m].iter().any(|dv| dv.dist == candidate);
                boundary = if has_tie {
                    self.block_upper_bound(self.current_block)
                } else {
                    candidate
                };
                self.blocks[self.current_block] = valid[self.m..].to_vec();
                self.total_size = self.total_size.saturating_sub(self.m);
            }

            return (boundary, nodes);
        }
    }

    fn block_upper_bound(&self, idx: usize) -> f64 {
        if idx + 1 < self.block_count {
            self.lower_bound + (idx + 1) as f64 * self.step
        } else {
            self.upper_bound
        }
    }

    // Castro fix (Remark 3.8): remove completed vertices from D so they don't
    // linger with stale keys. Uses lazy deletion -- best_dist removal causes
    // the entry to be skipped as stale during pull.
    pub fn erase(&mut self, node: u32) {
        self.best_dist.remove(&node);
    }

    pub fn size(&self) -> usize {
        self.total_size
    }
}
