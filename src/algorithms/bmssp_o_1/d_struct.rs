use super::dist_value::{safe_infinity, DistValue};
use rustc_hash::FxHashMap as HashMap;
use std::cmp::Ordering;
use std::collections::{BTreeMap, VecDeque};

type BlockId = usize;

#[derive(Copy, Clone, Debug)]
struct OrderedKey {
    upper: f64,
    counter: u64,
}

impl PartialEq for OrderedKey {
    fn eq(&self, o: &Self) -> bool {
        self.upper.partial_cmp(&o.upper) == Some(Ordering::Equal) && self.counter == o.counter
    }
}
impl Eq for OrderedKey {}
impl PartialOrd for OrderedKey {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrderedKey {
    fn cmp(&self, o: &Self) -> Ordering {
        self.upper
            .partial_cmp(&o.upper)
            .expect("DStruct: NaN encountered in upper bound")
            .then(self.counter.cmp(&o.counter))
    }
}

struct Block {
    items: Vec<DistValue>,
    upper: f64,
    counter: u64,
}

impl Block {
    fn key(&self) -> OrderedKey {
        OrderedKey {
            upper: self.upper,
            counter: self.counter,
        }
    }
}

pub struct DStruct {
    storage: Vec<Option<Block>>,
    free: Vec<BlockId>,

    d0: VecDeque<BlockId>,
    d1: BTreeMap<OrderedKey, BlockId>,

    next_counter: u64,
    m: usize,
    upper_bound: f64,

    total_size: usize,
    best: HashMap<u32, f64>,
}

impl DStruct {
    pub fn new() -> Self {
        Self {
            storage: Vec::new(),
            free: Vec::new(),
            d0: VecDeque::new(),
            d1: BTreeMap::new(),
            next_counter: 0,
            m: 1,
            upper_bound: safe_infinity(),
            total_size: 0,
            best: HashMap::default(),
        }
    }

    pub fn initialize(&mut self, m: usize, _base_bound: f64, upper_bound: f64) {
        self.m = m.max(1);
        self.upper_bound = upper_bound;
        self.storage.clear();
        self.free.clear();
        self.d0.clear();
        self.d1.clear();
        self.best.clear();
        self.total_size = 0;
        self.next_counter = 0;

        let counter = self.bump_counter();
        let id = self.alloc_block(Block {
            items: Vec::new(),
            upper: upper_bound,
            counter,
        });
        let key = self.block(id).key();
        self.d1.insert(key, id);
    }

    pub fn size(&self) -> usize {
        self.best.len()
    }

    fn bump_counter(&mut self) -> u64 {
        let c = self.next_counter;
        self.next_counter += 1;
        c
    }

    fn alloc_block(&mut self, block: Block) -> BlockId {
        if let Some(id) = self.free.pop() {
            self.storage[id] = Some(block);
            id
        } else {
            self.storage.push(Some(block));
            self.storage.len() - 1
        }
    }

    fn free_block(&mut self, id: BlockId) {
        self.storage[id] = None;
        self.free.push(id);
    }

    fn block(&self, id: BlockId) -> &Block {
        self.storage[id].as_ref().expect("freed block accessed")
    }

    fn block_mut(&mut self, id: BlockId) -> &mut Block {
        self.storage[id].as_mut().expect("freed block accessed")
    }

    pub fn insert(&mut self, dv: DistValue) {
        if dv.dist >= safe_infinity() {
            return;
        }
        if let Some(&best) = self.best.get(&dv.node) {
            if best <= dv.dist {
                return;
            }
        }
        self.best.insert(dv.node, dv.dist);

        let target_id = self
            .d1
            .range(
                OrderedKey {
                    upper: dv.dist,
                    counter: 0,
                }..,
            )
            .next()
            .map(|(_, &id)| id)
            .expect("D₁ should always contain a block whose upper ≥ value");

        {
            let block = self.block_mut(target_id);
            block.items.push(dv);
        }
        self.total_size += 1;

        if self.block(target_id).items.len() > self.m {
            self.split_block(target_id);
        }
    }

    fn split_block(&mut self, id: BlockId) {
        let items_len = self.block(id).items.len();
        let mid = items_len / 2;
        let original_upper = self.block(id).upper;

        self.block_mut(id).items.select_nth_unstable_by(mid, |a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .expect("DStruct::split_block: NaN dist")
                .then_with(|| a.hops.cmp(&b.hops))
                .then_with(|| a.node.cmp(&b.node))
        });

        let mut items = std::mem::take(&mut self.block_mut(id).items);
        let upper_half: Vec<DistValue> = items.split_off(mid);
        let lower_half = items;

        let lower_max = lower_half
            .iter()
            .map(|dv| dv.dist)
            .fold(f64::NEG_INFINITY, f64::max);
        let new_upper_lower = if lower_max.is_finite() {
            lower_max
        } else {
            self.block(id).upper
        };

        let old_key = self.block(id).key();
        self.d1.remove(&old_key);

        let new_lower_counter = self.bump_counter();
        {
            let b = self.block_mut(id);
            b.items = lower_half;
            b.upper = new_upper_lower;
            b.counter = new_lower_counter;
        }
        let lower_key = self.block(id).key();
        self.d1.insert(lower_key, id);

        let upper_counter = self.bump_counter();
        let upper_block = Block {
            items: upper_half,
            upper: original_upper,
            counter: upper_counter,
        };
        let upper_id = self.alloc_block(upper_block);
        let upper_key = self.block(upper_id).key();
        self.d1.insert(upper_key, upper_id);
    }

    pub fn batch_prepend(&mut self, entries: &[DistValue]) {
        let mut keep: Vec<DistValue> = Vec::with_capacity(entries.len());
        for &dv in entries {
            if dv.dist >= safe_infinity() {
                continue;
            }
            if let Some(&best) = self.best.get(&dv.node) {
                if best <= dv.dist {
                    continue;
                }
            }
            self.best.insert(dv.node, dv.dist);
            keep.push(dv);
        }

        if keep.is_empty() {
            return;
        }

        if keep.len() <= self.m {
            self.total_size += keep.len();
            let counter = self.bump_counter();
            let id = self.alloc_block(Block {
                items: keep,
                upper: f64::NEG_INFINITY,
                counter,
            });
            self.d0.push_front(id);
        } else {
            let chunks = self.partition_by_medians(keep);
            self.total_size += chunks.iter().map(|c| c.len()).sum::<usize>();
            for chunk in chunks.into_iter().rev() {
                let counter = self.bump_counter();
                let id = self.alloc_block(Block {
                    items: chunk,
                    upper: f64::NEG_INFINITY,
                    counter,
                });
                self.d0.push_front(id);
            }
        }
    }

    fn partition_by_medians(&mut self, mut items: Vec<DistValue>) -> Vec<Vec<DistValue>> {
        let m = self.m;
        let mut out: Vec<Vec<DistValue>> = Vec::new();
        let mut stack: Vec<Vec<DistValue>> = Vec::new();
        stack.push(std::mem::take(&mut items));
        while let Some(chunk) = stack.pop() {
            if chunk.len() <= m {
                out.push(chunk);
                continue;
            }
            let mid = chunk.len() / 2;
            let mut chunk = chunk;
            chunk.select_nth_unstable_by(mid, |a, b| {
                a.dist
                    .partial_cmp(&b.dist)
                    .expect("DStruct::partition_by_medians: NaN")
                    .then_with(|| a.hops.cmp(&b.hops))
                    .then_with(|| a.node.cmp(&b.node))
            });
            let upper_half = chunk.split_off(mid);
            stack.push(upper_half);
            stack.push(chunk);
        }
        out
    }

    fn block_fresh_min(&self, id: BlockId) -> f64 {
        self.block(id)
            .items
            .iter()
            .filter(|dv| self.best.get(&dv.node).copied() == Some(dv.dist))
            .map(|dv| dv.dist)
            .fold(f64::INFINITY, f64::min)
    }

    fn scan_d0_unwalked_min(&self, d0_visited: usize) -> f64 {
        (d0_visited..self.d0.len())
            .find_map(|i| {
                let m = self.block_fresh_min(self.d0[i]);
                if m.is_finite() {
                    Some(m)
                } else {
                    None
                }
            })
            .unwrap_or(f64::INFINITY)
    }

    fn scan_d1_unvisited_min(&self, visited: &std::collections::HashSet<BlockId>) -> f64 {
        self.d1
            .iter()
            .filter(|(_, id)| !visited.contains(id))
            .find_map(|(_, &id)| {
                let m = self.block_fresh_min(id);
                if m.is_finite() {
                    Some(m)
                } else {
                    None
                }
            })
            .unwrap_or(f64::INFINITY)
    }

    pub fn pull(&mut self) -> (f64, Vec<u32>) {
        use std::collections::HashSet;
        let cmp = |a: &DistValue, b: &DistValue| {
            a.dist
                .partial_cmp(&b.dist)
                .expect("DStruct::pull: NaN")
                .then_with(|| a.hops.cmp(&b.hops))
                .then_with(|| a.node.cmp(&b.node))
        };
        let is_fresh = |best: &HashMap<u32, f64>, dv: &DistValue| {
            best.get(&dv.node).copied() == Some(dv.dist)
        };

        let mut s_prime: Vec<DistValue> = Vec::new();
        let mut d0_visited = 0usize;
        while d0_visited < self.d0.len() {
            let id = self.d0[d0_visited];
            for &dv in &self.block(id).items {
                if is_fresh(&self.best, &dv) {
                    s_prime.push(dv);
                }
            }
            d0_visited += 1;
            if s_prime.len() >= self.m {
                break;
            }
        }

        let mut visited_d1_ids: HashSet<BlockId> = HashSet::new();
        let d1_keys_iter: Vec<(OrderedKey, BlockId)> = self
            .d1
            .iter()
            .map(|(k, &v)| (*k, v))
            .collect();
        let mut d1_taken_local = 0usize;
        for (_, id) in &d1_keys_iter {
            if d1_taken_local >= self.m {
                break;
            }
            visited_d1_ids.insert(*id);
            for &dv in &self.block(*id).items {
                if is_fresh(&self.best, &dv) {
                    s_prime.push(dv);
                    d1_taken_local += 1;
                }
            }
        }

        loop {
            let next_d0_min = self.scan_d0_unwalked_min(d0_visited);
            let next_d1_min = self.scan_d1_unvisited_min(&visited_d1_ids);
            let cap = next_d0_min.min(next_d1_min);
            if !cap.is_finite() {
                break;
            }
            let walked_max = s_prime
                .iter()
                .map(|dv| dv.dist)
                .fold(f64::NEG_INFINITY, f64::max);
            if cap > walked_max {
                break;
            }
            let mut walked_more = false;
            while d0_visited < self.d0.len() {
                let id = self.d0[d0_visited];
                let m = self.block_fresh_min(id);
                if m.is_finite() && m > cap {
                    break;
                }
                for &dv in &self.block(id).items {
                    if is_fresh(&self.best, &dv) {
                        s_prime.push(dv);
                    }
                }
                d0_visited += 1;
                walked_more = true;
            }
            let to_walk: Vec<BlockId> = self
                .d1
                .iter()
                .filter(|(_, id)| !visited_d1_ids.contains(id))
                .take_while(|(_, &id)| {
                    let m = self.block_fresh_min(id);
                    !m.is_finite() || m <= cap
                })
                .map(|(_, &id)| id)
                .collect();
            for id in to_walk {
                visited_d1_ids.insert(id);
                for &dv in &self.block(id).items {
                    if is_fresh(&self.best, &dv) {
                        s_prime.push(dv);
                    }
                }
                walked_more = true;
            }
            if !walked_more {
                break;
            }
        }

        let next_d0_min = self.scan_d0_unwalked_min(d0_visited);
        let next_d1_min = self.scan_d1_unvisited_min(&visited_d1_ids);

        if s_prime.is_empty() {
            self.compact_visited(d0_visited, &visited_d1_ids, &HashSet::new());
            self.drop_empty_d0_prefix(d0_visited);
            self.drop_empty_d1_visited(&visited_d1_ids);
            self.ensure_d1_nonempty();
            return (
                self.upper_bound.min(next_d0_min).min(next_d1_min),
                Vec::new(),
            );
        }

        let cap = next_d0_min.min(next_d1_min);
        let (mut eligible, deferred): (Vec<DistValue>, Vec<DistValue>) =
            s_prime.into_iter().partition(|dv| dv.dist < cap);
        let deferred_min = deferred
            .iter()
            .map(|dv| dv.dist)
            .fold(f64::INFINITY, f64::min);

        let to_return: Vec<DistValue>;
        let separator: f64;
        if eligible.is_empty() {
            self.compact_visited(d0_visited, &visited_d1_ids, &HashSet::new());
            self.drop_empty_d0_prefix(d0_visited);
            self.drop_empty_d1_visited(&visited_d1_ids);
            self.ensure_d1_nonempty();
            return (
                self.upper_bound.min(next_d0_min).min(next_d1_min),
                Vec::new(),
            );
        } else if eligible.len() <= self.m {
            to_return = eligible;
            separator = self.upper_bound.min(cap).min(deferred_min);
        } else {
            eligible.select_nth_unstable_by(self.m, cmp);
            let pivot_dist = eligible[self.m].dist;
            let max_in_returned = eligible[..self.m]
                .iter()
                .map(|dv| dv.dist)
                .fold(f64::NEG_INFINITY, f64::max);
            let needs_tie_extension = max_in_returned == pivot_dist;
            let (returned_part, leftover_part): (Vec<DistValue>, Vec<DistValue>) =
                if needs_tie_extension {
                    eligible.into_iter().partition(|dv| dv.dist <= pivot_dist)
                } else {
                    let leftover = eligible.split_off(self.m);
                    (eligible, leftover)
                };
            let leftover_min = leftover_part
                .iter()
                .map(|dv| dv.dist)
                .fold(f64::INFINITY, f64::min);
            separator = self
                .upper_bound
                .min(cap)
                .min(leftover_min)
                .min(deferred_min);
            to_return = returned_part;
        }

        let returned_keys: HashSet<(u32, u64)> = to_return
            .iter()
            .map(|dv| (dv.node, dv.dist.to_bits()))
            .collect();

        self.compact_visited(d0_visited, &visited_d1_ids, &returned_keys);

        for dv in &to_return {
            self.best.remove(&dv.node);
        }
        self.total_size = self.total_size.saturating_sub(to_return.len());

        self.drop_empty_d0_prefix(d0_visited);
        self.drop_empty_d1_visited(&visited_d1_ids);
        self.ensure_d1_nonempty();

        let _ = d1_keys_iter;
        let nodes: Vec<u32> = to_return.iter().map(|dv| dv.node).collect();
        (separator, nodes)
    }

    fn compact_visited(
        &mut self,
        d0_visited: usize,
        visited_d1_ids: &std::collections::HashSet<BlockId>,
        returned_keys: &std::collections::HashSet<(u32, u64)>,
    ) {
        let best_snapshot: HashMap<u32, f64> = self.best.clone();
        let returned = returned_keys;
        let filter = |dv: &DistValue| -> bool {
            if returned.contains(&(dv.node, dv.dist.to_bits())) {
                return false;
            }
            best_snapshot.get(&dv.node).copied() == Some(dv.dist)
        };

        for i in 0..d0_visited {
            let id = self.d0[i];
            let items = std::mem::take(&mut self.block_mut(id).items);
            self.block_mut(id).items = items.into_iter().filter(filter).collect();
        }
        let visited_d1_ids_clone: Vec<BlockId> = visited_d1_ids.iter().copied().collect();
        for id in visited_d1_ids_clone {
            let items = std::mem::take(&mut self.block_mut(id).items);
            self.block_mut(id).items = items.into_iter().filter(filter).collect();
        }
    }

    fn drop_empty_d0_prefix(&mut self, d0_visited: usize) {
        let mut to_check = d0_visited;
        while to_check > 0 {
            let front_id = match self.d0.front() {
                Some(&id) => id,
                None => break,
            };
            if self.block(front_id).items.is_empty() {
                self.d0.pop_front();
                self.free_block(front_id);
                to_check -= 1;
            } else {
                break;
            }
        }
    }

    fn drop_empty_d1_visited(&mut self, visited_d1_ids: &std::collections::HashSet<BlockId>) {
        let to_drop: Vec<(OrderedKey, BlockId)> = visited_d1_ids
            .iter()
            .filter_map(|&id| {
                if self.storage[id].as_ref().map(|b| b.items.is_empty()).unwrap_or(false) {
                    let key = self.block(id).key();
                    Some((key, id))
                } else {
                    None
                }
            })
            .collect();
        for (key, id) in to_drop {
            self.d1.remove(&key);
            self.free_block(id);
        }
    }

    fn ensure_d1_nonempty(&mut self) {
        let needs_top = match self.d1.iter().next_back() {
            None => true,
            Some((key, _)) => key.upper < self.upper_bound,
        };
        if needs_top {
            let counter = self.bump_counter();
            let upper = self.upper_bound;
            let id = self.alloc_block(Block {
                items: Vec::new(),
                upper,
                counter,
            });
            let key = self.block(id).key();
            self.d1.insert(key, id);
        }
    }

    pub fn erase(&mut self, node: u32) {
        self.best.remove(&node);
    }
}

impl Default for DStruct {
    fn default() -> Self {
        Self::new()
    }
}
