use super::dist_value::{sanitize, DistValue};
use super::state::BmsspState;
use crate::graph_repr::CsrAosGraph;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone)]
struct HeapEntry {
    dv: DistValue,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dv.cmp(&self.dv)
    }
}

pub fn base_case(
    graph: &CsrAosGraph,
    state: &mut BmsspState,
    bound: f64,
    sources: &[u32],
    in_complete: &mut Vec<bool>,
) -> (f64, Vec<u32>) {
    let k = state.k;
    let n = state.d_hat.len();
    debug_assert!(sources.len() == 1);
    if in_complete.len() < n {
        in_complete.resize(n, false);
    }
    let s = sources[0];
    let max_complete = k + 1;
    let mut complete: Vec<u32> = Vec::with_capacity(max_complete);
    let mut dirty: Vec<u32> = Vec::new();
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(max_complete);

    heap.push(HeapEntry {
        dv: state.dist_value(s),
    });

    while let Some(HeapEntry { dv }) = heap.pop() {
        let u = dv.node;
        let current_dv = state.dist_value(u);
        if dv > current_dv {
            continue;
        }
        if current_dv.dist >= bound {
            break;
        }
        if complete.len() >= max_complete {
            break;
        }
        if in_complete[u as usize] {
            continue;
        }

        complete.push(u);
        in_complete[u as usize] = true;
        dirty.push(u);

        for edge in graph.edges(u as usize) {
            let v = edge.target;
            let w = edge.weight;
            let relaxed = unsafe {
                let new_dist = sanitize(*state.d_hat.get_unchecked(u as usize) + w);
                let new_hops = state.hop_count.get_unchecked(u as usize).saturating_add(1);
                let old_dist = *state.d_hat.get_unchecked(v as usize);
                let old_hops = *state.hop_count.get_unchecked(v as usize);
                let new_key = ((new_dist.to_bits() as u128) << 64) | (new_hops as u128);
                let old_key = ((old_dist.to_bits() as u128) << 64) | (old_hops as u128);
                if new_key <= old_key {
                    *state.d_hat.get_unchecked_mut(v as usize) = new_dist;
                    *state.hop_count.get_unchecked_mut(v as usize) = new_hops;
                    *state.pred.get_unchecked_mut(v as usize) = u;
                    true
                } else {
                    false
                }
            };
            if relaxed {
                heap.push(HeapEntry {
                    dv: state.dist_value(v),
                });
            }
        }
    }

    for &u in &dirty {
        in_complete[u as usize] = false;
    }

    if complete.len() <= k {
        (bound, complete)
    } else {
        let max_dv = complete
            .iter()
            .map(|&u| state.dist_value(u))
            .max()
            .unwrap();
        let b_prime = max_dv.dist;
        let filtered: Vec<u32> = complete
            .into_iter()
            .filter(|&u| state.dist_value(u) < max_dv)
            .collect();
        (b_prime, filtered)
    }
}
