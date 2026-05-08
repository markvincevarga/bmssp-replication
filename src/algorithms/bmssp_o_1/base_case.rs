use super::dist_value::DistValue;
use super::state::BmsspState;
use crate::graph_repr::CsrAosGraph;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

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
) -> (f64, Vec<u32>) {
    let k = state.k;
    debug_assert!(sources.len() == 1);
    let max_complete = (k + 1) * sources.len().max(1);
    let mut complete: Vec<u32> = Vec::new();
    let mut complete_set: HashSet<u32> = HashSet::new();
    let mut heap = BinaryHeap::new();

    for &s in sources {
        heap.push(HeapEntry {
            dv: state.dist_value(s),
        });
    }

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
        if complete_set.contains(&u) {
            continue;
        }

        complete.push(u);
        complete_set.insert(u);

        for edge in graph.edges(u as usize) {
            if state.relax(u, edge.target, edge.weight) {
                heap.push(HeapEntry {
                    dv: state.dist_value(edge.target),
                });
            }
        }
    }

    let threshold = k * sources.len().max(1);
    if complete.len() <= threshold {
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
