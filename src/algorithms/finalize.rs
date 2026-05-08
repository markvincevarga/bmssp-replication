use crate::graph_repr::CsrAosGraph;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone)]
struct HeapEntry {
    cost: f64,
    node: u32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
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
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

pub fn finalize_distances(
    csr: &CsrAosGraph,
    d_hat: &mut [f64],
    pred: &mut [u32],
    safe_infinity: f64,
) {
    let n = d_hat.len();
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(n);
    for i in 0..n {
        if d_hat[i] < safe_infinity {
            heap.push(HeapEntry {
                cost: d_hat[i],
                node: i as u32,
            });
        }
    }
    while let Some(HeapEntry { cost, node }) = heap.pop() {
        let u = node as usize;
        if cost > d_hat[u] {
            continue;
        }
        for edge in csr.edges(u) {
            let v = edge.target as usize;
            let nd = cost + edge.weight;
            if nd < d_hat[v] - 1e-12 {
                d_hat[v] = nd;
                pred[v] = node;
                heap.push(HeapEntry {
                    cost: nd,
                    node: edge.target,
                });
            }
        }
    }
}

pub fn finalize_distances_with<G, S>(
    csr: &CsrAosGraph,
    n: usize,
    safe_infinity: f64,
    mut get_dist: G,
    mut set: S,
) where
    G: FnMut(usize) -> f64,
    S: FnMut(usize, f64, u32),
{
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(n);
    for i in 0..n {
        let d = get_dist(i);
        if d < safe_infinity {
            heap.push(HeapEntry {
                cost: d,
                node: i as u32,
            });
        }
    }
    while let Some(HeapEntry { cost, node }) = heap.pop() {
        let u = node as usize;
        if cost > get_dist(u) {
            continue;
        }
        for edge in csr.edges(u) {
            let v = edge.target as usize;
            let nd = cost + edge.weight;
            if nd < get_dist(v) - 1e-12 {
                set(v, nd, node);
                heap.push(HeapEntry {
                    cost: nd,
                    node: edge.target,
                });
            }
        }
    }
}
