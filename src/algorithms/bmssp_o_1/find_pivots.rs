use super::epoch_map::EpochMap;
use super::state::BmsspState;
use crate::graph_repr::CsrAosGraph;

pub fn find_pivots(
    graph: &CsrAosGraph,
    state: &mut BmsspState,
    bound: f64,
    sources: &[u32],
    in_set: &mut EpochMap,
) -> (Vec<u32>, Vec<u32>) {
    let k = state.k;

    if sources.is_empty() {
        return (Vec::new(), Vec::new());
    }

    in_set.clear();
    let mut explored: Vec<u32> = Vec::new();
    let mut active: Vec<u32> = Vec::new();

    for &s in sources {
        if !in_set.contains(s as usize) {
            explored.push(s);
            in_set.insert(s as usize);
            active.push(s);
        }
        state.root[s as usize] = s;
        state.tree_size[s as usize] = 0;
    }

    for _ in 0..k {
        let mut next_active: Vec<u32> = Vec::new();
        for &u in &active {
            if state.d_hat[u as usize] >= bound {
                continue;
            }
            for edge in graph.edges(u as usize) {
                let v = edge.target;
                let w = edge.weight;
                if state.relax(u, v, w) {
                    state.root[v as usize] = state.root[u as usize];
                    if state.d_hat[v as usize] < bound {
                        if !in_set.contains(v as usize) {
                            explored.push(v);
                            in_set.insert(v as usize);
                        }
                        next_active.push(v);
                    }
                }
            }
        }
        active = next_active;
    }

    let s_len = sources.len();
    if explored.len() > k * s_len {
        return (sources.to_vec(), explored);
    }

    for &v in &explored {
        let r = state.root[v as usize];
        state.tree_size[r as usize] += 1;
    }

    let pivots: Vec<u32> = sources
        .iter()
        .filter(|&&s| state.tree_size[s as usize] >= k as u32)
        .copied()
        .collect();

    (pivots, explored)
}
