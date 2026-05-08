use super::dist_value::sanitize;
use super::epoch_map::EpochMap;
use super::state::BmsspState;
use crate::graph_repr::CsrAosGraph;

pub fn find_pivots(
    graph: &CsrAosGraph,
    state: &mut BmsspState,
    bound: f64,
    sources: &[u32],
    in_set: &mut EpochMap,
    explored: &mut Vec<u32>,
) -> Vec<u32> {
    let k = state.k;

    if sources.is_empty() {
        explored.clear();
        return Vec::new();
    }

    in_set.clear();
    explored.clear();
    let mut active: Vec<u32> = Vec::with_capacity(sources.len());
    let mut next_active: Vec<u32> = Vec::with_capacity(k * sources.len());

    for &s in sources {
        if !in_set.contains(s as usize) {
            explored.push(s);
            in_set.insert(s as usize);
            active.push(s);
        }
        state.pivot_data[s as usize].root = s;
        state.pivot_data[s as usize].tree_size = 0;
    }

    for _ in 0..k {
        next_active.clear();
        for &u in &active {
            if state.relax_data[u as usize].dist >= bound {
                continue;
            }
            for edge in graph.edges(u as usize) {
                let v = edge.target;
                let w = edge.weight;
                let relaxed = unsafe {
                    let ru = state.relax_data.get_unchecked(u as usize);
                    let new_dist = sanitize(ru.dist + w);
                    let new_hops = ru.hops.saturating_add(1);
                    let rv = state.relax_data.get_unchecked(v as usize);
                    let old_dist = rv.dist;
                    let old_hops = rv.hops;
                    if new_dist < old_dist || (new_dist == old_dist && new_hops <= old_hops) {
                        let rv_mut = state.relax_data.get_unchecked_mut(v as usize);
                        rv_mut.dist = new_dist;
                        rv_mut.hops = new_hops;
                        rv_mut.pred = u;
                        true
                    } else {
                        false
                    }
                };
                if relaxed {
                    state.pivot_data[v as usize].root = state.pivot_data[u as usize].root;
                    if state.relax_data[v as usize].dist < bound {
                        if !in_set.contains(v as usize) {
                            explored.push(v);
                            in_set.insert(v as usize);
                        }
                        next_active.push(v);
                    }
                }
            }
        }
        std::mem::swap(&mut active, &mut next_active);
    }

    let s_len = sources.len();
    if explored.len() > k * s_len {
        return sources.to_vec();
    }

    for &v in explored.iter() {
        let r = state.pivot_data[v as usize].root;
        state.pivot_data[r as usize].tree_size += 1;
    }

    sources
        .iter()
        .filter(|&&s| state.pivot_data[s as usize].tree_size >= k as u32)
        .copied()
        .collect()
}
