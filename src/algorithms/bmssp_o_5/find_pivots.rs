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
        state.root[s as usize] = s;
        state.tree_size[s as usize] = 0;
    }

    for _ in 0..k {
        next_active.clear();
        for &u in &active {
            if state.d_hat[u as usize] >= bound {
                continue;
            }
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
        std::mem::swap(&mut active, &mut next_active);
        if active.is_empty() {
            break;
        }
    }

    let s_len = sources.len();
    if explored.len() > k * s_len {
        return sources.to_vec();
    }

    for &v in explored.iter() {
        let r = state.root[v as usize];
        state.tree_size[r as usize] += 1;
    }

    sources
        .iter()
        .filter(|&&s| state.tree_size[s as usize] >= k as u32)
        .copied()
        .collect()
}
