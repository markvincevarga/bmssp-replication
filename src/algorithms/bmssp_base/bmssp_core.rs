use super::base_case::base_case;
use super::d_struct::DStruct;
use super::dist_value::safe_infinity;
use super::epoch_map::EpochMap;
use super::find_pivots::find_pivots;
use super::state::BmsspState;
use crate::graph_repr::CsrAosGraph;
use std::collections::VecDeque;

pub fn bmssp_recursive(
    graph: &CsrAosGraph,
    state: &mut BmsspState,
    level: usize,
    bound: f64,
    sources: &[u32],
    queues: &mut Vec<DStruct>,
    epoch_map: &mut EpochMap,
) -> (f64, Vec<u32>) {
    if sources.is_empty() {
        return (f64::NEG_INFINITY, Vec::new());
    }

    if level == 0 {
        return base_case(graph, state, bound, sources);
    }

    let k = state.k;
    let t = state.t;

    let (pivots, explored) = find_pivots(graph, state, bound, sources, epoch_map);

    let m = 1usize.checked_shl(((level - 1) * t) as u32).unwrap_or(usize::MAX);
    let lower = pivots
        .iter()
        .map(|&p| state.d_hat[p as usize])
        .fold(f64::INFINITY, f64::min);

    while queues.len() <= level {
        queues.push(DStruct::new());
    }
    queues[level].initialize(m, lower, bound);

    for &p in &pivots {
        let dv = state.dist_value(p);
        queues[level].insert(dv);
    }

    let mut all_complete: Vec<u32> = Vec::new();
    let quota = k.saturating_mul(1usize.checked_shl((level * t) as u32).unwrap_or(usize::MAX));

    let mut last_b_prime = bound;
    let mut queue_exhausted = false;

    let mut to_propagate: VecDeque<u32> = VecDeque::new();

    while all_complete.len() < quota && queues[level].size() > 0 {
        let (batch_bound, batch) = queues[level].pull();
        if batch.is_empty() {
            queue_exhausted = true;
            break;
        }

        let effective_bound = batch_bound.min(bound);
        let (b_prime_i, newly_complete) =
            bmssp_recursive(graph, state, level - 1, effective_bound, &batch, queues, epoch_map);

        last_b_prime = b_prime_i;

        // Castro fix: erase each u in U_i from D before relaxing edges (before
        // line 15 of Algorithm 3). Prevents completed vertices from lingering
        // in D with stale keys, ensuring U_i sets are disjoint (Remark 3.8).
        for &u in &newly_complete {
            queues[level].erase(u);
            state.last_complete_level[u as usize] = level as i16;
        }
        all_complete.extend_from_slice(&newly_complete);

        let mut prepend_entries: Vec<super::dist_value::DistValue> = Vec::new();

        // Finalised v whose d_hat just decreased: propagate inline rather than
        // re-queuing (which would force a full recursive call). Strict-decrease
        // guard below prevents cycling on cmp-tuple tie updates.
        debug_assert!(to_propagate.is_empty());
        for &u in &newly_complete {
            for edge in graph.edges(u as usize) {
                let v = edge.target;
                let w = edge.weight;
                let old_dist = state.d_hat[v as usize];
                if !state.relax(u, v, w) {
                    continue;
                }
                let new_dv = state.dist_value(v);
                if new_dv.dist >= bound || new_dv.dist >= safe_infinity() {
                    continue;
                }
                if state.last_complete_level[v as usize] >= 0 {
                    // strict decrease only: tie-only relaxations would cycle indefinitely
                    if new_dv.dist < old_dist {
                        to_propagate.push_back(v);
                    }
                    continue;
                }
                if new_dv.dist >= batch_bound {
                    queues[level].insert(new_dv);
                } else {
                    prepend_entries.push(new_dv);
                }
            }
        }
        while let Some(u) = to_propagate.pop_front() {
            for edge in graph.edges(u as usize) {
                let v = edge.target;
                let w = edge.weight;
                let old_dist = state.d_hat[v as usize];
                if !state.relax(u, v, w) {
                    continue;
                }
                let new_dv = state.dist_value(v);
                if new_dv.dist >= bound || new_dv.dist >= safe_infinity() {
                    continue;
                }
                if state.last_complete_level[v as usize] >= 0 {
                    if new_dv.dist < old_dist {
                        to_propagate.push_back(v);
                    }
                    continue;
                }
                if new_dv.dist >= batch_bound {
                    queues[level].insert(new_dv);
                } else {
                    prepend_entries.push(new_dv);
                }
            }
        }

        for &s in &batch {
            if state.last_complete_level[s as usize] == level as i16 {
                continue;
            }
            let d = state.d_hat[s as usize];
            if d >= b_prime_i && d < batch_bound && d < bound {
                prepend_entries.push(state.dist_value(s));
            }
        }

        if !prepend_entries.is_empty() {
            queues[level].batch_prepend(&prepend_entries);
        }
    }

    let result_bound = if queues[level].size() == 0 || queue_exhausted {
        bound
    } else {
        last_b_prime.min(bound)
    };

    for &v in &explored {
        if state.last_complete_level[v as usize] != level as i16
            && state.d_hat[v as usize] < result_bound
        {
            all_complete.push(v);
        }
    }

    (result_bound, all_complete)
}
