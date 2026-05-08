use super::base_case::base_case;
use super::d_struct::DStruct;
use std::collections::VecDeque;
use super::dist_value::{safe_infinity, sanitize};
use super::epoch_map::EpochMap;
use super::find_pivots::find_pivots;
use super::state::BmsspState;
use crate::graph_repr::CsrAosGraph;

enum FramePhase {
    Init,
    ProcessBatch,
    AfterRecurse { batch_bound: f64, batch: Vec<u32> },
}

struct StackFrame {
    level: usize,
    bound: f64,
    sources: Vec<u32>,
    quota: usize,
    all_complete: Vec<u32>,
    last_b_prime: f64,
    queue_exhausted: bool,
    explored: Vec<u32>,
    prepend_buf: Vec<super::dist_value::DistValue>,
    phase: FramePhase,
}

pub fn bmssp_iterative(
    graph: &CsrAosGraph,
    state: &mut BmsspState,
    top_level: usize,
    bound: f64,
    sources: &[u32],
    queues: &mut Vec<DStruct>,
    epoch_map: &mut EpochMap,
) -> (f64, Vec<u32>) {
    let n = state.d_hat.len();
    let mut in_complete: Vec<bool> = vec![false; n];
    let mut stack: Vec<StackFrame> = Vec::new();
    let mut pivots_buf: Vec<u32> = Vec::new();

    stack.push(StackFrame {
        level: top_level,
        bound,
        sources: sources.to_vec(),
        quota: 0,
        all_complete: Vec::new(),
        last_b_prime: bound,
        queue_exhausted: false,
        explored: Vec::with_capacity(state.k * sources.len()),
        prepend_buf: Vec::new(),
        phase: FramePhase::Init,
    });

    let mut return_val: (f64, Vec<u32>) = (f64::NEG_INFINITY, Vec::new());

    while let Some(frame) = stack.last_mut() {
        match std::mem::replace(&mut frame.phase, FramePhase::Init) {
            FramePhase::Init => {
                if frame.sources.is_empty() {
                    return_val = (f64::NEG_INFINITY, Vec::new());
                    stack.pop();
                    continue;
                }

                if frame.level == 0 || (frame.level <= 1 && frame.sources.len() == 1) {
                    return_val = base_case(graph, state, frame.bound, &frame.sources, &mut in_complete);
                    stack.pop();
                    continue;
                }

                let k = state.k;
                let t = state.t;
                let level = frame.level;
                let bound = frame.bound;

                let mut explored = std::mem::take(&mut frame.explored);
                find_pivots(graph, state, bound, &frame.sources, epoch_map, &mut explored, &mut pivots_buf);
                frame.explored = explored;

                let m = 1usize
                    .checked_shl(((level - 1) * t) as u32)
                    .unwrap_or(usize::MAX);
                let lower = pivots_buf
                    .iter()
                    .map(|&p| state.d_hat[p as usize])
                    .fold(f64::INFINITY, f64::min);

                while queues.len() <= level {
                    queues.push(DStruct::new());
                }
                queues[level].initialize(m, lower, bound);

                for &p in &pivots_buf {
                    let dv = state.dist_value(p);
                    queues[level].insert(dv);
                }

                frame.quota =
                    k.saturating_mul(1usize.checked_shl((level * t) as u32).unwrap_or(usize::MAX));
                frame.last_b_prime = bound;
                frame.queue_exhausted = false;
                frame.all_complete = Vec::with_capacity(frame.quota.min(1 << 20));

                frame.phase = FramePhase::ProcessBatch;
            }

            FramePhase::ProcessBatch => {
                let level = frame.level;
                let bound = frame.bound;

                if frame.all_complete.len() >= frame.quota || queues[level].size() == 0 {
                    let result_bound = if queues[level].size() == 0 || frame.queue_exhausted {
                        bound
                    } else {
                        frame.last_b_prime.min(bound)
                    };

                    for &v in &frame.explored {
                        if state.last_complete_level[v as usize] != level as i16
                            && state.d_hat[v as usize] < result_bound
                        {
                            frame.all_complete.push(v);
                        }
                    }

                    return_val = (result_bound, std::mem::take(&mut frame.all_complete));
                    stack.pop();
                    continue;
                }

                let (batch_bound, batch) = queues[level].pull();
                if batch.is_empty() {
                    frame.queue_exhausted = true;
                    frame.phase = FramePhase::ProcessBatch;
                    continue;
                }

                let effective_bound = batch_bound.min(bound);

                let child_sources = batch.clone();
                let child_explored_cap = state.k * child_sources.len();
                frame.phase = FramePhase::AfterRecurse {
                    batch_bound,
                    batch,
                };

                stack.push(StackFrame {
                    level: level - 1,
                    bound: effective_bound,
                    sources: child_sources,
                    quota: 0,
                    all_complete: Vec::new(),
                    last_b_prime: effective_bound,
                    queue_exhausted: false,
                    explored: Vec::with_capacity(child_explored_cap),
                    prepend_buf: Vec::new(),
                    phase: FramePhase::Init,
                });
            }

            FramePhase::AfterRecurse { batch_bound, batch } => {
                let (b_prime_i, newly_complete) =
                    std::mem::replace(&mut return_val, (f64::NEG_INFINITY, Vec::new()));

                let level = frame.level;
                let bound = frame.bound;

                frame.last_b_prime = b_prime_i;

                for &u in &newly_complete {
                    queues[level].erase(u);
                    state.last_complete_level[u as usize] = level as i16;
                }
                frame.all_complete.extend_from_slice(&newly_complete);

                frame.prepend_buf.clear();
                let mut to_propagate: VecDeque<u32> = VecDeque::new();

                for &u in &newly_complete {
                    for edge in graph.edges(u as usize) {
                        let v = edge.target;
                        let w = edge.weight;
                        let (relaxed, strict_decrease) = unsafe {
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
                                (true, new_dist < old_dist)
                            } else {
                                (false, false)
                            }
                        };
                        if !relaxed {
                            continue;
                        }
                        let new_dv = state.dist_value(v);
                        if new_dv.dist >= bound || new_dv.dist >= safe_infinity() {
                            continue;
                        }
                        if state.last_complete_level[v as usize] >= 0 {
                            if strict_decrease {
                                to_propagate.push_back(v);
                            }
                            continue;
                        }
                        if new_dv.dist >= batch_bound {
                            queues[level].insert(new_dv);
                        } else {
                            frame.prepend_buf.push(new_dv);
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
                            frame.prepend_buf.push(new_dv);
                        }
                    }
                }

                for &s in &batch {
                    if state.last_complete_level[s as usize] == level as i16 {
                        continue;
                    }
                    let d = state.d_hat[s as usize];
                    if d >= b_prime_i && d < batch_bound && d < bound {
                        frame.prepend_buf.push(state.dist_value(s));
                    }
                }

                if !frame.prepend_buf.is_empty() {
                    queues[level].batch_prepend(&frame.prepend_buf);
                }

                frame.phase = FramePhase::ProcessBatch;
            }
        }
    }

    return_val
}
