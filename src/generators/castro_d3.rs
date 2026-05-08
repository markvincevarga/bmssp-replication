use petgraph::graph::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const D3_AVERAGE_OUTDEGREE: usize = 3;
pub const D3_MAX_WEIGHT: u64 = 100_000;
pub const D3_DEFAULT_SEED: u64 = 1;

#[derive(Debug, Clone)]
pub struct CastroD3Config {
    pub node_count: usize,
    pub average_outdegree: usize,
    pub max_weight: u64,
}

impl Default for CastroD3Config {
    fn default() -> Self {
        Self {
            node_count: 1024,
            average_outdegree: D3_AVERAGE_OUTDEGREE,
            max_weight: D3_MAX_WEIGHT,
        }
    }
}

pub fn generate_seeded(config: &CastroD3Config, seed: u64) -> Graph<(), f64> {
    let n = config.node_count;
    assert!(n > 0, "node_count must be > 0");
    let avg = config.average_outdegree;
    let max_deg = avg + 1;
    let target_m = n.checked_mul(avg).expect("edge count overflow");

    let mut rng = StdRng::seed_from_u64(seed);
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n + 1];

    let can_add_edge = |adj: &Vec<Vec<usize>>, i: usize, j: usize| -> bool {
        if adj[i].len() >= max_deg {
            return false;
        }
        if i == j {
            return false;
        }
        !adj[i].contains(&j)
    };

    for i in 2..=n {
        loop {
            let j = rng.gen_range(1..i);
            if can_add_edge(&adj, i, j) && can_add_edge(&adj, j, i) {
                adj[i].push(j);
                adj[j].push(i);
                break;
            }
        }
    }

    let mut adj2: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    let olds = rng.gen_range(1..=n);
    let mut count = 0usize;
    {
        let mut stack: Vec<(usize, usize, usize)> = Vec::with_capacity(n);
        stack.push((olds, 0, 0));
        let mut visited_mark = vec![false; n + 1];
        visited_mark[olds] = true;
        while let Some((u, dad, idx)) = stack.pop() {
            if idx == 0 {
                count += 1;
            }
            if idx < adj[u].len() {
                stack.push((u, dad, idx + 1));
                let v = adj[u][idx];
                if v != dad && !visited_mark[v] {
                    visited_mark[v] = true;
                    adj2[u].push(v);
                    stack.push((v, u, 0));
                }
            }
        }
    }
    assert_eq!(count, n, "tree DFS did not cover all vertices");

    let s: usize = 1;
    let mut adj = adj2;
    if olds != s {
        for u in 1..=n {
            for v in adj[u].iter_mut() {
                if *v == olds {
                    *v = s;
                } else if *v == s {
                    *v = olds;
                }
            }
        }
        adj.swap(s, olds);
    }

    let mut graph: Graph<(), f64> = Graph::with_capacity(n, target_m);
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

    let mut emitted = 0usize;
    let mut adj_directed: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for u in 1..=n {
        for &v in &adj[u] {
            let w = rng.gen_range(0..=config.max_weight) as f64;
            graph.add_edge(nodes[u - 1], nodes[v - 1], w);
            adj_directed[u].push(v);
            emitted += 1;
        }
    }

    while emitted < target_m {
        let i = rng.gen_range(1..=n);
        let j = rng.gen_range(1..=n);
        if i == j {
            continue;
        }
        if adj_directed[i].len() >= max_deg {
            continue;
        }
        if adj_directed[i].contains(&j) {
            continue;
        }
        let w = rng.gen_range(0..=config.max_weight) as f64;
        graph.add_edge(nodes[i - 1], nodes[j - 1], w);
        adj_directed[i].push(j);
        emitted += 1;
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::NodeIndex;
    use petgraph::visit::EdgeRef;
    use std::collections::VecDeque;

    fn count_out_neighbors(g: &Graph<(), f64>) -> Vec<usize> {
        (0..g.node_count())
            .map(|i| g.edges(NodeIndex::new(i)).count())
            .collect()
    }

    fn bfs_reach_count(g: &Graph<(), f64>, src: usize) -> usize {
        let n = g.node_count();
        let mut seen = vec![false; n];
        let mut q = VecDeque::new();
        q.push_back(src);
        seen[src] = true;
        let mut count = 1;
        while let Some(u) = q.pop_front() {
            for e in g.edges(NodeIndex::new(u)) {
                let v = e.target().index();
                if !seen[v] {
                    seen[v] = true;
                    count += 1;
                    q.push_back(v);
                }
            }
        }
        count
    }

    #[test]
    fn correct_size() {
        let config = CastroD3Config {
            node_count: 256,
            ..Default::default()
        };
        let g = generate_seeded(&config, 1);
        assert_eq!(g.node_count(), 256);
        assert_eq!(g.edge_count(), 256 * 3);
    }

    #[test]
    fn max_outdegree_at_most_four() {
        let config = CastroD3Config {
            node_count: 1024,
            ..Default::default()
        };
        let g = generate_seeded(&config, 1);
        for d in count_out_neighbors(&g) {
            assert!(d <= 4, "vertex out-degree {} exceeds cap 4", d);
        }
    }

    #[test]
    fn vertex_zero_reaches_all() {
        let config = CastroD3Config {
            node_count: 512,
            ..Default::default()
        };
        let g = generate_seeded(&config, 1);
        assert_eq!(bfs_reach_count(&g, 0), 512);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let config = CastroD3Config {
            node_count: 128,
            ..Default::default()
        };
        let g1 = generate_seeded(&config, 42);
        let g2 = generate_seeded(&config, 42);
        let e1: Vec<_> = g1
            .raw_edges()
            .iter()
            .map(|e| (e.source().index(), e.target().index(), e.weight))
            .collect();
        let e2: Vec<_> = g2
            .raw_edges()
            .iter()
            .map(|e| (e.source().index(), e.target().index(), e.weight))
            .collect();
        assert_eq!(e1, e2);
    }

    #[test]
    fn no_self_loops_or_duplicates() {
        let config = CastroD3Config {
            node_count: 256,
            ..Default::default()
        };
        let g = generate_seeded(&config, 1);
        let mut seen = std::collections::HashSet::new();
        for e in g.raw_edges() {
            let u = e.source().index();
            let v = e.target().index();
            assert_ne!(u, v);
            assert!(seen.insert((u, v)), "duplicate edge ({},{})", u, v);
        }
    }
}
