use petgraph::graph::Graph;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

pub const H3_DENSITY: usize = 3;
pub const H3_MAX_WEIGHT: u64 = 100_000;
pub const H3_DEFAULT_SEED: u64 = 1971;

#[derive(Debug, Clone)]
pub struct CastroH3Config {
    pub node_count: usize,
    pub density: usize,
    pub max_weight: u64,
}

impl Default for CastroH3Config {
    fn default() -> Self {
        Self {
            node_count: 1024,
            density: H3_DENSITY,
            max_weight: H3_MAX_WEIGHT,
        }
    }
}

pub fn generate_seeded(config: &CastroH3Config, seed: u64) -> Graph<(), f64> {
    let n = config.node_count;
    assert!(n > 0, "node_count must be > 0");

    let max_simple_edges = n.saturating_mul(n.saturating_sub(1));
    let m_target = (n.saturating_mul(config.density)).min(max_simple_edges);
    let create_m_signed: isize = m_target as isize - n as isize;
    let extra_edges: usize = if create_m_signed > 0 {
        create_m_signed as usize
    } else {
        0
    };
    let total_m = n + extra_edges;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut graph: Graph<(), f64> = Graph::with_capacity(n, total_m);
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

    let mut perm: Vec<usize> = (1..=n).collect();
    perm.shuffle(&mut rng);

    let mut adj: Vec<std::collections::HashSet<usize>> = (0..=n)
        .map(|_| std::collections::HashSet::new())
        .collect();

    for i in 0..n {
        let u = perm[i];
        let v = perm[(i + 1) % n];
        let w = rng.gen_range(0..=config.max_weight) as f64;
        graph.add_edge(nodes[u - 1], nodes[v - 1], w);
        adj[u].insert(v);
    }

    let mut remaining = extra_edges;
    while remaining > 0 {
        let i = rng.gen_range(1..=n);
        let j = rng.gen_range(1..=n);
        if i == j {
            continue;
        }
        if !adj[i].insert(j) {
            continue;
        }
        let w = rng.gen_range(0..=config.max_weight) as f64;
        graph.add_edge(nodes[i - 1], nodes[j - 1], w);
        remaining -= 1;
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::NodeIndex;
    use petgraph::visit::EdgeRef;
    use std::collections::VecDeque;

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
        let config = CastroH3Config {
            node_count: 256,
            ..Default::default()
        };
        let g = generate_seeded(&config, 1971);
        assert_eq!(g.node_count(), 256);
        assert_eq!(g.edge_count(), 256 * 3);
    }

    #[test]
    fn strongly_connected_via_hamiltonian() {
        let config = CastroH3Config {
            node_count: 512,
            ..Default::default()
        };
        let g = generate_seeded(&config, 1971);
        assert_eq!(bfs_reach_count(&g, 0), 512);
    }

    #[test]
    fn no_max_outdegree_cap() {
        let config = CastroH3Config {
            node_count: 64,
            density: 32,
            ..Default::default()
        };
        let g = generate_seeded(&config, 1971);
        let max_out = (0..g.node_count())
            .map(|i| g.edges(NodeIndex::new(i)).count())
            .max()
            .unwrap();
        assert!(max_out > 4);
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let config = CastroH3Config {
            node_count: 128,
            ..Default::default()
        };
        let g1 = generate_seeded(&config, 1971);
        let g2 = generate_seeded(&config, 1971);
        assert_eq!(g1.edge_count(), g2.edge_count());
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
}
