use petgraph::graph::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

pub struct WattsStrogatzConfig {
    pub node_count: usize,
    pub k: usize,
    pub beta: f64,
    pub min_weight: f64,
    pub max_weight: f64,
}

pub fn generate_seeded(config: &WattsStrogatzConfig, seed: u64) -> Graph<(), f64> {
    assert!(config.k > 0 && config.k < config.node_count);
    assert!((0.0..=1.0).contains(&config.beta));
    let mut rng = StdRng::seed_from_u64(seed);
    let n = config.node_count;
    let half_k = config.k / 2;

    let mut edges: HashSet<(usize, usize)> = HashSet::new();

    for i in 0..n {
        for offset in 1..=half_k {
            let j = (i + offset) % n;
            edges.insert((i, j));
            edges.insert((j, i));
        }
    }

    let ring_edges: Vec<(usize, usize)> = edges
        .iter()
        .filter(|(a, b)| a < b)
        .copied()
        .collect();

    for (i, j) in &ring_edges {
        if rng.gen::<f64>() < config.beta {
            edges.remove(&(*i, *j));
            edges.remove(&(*j, *i));

            let mut new_target = rng.gen_range(0..n);
            let mut attempts = 0;
            while (new_target == *i || edges.contains(&(*i, new_target))) && attempts < n {
                new_target = rng.gen_range(0..n);
                attempts += 1;
            }
            if attempts < n {
                edges.insert((*i, new_target));
                edges.insert((new_target, *i));
            } else {
                edges.insert((*i, *j));
                edges.insert((*j, *i));
            }
        }
    }

    let mut graph = Graph::new();
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();
    for (a, b) in &edges {
        let w = rng.gen_range(config.min_weight..=config.max_weight);
        graph.add_edge(nodes[*a], nodes[*b], w);
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_node_count() {
        let config = WattsStrogatzConfig {
            node_count: 100,
            k: 4,
            beta: 0.3,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        assert_eq!(g.node_count(), 100);
    }

    #[test]
    fn zero_beta_is_ring_lattice() {
        let config = WattsStrogatzConfig {
            node_count: 20,
            k: 4,
            beta: 0.0,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        assert_eq!(g.edge_count(), 20 * 4);
    }

    #[test]
    fn deterministic() {
        let config = WattsStrogatzConfig {
            node_count: 50,
            k: 6,
            beta: 0.5,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g1 = generate_seeded(&config, 99);
        let g2 = generate_seeded(&config, 99);
        assert_eq!(g1.edge_count(), g2.edge_count());
    }

    #[test]
    fn no_self_loops() {
        let config = WattsStrogatzConfig {
            node_count: 50,
            k: 6,
            beta: 0.5,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        for e in g.raw_edges() {
            assert_ne!(e.source(), e.target());
        }
    }
}
