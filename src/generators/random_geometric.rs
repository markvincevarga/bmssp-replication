use petgraph::graph::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct RandomGeometricConfig {
    pub node_count: usize,
    pub radius: f64,
    pub min_weight: f64,
    pub max_weight: f64,
}

pub fn generate_seeded(config: &RandomGeometricConfig, seed: u64) -> Graph<(), f64> {
    assert!(config.radius > 0.0);
    let mut rng = StdRng::seed_from_u64(seed);
    let n = config.node_count;
    let r_sq = config.radius * config.radius;

    let positions: Vec<(f64, f64)> = (0..n)
        .map(|_| (rng.gen::<f64>(), rng.gen::<f64>()))
        .collect();

    let mut graph = Graph::new();
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = positions[i].0 - positions[j].0;
            let dy = positions[i].1 - positions[j].1;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq <= r_sq {
                let dist = dist_sq.sqrt();
                let weight = dist.max(config.min_weight).min(config.max_weight);
                graph.add_edge(nodes[i], nodes[j], weight);
                graph.add_edge(nodes[j], nodes[i], weight);
            }
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_node_count() {
        let config = RandomGeometricConfig {
            node_count: 100,
            radius: 0.3,
            min_weight: 0.001,
            max_weight: 1.0,
        };
        let g = generate_seeded(&config, 42);
        assert_eq!(g.node_count(), 100);
    }

    #[test]
    fn large_radius_is_nearly_complete() {
        let config = RandomGeometricConfig {
            node_count: 20,
            radius: 2.0,
            min_weight: 0.001,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        let max_edges = 20 * 19;
        assert_eq!(g.edge_count(), max_edges);
    }

    #[test]
    fn tiny_radius_is_sparse() {
        let config = RandomGeometricConfig {
            node_count: 100,
            radius: 0.01,
            min_weight: 0.001,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        assert!(g.edge_count() < 100);
    }

    #[test]
    fn deterministic() {
        let config = RandomGeometricConfig {
            node_count: 50,
            radius: 0.2,
            min_weight: 0.001,
            max_weight: 10.0,
        };
        let g1 = generate_seeded(&config, 99);
        let g2 = generate_seeded(&config, 99);
        assert_eq!(g1.edge_count(), g2.edge_count());
    }

    #[test]
    fn no_self_loops() {
        let config = RandomGeometricConfig {
            node_count: 50,
            radius: 0.3,
            min_weight: 0.001,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        for e in g.raw_edges() {
            assert_ne!(e.source(), e.target());
        }
    }
}
