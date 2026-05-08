use petgraph::graph::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct GridConfig {
    pub node_count: usize,
    pub min_weight: f64,
    pub max_weight: f64,
}

pub fn generate_seeded(config: &GridConfig, seed: u64) -> Graph<(), f64> {
    assert!(config.node_count > 0);
    let mut rng = StdRng::seed_from_u64(seed);

    let cols = (config.node_count as f64).sqrt().ceil() as usize;
    let rows = (config.node_count + cols - 1) / cols;
    let n = config.node_count;

    let mut graph = Graph::new();
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

    for idx in 0..n {
        let r = idx / cols;
        let c = idx % cols;

        if c + 1 < cols {
            let right = r * cols + c + 1;
            if right < n {
                let w = rng.gen_range(config.min_weight..=config.max_weight);
                graph.add_edge(nodes[idx], nodes[right], w);
                let w2 = rng.gen_range(config.min_weight..=config.max_weight);
                graph.add_edge(nodes[right], nodes[idx], w2);
            }
        }

        if r + 1 < rows {
            let down = (r + 1) * cols + c;
            if down < n {
                let w = rng.gen_range(config.min_weight..=config.max_weight);
                graph.add_edge(nodes[idx], nodes[down], w);
                let w2 = rng.gen_range(config.min_weight..=config.max_weight);
                graph.add_edge(nodes[down], nodes[idx], w2);
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
        let config = GridConfig {
            node_count: 100,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        assert_eq!(g.node_count(), 100);
    }

    #[test]
    fn perfect_square_edges() {
        let config = GridConfig {
            node_count: 9,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        assert_eq!(g.edge_count(), 24);
    }

    #[test]
    fn deterministic() {
        let config = GridConfig {
            node_count: 50,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g1 = generate_seeded(&config, 99);
        let g2 = generate_seeded(&config, 99);
        assert_eq!(g1.edge_count(), g2.edge_count());
    }

    #[test]
    fn no_self_loops() {
        let config = GridConfig {
            node_count: 100,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        for e in g.raw_edges() {
            assert_ne!(e.source(), e.target());
        }
    }
}
