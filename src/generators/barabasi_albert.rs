use petgraph::graph::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct BarabasiAlbertConfig {
    pub node_count: usize,
    pub m: usize,
    pub min_weight: f64,
    pub max_weight: f64,
}

pub fn generate_seeded(config: &BarabasiAlbertConfig, seed: u64) -> Graph<(), f64> {
    assert!(config.m > 0 && config.m < config.node_count);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut graph = Graph::new();
    let nodes: Vec<_> = (0..config.node_count)
        .map(|_| graph.add_node(()))
        .collect();

    let mut degrees = vec![0usize; config.node_count];

    for i in 0..config.m {
        for j in (i + 1)..config.m {
            let w = rng.gen_range(config.min_weight..=config.max_weight);
            graph.add_edge(nodes[i], nodes[j], w);
            let w2 = rng.gen_range(config.min_weight..=config.max_weight);
            graph.add_edge(nodes[j], nodes[i], w2);
            degrees[i] += 1;
            degrees[j] += 1;
        }
    }

    for new_node in config.m..config.node_count {
        let degree_sum: usize = degrees.iter().take(new_node).sum();
        if degree_sum == 0 {
            for target in 0..config.m {
                let w = rng.gen_range(config.min_weight..=config.max_weight);
                graph.add_edge(nodes[new_node], nodes[target], w);
                let w2 = rng.gen_range(config.min_weight..=config.max_weight);
                graph.add_edge(nodes[target], nodes[new_node], w2);
                degrees[new_node] += 1;
                degrees[target] += 1;
            }
            continue;
        }

        let mut attached = 0;
        let mut connected = vec![false; new_node];
        while attached < config.m {
            let r = rng.gen_range(0..degree_sum);
            let mut cumulative = 0;
            for target in 0..new_node {
                cumulative += degrees[target];
                if r < cumulative && !connected[target] {
                    connected[target] = true;
                    let w = rng.gen_range(config.min_weight..=config.max_weight);
                    graph.add_edge(nodes[new_node], nodes[target], w);
                    let w2 = rng.gen_range(config.min_weight..=config.max_weight);
                    graph.add_edge(nodes[target], nodes[new_node], w2);
                    degrees[new_node] += 1;
                    degrees[target] += 1;
                    attached += 1;
                    break;
                }
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
        let config = BarabasiAlbertConfig {
            node_count: 100,
            m: 3,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        assert_eq!(g.node_count(), 100);
    }

    #[test]
    fn has_edges() {
        let config = BarabasiAlbertConfig {
            node_count: 50,
            m: 2,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        assert!(g.edge_count() > 0);
    }

    #[test]
    fn deterministic() {
        let config = BarabasiAlbertConfig {
            node_count: 50,
            m: 3,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g1 = generate_seeded(&config, 99);
        let g2 = generate_seeded(&config, 99);
        assert_eq!(g1.edge_count(), g2.edge_count());
    }

    #[test]
    fn no_self_loops() {
        let config = BarabasiAlbertConfig {
            node_count: 100,
            m: 4,
            min_weight: 1.0,
            max_weight: 10.0,
        };
        let g = generate_seeded(&config, 42);
        for e in g.raw_edges() {
            assert_ne!(e.source(), e.target());
        }
    }
}
