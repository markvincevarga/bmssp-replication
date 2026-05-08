use super::traits::{GeneratorConfig, GraphGenerator};
use petgraph::graph::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct FixedEdgeCountConfig {
    pub node_count: usize,
    pub edge_count: usize,
    pub min_weight: f64,
    pub max_weight: f64,
}

impl FixedEdgeCountConfig {
    pub fn new(
        node_count: usize,
        edge_count: usize,
        min_weight: f64,
        max_weight: f64,
    ) -> Self {
        FixedEdgeCountConfig {
            node_count,
            edge_count,
            min_weight,
            max_weight,
        }
    }
}

impl GeneratorConfig for FixedEdgeCountConfig {
    fn validate(&self) -> Result<(), String> {
        if self.node_count == 0 {
            return Err("node_count must be greater than 0".to_string());
        }

        let max_edges = self.node_count * (self.node_count - 1);
        if self.edge_count > max_edges {
            return Err(format!(
                "edge_count ({}) exceeds maximum possible edges ({}) for {} nodes",
                self.edge_count, max_edges, self.node_count
            ));
        }

        if self.min_weight > self.max_weight {
            return Err("min_weight must be less than or equal to max_weight".to_string());
        }

        if self.min_weight < 0.0 {
            return Err("min_weight must be non-negative".to_string());
        }

        Ok(())
    }
}

pub struct FixedEdgeCountGenerator;

impl FixedEdgeCountGenerator {
    pub fn new() -> Self {
        FixedEdgeCountGenerator
    }
}

impl Default for FixedEdgeCountGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl FixedEdgeCountGenerator {
    pub fn generate_seeded(&self, config: &FixedEdgeCountConfig, seed: u64) -> Graph<(), f64> {
        Self::generate_with_rng(config, &mut StdRng::seed_from_u64(seed))
    }

    fn generate_with_rng(config: &FixedEdgeCountConfig, rng: &mut impl Rng) -> Graph<(), f64> {
        config.validate().expect("Invalid configuration");

        let mut graph = Graph::new();

        let nodes: Vec<_> = (0..config.node_count)
            .map(|_| graph.add_node(()))
            .collect();

        let mut added_edges = HashSet::new();
        while added_edges.len() < config.edge_count {
            let i = rng.gen_range(0..config.node_count);
            let j = rng.gen_range(0..config.node_count);
            if i != j {
                added_edges.insert((i, j));
            }
        }

        for (i, j) in added_edges {
            let weight = rng.gen_range(config.min_weight..=config.max_weight);
            graph.add_edge(nodes[i], nodes[j], weight);
        }

        graph
    }
}

impl GraphGenerator for FixedEdgeCountGenerator {
    type Config = FixedEdgeCountConfig;

    fn generate_weighted(&self, config: &Self::Config) -> Graph<(), f64> {
        Self::generate_with_rng(config, &mut rand::thread_rng())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::NodeIndex;

    #[test]
    fn test_config_validation_valid() {
        let config = FixedEdgeCountConfig::new(10, 50, 1.0, 10.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_zero_nodes() {
        let config = FixedEdgeCountConfig::new(0, 10, 1.0, 10.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_excessive_edges() {
        let config = FixedEdgeCountConfig::new(10, 100, 1.0, 10.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_weights() {
        let config = FixedEdgeCountConfig::new(10, 50, 10.0, 1.0);
        assert!(config.validate().is_err());

        let config = FixedEdgeCountConfig::new(10, 50, -1.0, 10.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_generation_exact_edge_count() {
        let generator = FixedEdgeCountGenerator::new();
        let config = FixedEdgeCountConfig::new(20, 50, 1.0, 10.0);
        let graph = generator.generate_weighted(&config);

        assert_eq!(graph.edge_count(), 50);
    }

    #[test]
    fn test_generation_node_count() {
        let generator = FixedEdgeCountGenerator::new();
        let config = FixedEdgeCountConfig::new(20, 50, 1.0, 10.0);
        let graph = generator.generate_weighted(&config);

        assert_eq!(graph.node_count(), 20);
    }

    #[test]
    fn test_generation_no_self_loops() {
        let generator = FixedEdgeCountGenerator::new();
        let config = FixedEdgeCountConfig::new(10, 30, 1.0, 10.0);
        let graph = generator.generate_weighted(&config);

        for edge in graph.raw_edges() {
            assert_ne!(edge.source(), edge.target());
        }
    }

    #[test]
    fn test_generation_weight_range() {
        let generator = FixedEdgeCountGenerator::new();
        let config = FixedEdgeCountConfig::new(10, 20, 5.0, 5.0);
        let graph = generator.generate_weighted(&config);

        for edge in graph.raw_edges() {
            assert_eq!(edge.weight, 5.0);
        }
    }

    #[test]
    fn test_generation_zero_edges() {
        let generator = FixedEdgeCountGenerator::new();
        let config = FixedEdgeCountConfig::new(10, 0, 1.0, 10.0);
        let graph = generator.generate_weighted(&config);

        assert_eq!(graph.node_count(), 10);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_generation_maximum_edges() {
        let generator = FixedEdgeCountGenerator::new();
        let n = 5;
        let max_edges = n * (n - 1);
        let config = FixedEdgeCountConfig::new(n, max_edges, 1.0, 10.0);
        let graph = generator.generate_weighted(&config);

        assert_eq!(graph.node_count(), n);
        assert_eq!(graph.edge_count(), max_edges);

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let has_edge = graph
                        .find_edge(NodeIndex::new(i), NodeIndex::new(j))
                        .is_some();
                    assert!(
                        has_edge,
                        "Edge from {} to {} should exist in complete graph",
                        i,
                        j
                    );
                }
            }
        }
    }
}
