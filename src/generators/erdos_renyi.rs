use super::traits::{GeneratorConfig, GraphGenerator};
use petgraph::graph::Graph;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct ErdosRenyiConfig {
    pub node_count: usize,
    pub edge_probability: f64,
    pub min_weight: f64,
    pub max_weight: f64,
}

impl ErdosRenyiConfig {
    pub fn new(
        node_count: usize,
        edge_probability: f64,
        min_weight: f64,
        max_weight: f64,
    ) -> Self {
        ErdosRenyiConfig {
            node_count,
            edge_probability,
            min_weight,
            max_weight,
        }
    }
}

impl GeneratorConfig for ErdosRenyiConfig {
    fn validate(&self) -> Result<(), String> {
        if self.node_count == 0 {
            return Err("node_count must be greater than 0".to_string());
        }

        if !(0.0..=1.0).contains(&self.edge_probability) {
            return Err("edge_probability must be between 0.0 and 1.0".to_string());
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

pub struct ErdosRenyiGenerator;

impl ErdosRenyiGenerator {
    pub fn new() -> Self {
        ErdosRenyiGenerator
    }
}

impl Default for ErdosRenyiGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphGenerator for ErdosRenyiGenerator {
    type Config = ErdosRenyiConfig;

    fn generate_weighted(&self, config: &Self::Config) -> Graph<(), f64> {
        config.validate().expect("Invalid configuration");

        let mut graph = Graph::new();
        let mut rng = rand::thread_rng();

        let nodes: Vec<_> = (0..config.node_count)
            .map(|_| graph.add_node(()))
            .collect();

        for i in 0..config.node_count {
            for j in 0..config.node_count {
                if i != j && rng.gen::<f64>() < config.edge_probability {
                    let weight = rng.gen_range(config.min_weight..=config.max_weight);
                    graph.add_edge(nodes[i], nodes[j], weight);
                }
            }
        }

        graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation_valid() {
        let config = ErdosRenyiConfig::new(10, 0.5, 1.0, 10.0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_zero_nodes() {
        let config = ErdosRenyiConfig::new(0, 0.5, 1.0, 10.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_probability() {
        let config = ErdosRenyiConfig::new(10, 1.5, 1.0, 10.0);
        assert!(config.validate().is_err());

        let config = ErdosRenyiConfig::new(10, -0.1, 1.0, 10.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_invalid_weights() {
        let config = ErdosRenyiConfig::new(10, 0.5, 10.0, 1.0);
        assert!(config.validate().is_err());

        let config = ErdosRenyiConfig::new(10, 0.5, -1.0, 10.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_generation_node_count() {
        let generator = ErdosRenyiGenerator::new();
        let config = ErdosRenyiConfig::new(20, 0.5, 1.0, 10.0);
        let graph = generator.generate_weighted(&config);

        assert_eq!(graph.node_count(), 20);
    }

    #[test]
    fn test_generation_zero_probability() {
        let generator = ErdosRenyiGenerator::new();
        let config = ErdosRenyiConfig::new(10, 0.0, 1.0, 10.0);
        let graph = generator.generate_weighted(&config);

        assert_eq!(graph.node_count(), 10);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_generation_weight_range() {
        let generator = ErdosRenyiGenerator::new();
        let config = ErdosRenyiConfig::new(5, 1.0, 5.0, 5.0);
        let graph = generator.generate_weighted(&config);

        for edge in graph.raw_edges() {
            assert_eq!(edge.weight, 5.0);
        }
    }
}
