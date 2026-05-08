use petgraph::graph::Graph;

pub trait GeneratorConfig {
    fn validate(&self) -> Result<(), String>;
}

pub trait GraphGenerator {
    type Config: GeneratorConfig;

    fn generate_weighted(&self, config: &Self::Config) -> Graph<(), f64>;
}
