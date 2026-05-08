use criterion::{black_box, BenchmarkId, Criterion};
use duan::algorithms::PathFinder;
use duan::graph_source::GraphSource;
use petgraph::graph::NodeIndex;
use std::env;
#[cfg(feature = "track-alloc")]
use std::io::Write;

pub struct BenchmarkConfig {
    pub algorithms: Option<Vec<String>>,
    pub sizes: Vec<usize>,
    pub graph_sources: Vec<GraphSource>,
    pub min_weight: f64,
    pub max_weight: f64,
    pub graph_instances: usize,
    pub base_seed: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            algorithms: None,
            sizes: vec![1000, 5000],
            graph_sources: vec![
                GraphSource::FixedEdgeCount { edge_factor: 4 },
                GraphSource::FixedEdgeCount { edge_factor: 8 },
                GraphSource::FixedEdgeCount { edge_factor: 16 },
                GraphSource::FixedEdgeCount { edge_factor: 32 },
            ],
            min_weight: 1.0,
            max_weight: 10.0,
            graph_instances: 1,
            base_seed: 42,
        }
    }
}

impl BenchmarkConfig {
    pub fn from_env() -> Self {
        let defaults = Self::default();

        let algorithms = env::var("BENCH_ALGORITHMS").ok().map(|s| {
            s.split(',')
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .collect()
        });

        let sizes = env::var("BENCH_SIZES")
            .ok()
            .map(|s| parse_csv_usize(&s))
            .unwrap_or(defaults.sizes);

        let graph_sources = if let Ok(s) = env::var("BENCH_GRAPH_SOURCES") {
            parse_graph_sources(&s)
        } else if let Ok(s) = env::var("BENCH_EDGE_FACTORS") {
            parse_csv_usize(&s)
                .into_iter()
                .map(|ef| GraphSource::FixedEdgeCount { edge_factor: ef })
                .collect()
        } else {
            defaults.graph_sources
        };

        let min_weight = env::var("BENCH_MIN_WEIGHT")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(defaults.min_weight);

        let max_weight = env::var("BENCH_MAX_WEIGHT")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(defaults.max_weight);

        let graph_instances = env::var("BENCH_GRAPH_INSTANCES")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(1)
            .max(1);

        let base_seed = env::var("BENCH_BASE_SEED")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(42);

        let config = Self {
            algorithms,
            sizes,
            graph_sources,
            min_weight,
            max_weight,
            graph_instances,
            base_seed,
        };

        match config.validate() {
            Ok(()) => config,
            Err(e) => {
                eprintln!("Invalid benchmark config from env: {}", e);
                eprintln!("Using default configuration");
                Self::default()
            }
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.sizes.is_empty() {
            return Err("sizes cannot be empty".to_string());
        }

        if self.graph_sources.is_empty() {
            return Err("graph_sources cannot be empty".to_string());
        }

        if self.sizes.iter().any(|&s| s == 0) {
            return Err("all sizes must be greater than 0".to_string());
        }

        if self.min_weight > self.max_weight {
            return Err(format!(
                "min_weight ({}) must be <= max_weight ({})",
                self.min_weight, self.max_weight
            ));
        }

        if self.min_weight < 0.0 || self.max_weight < 0.0 {
            return Err("weights must be non-negative".to_string());
        }

        Ok(())
    }
}

fn parse_csv_usize(s: &str) -> Vec<usize> {
    s.split(',')
        .filter_map(|v| v.trim().parse().ok())
        .collect()
}

fn parse_graph_sources(s: &str) -> Vec<GraphSource> {
    s.split(',')
        .filter_map(|v| {
            let v = v.trim();
            if v.is_empty() {
                return None;
            }
            match GraphSource::parse(v) {
                Ok(src) => Some(src),
                Err(e) => {
                    eprintln!("warning: skipping invalid graph source '{}': {}", v, e);
                    None
                }
            }
        })
        .collect()
}

pub struct AlgorithmEntry {
    pub name: &'static str,
    pub finder: Box<dyn PathFinder>,
}

pub struct AlgorithmRegistry {
    algorithms: Vec<AlgorithmEntry>,
    filter: Option<Vec<String>>,
}

impl AlgorithmRegistry {
    pub fn new(filter: Option<Vec<String>>) -> Self {
        Self {
            algorithms: Vec::new(),
            filter,
        }
    }

    pub fn register<T: PathFinder + 'static>(mut self, name: &'static str, finder: T) -> Self {
        let dominated = match &self.filter {
            Some(allowed) => allowed.iter().any(|a| a == name),
            None => true,
        };
        if dominated {
            self.algorithms.push(AlgorithmEntry {
                name,
                finder: Box::new(finder),
            });
        }
        self
    }

    pub fn algorithms(&self) -> &[AlgorithmEntry] {
        &self.algorithms
    }
}

fn warm_up_time_for_size(size: usize) -> std::time::Duration {
    match size {
        s if s >= 4_000_000 => std::time::Duration::from_millis(100),
        s if s >= 500_000 => std::time::Duration::from_millis(500),
        _ => std::time::Duration::from_secs(3),
    }
}

fn measurement_time_for_size(size: usize) -> std::time::Duration {
    match size {
        s if s >= 4_000_000 => std::time::Duration::from_secs(10),
        s if s >= 500_000 => std::time::Duration::from_secs(5),
        _ => std::time::Duration::from_secs(5),
    }
}

fn sample_size_for_size(size: usize) -> usize {
    match size {
        s if s >= 4_000_000 => 10,
        s if s >= 500_000 => 10,
        _ => 100,
    }
}


pub fn benchmark_algorithms_comprehensive(
    c: &mut Criterion,
    config: &BenchmarkConfig,
    registry: &AlgorithmRegistry,
) {
    for entry in registry.algorithms() {
        let group_name = format!("{}_comprehensive", entry.name);
        let mut group = c.benchmark_group(&group_name);

        for &size in &config.sizes {
            group.warm_up_time(warm_up_time_for_size(size));
            group.measurement_time(measurement_time_for_size(size));
            group.sample_size(sample_size_for_size(size));

            for source in &config.graph_sources {
                for gi in 0..config.graph_instances {
                    let graph = duan::graph_store::load_or_generate_instance(
                        size, source, gi, config.base_seed,
                    );

                    let preprocess_result = entry.finder.preprocess(&graph, NodeIndex::new(0));

                    let bench_id = if config.graph_instances > 1 {
                        format!("size_{}_{}_g{}", size, source.cache_key(), gi)
                    } else {
                        format!("size_{}_{}", size, source.cache_key())
                    };
                    group.bench_with_input(
                        BenchmarkId::new("comprehensive", &bench_id),
                        &preprocess_result,
                        |b, prep| {
                            b.iter(|| {
                                let result = entry.finder.shortest_paths(
                                    black_box(&prep.graph),
                                    black_box(prep.source),
                                );
                                black_box(result);
                            });
                        },
                    );
                }
            }
        }

        group.finish();
    }
}

#[cfg(feature = "track-alloc")]
pub fn benchmark_memory(config: &BenchmarkConfig, registry: &AlgorithmRegistry) {
    let base = std::path::Path::new("target/criterion");

    for entry in registry.algorithms() {
        for &size in &config.sizes {
            for source in &config.graph_sources {
                for gi in 0..config.graph_instances {
                    let graph = duan::graph_store::load_or_generate_instance(
                        size, source, gi, config.base_seed,
                    );
                    let preprocess_result = entry.finder.preprocess(&graph, NodeIndex::new(0));

                    duan::tracking_alloc::reset();
                    let _result = entry.finder.shortest_paths(
                        &preprocess_result.graph,
                        preprocess_result.source,
                    );
                    let peak = duan::tracking_alloc::peak_bytes();

                    let source_key = source.cache_key();
                    let dir_name = if config.graph_instances > 1 {
                        format!("size_{}_{}_g{}", size, source_key, gi)
                    } else {
                        format!("size_{}_{}", size, source_key)
                    };
                    let dir = base
                        .join(format!("{}_comprehensive", entry.name))
                        .join("comprehensive")
                        .join(&dir_name);
                    std::fs::create_dir_all(&dir).unwrap();

                    let json = format!("{{\"peak_bytes\":{}}}", peak);
                    std::fs::write(dir.join("memory.json"), &json).unwrap();

                    let gi_label = if config.graph_instances > 1 {
                        format!(" g{gi}")
                    } else {
                        String::new()
                    };
                    eprintln!(
                        "[memory] {:20} size={:>8} {:12}{} peak={:.1} MB",
                        entry.name, size, source_key, gi_label,
                        peak as f64 / 1_048_576.0
                    );
                }
            }
        }
    }
    eprintln!("[memory] Results written to target/criterion/*/comprehensive/*/memory.json");
}
