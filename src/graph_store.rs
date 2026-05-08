use crate::dimacs;
use crate::generators::barabasi_albert::{self, BarabasiAlbertConfig};
use crate::generators::castro_d3::{self, CastroD3Config, D3_DEFAULT_SEED};
use crate::generators::castro_grid::{
    self, CastroGridConfig, CastroGridShape, CastroGridWeights, CASTRO_GRID_DEFAULT_SEED,
};
use crate::generators::castro_h3::{self, CastroH3Config, H3_DEFAULT_SEED};
use crate::generators::grid::{self, GridConfig};
use crate::generators::random_geometric::{self, RandomGeometricConfig};
use crate::generators::watts_strogatz::{self, WattsStrogatzConfig};
use crate::generators::{FixedEdgeCountConfig, FixedEdgeCountGenerator};
use crate::graph_repr::CsrAosGraph;
use crate::graph_source::GraphSource;
use petgraph::graph::Graph;
use std::fs::{self, File};
use std::io::BufReader;
use std::io::BufWriter;
use std::path::PathBuf;

pub const BASE_SEED: u64 = 42;
#[deprecated(note = "use BASE_SEED instead")]
pub const SEED: u64 = BASE_SEED;
pub const MIN_WEIGHT: f64 = 1.0;
pub const MAX_WEIGHT: f64 = 10.0;
pub const COMMIT_MAX_NODES: usize = 131_072;

pub const GRAPH_SIZES: [usize; 13] = [
    1 << 9,
    1 << 10,
    1 << 11,
    1 << 12,
    1 << 13,
    1 << 14,
    1 << 15,
    1 << 16,
    1 << 17,
    1 << 18,
    1 << 19,
    1 << 20,
    1 << 21,
];

pub const EDGE_FACTORS: [usize; 4] = [4, 8, 16, 32];

pub fn graph_filename(node_count: usize, edge_factor: usize) -> String {
    format!("n{}_ef{}.bin", node_count, edge_factor)
}

pub fn graph_filename_for_source(node_count: usize, source: &GraphSource) -> String {
    format!("n{}_{}.bin", node_count, source.cache_key())
}

fn committed_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("graphs")
}

fn cache_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("graphs")
}

fn dimacs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("graphs")
        .join("dimacs")
}

pub fn graph_filename_for_instance(
    node_count: usize,
    source: &GraphSource,
    graph_index: usize,
) -> String {
    if source.ignores_node_count() {
        if graph_index == 0 {
            format!("{}.bin", source.cache_key())
        } else {
            format!("{}_g{}.bin", source.cache_key(), graph_index)
        }
    } else if graph_index == 0 {
        format!("n{}_{}.bin", node_count, source.cache_key())
    } else {
        format!("n{}_{}_g{}.bin", node_count, source.cache_key(), graph_index)
    }
}

pub fn load_or_generate(node_count: usize, source: &GraphSource) -> Graph<(), f64> {
    load_or_generate_instance(node_count, source, 0, BASE_SEED)
}

pub fn load_or_generate_instance(
    node_count: usize,
    source: &GraphSource,
    graph_index: usize,
    base_seed: u64,
) -> Graph<(), f64> {
    let filename = graph_filename_for_instance(node_count, source, graph_index);

    let committed = committed_dir().join(&filename);
    if committed.exists() {
        let f = File::open(&committed).expect("failed to open committed graph");
        let csr = CsrAosGraph::read_from(&mut BufReader::new(f)).expect("failed to read graph");
        return csr.to_petgraph();
    }

    let cached = cache_dir().join(&filename);
    if cached.exists() {
        let f = File::open(&cached).expect("failed to open cached graph");
        let csr = CsrAosGraph::read_from(&mut BufReader::new(f)).expect("failed to read graph");
        return csr.to_petgraph();
    }

    let seed = base_seed + graph_index as u64;
    let graph = generate(node_count, source, seed);

    let csr = CsrAosGraph::from_petgraph(&graph);
    fs::create_dir_all(cache_dir()).ok();
    if let Ok(f) = File::create(&cached) {
        let _ = csr.write_to(&mut BufWriter::new(f));
    }

    graph
}

fn generate(node_count: usize, source: &GraphSource, seed: u64) -> Graph<(), f64> {
    match source {
        GraphSource::FixedEdgeCount { edge_factor } => {
            let config = FixedEdgeCountConfig::new(
                node_count,
                node_count * edge_factor,
                MIN_WEIGHT,
                MAX_WEIGHT,
            );
            FixedEdgeCountGenerator::new().generate_seeded(&config, seed)
        }
        GraphSource::BarabasiAlbert { m } => {
            let config = BarabasiAlbertConfig {
                node_count,
                m: *m,
                min_weight: MIN_WEIGHT,
                max_weight: MAX_WEIGHT,
            };
            barabasi_albert::generate_seeded(&config, seed)
        }
        GraphSource::WattsStrogatz { k, beta } => {
            let config = WattsStrogatzConfig {
                node_count,
                k: *k,
                beta: *beta,
                min_weight: MIN_WEIGHT,
                max_weight: MAX_WEIGHT,
            };
            watts_strogatz::generate_seeded(&config, seed)
        }
        GraphSource::RandomGeometric { radius } => {
            let config = RandomGeometricConfig {
                node_count,
                radius: *radius,
                min_weight: MIN_WEIGHT,
                max_weight: MAX_WEIGHT,
            };
            random_geometric::generate_seeded(&config, seed)
        }
        GraphSource::Grid => {
            let config = GridConfig {
                node_count,
                min_weight: MIN_WEIGHT,
                max_weight: MAX_WEIGHT,
            };
            grid::generate_seeded(&config, seed)
        }
        GraphSource::CastroD3 => {
            let config = CastroD3Config {
                node_count,
                ..CastroD3Config::default()
            };
            castro_d3::generate_seeded(&config, D3_DEFAULT_SEED)
        }
        GraphSource::CastroH3 => {
            let config = CastroH3Config {
                node_count,
                ..CastroH3Config::default()
            };
            castro_h3::generate_seeded(&config, H3_DEFAULT_SEED)
        }
        GraphSource::CastroSGridED => {
            let config = CastroGridConfig::from_node_count(
                node_count,
                CastroGridShape::Square,
                CastroGridWeights::Euclidean,
            );
            castro_grid::generate_seeded(&config, CASTRO_GRID_DEFAULT_SEED)
        }
        GraphSource::CastroRGridED => {
            let config = CastroGridConfig::from_node_count(
                node_count,
                CastroGridShape::Rectangular,
                CastroGridWeights::Euclidean,
            );
            castro_grid::generate_seeded(&config, CASTRO_GRID_DEFAULT_SEED)
        }
        GraphSource::CastroSGridR => {
            let config = CastroGridConfig::from_node_count(
                node_count,
                CastroGridShape::Square,
                CastroGridWeights::RandomInt,
            );
            castro_grid::generate_seeded(&config, CASTRO_GRID_DEFAULT_SEED)
        }
        GraphSource::CastroRGridR => {
            let config = CastroGridConfig::from_node_count(
                node_count,
                CastroGridShape::Rectangular,
                CastroGridWeights::RandomInt,
            );
            castro_grid::generate_seeded(&config, CASTRO_GRID_DEFAULT_SEED)
        }
        GraphSource::DimacsRoad { name } => {
            let path = resolve_dimacs_path(name);
            dimacs::read_gr_file(&path).unwrap_or_else(|e| {
                panic!(
                    "DIMACS road file '{}' not readable at {} ({}). \
                     Run `scripts/fetch_dimacs_roads.sh` (all 12) or \
                     `scripts/fetch_dimacs_roads.sh {}` to download just this one.",
                    name,
                    path.display(),
                    e,
                    name.trim_start_matches("USA-road-t.")
                )
            })
        }
    }
}

fn resolve_dimacs_path(name: &str) -> PathBuf {
    let dir = dimacs_dir();
    for ext in ["gr", "gr.gz"] {
        let p = dir.join(format!("{}.{}", name, ext));
        if p.exists() {
            return p;
        }
        let p2 = dir.join(format!("USA-road-t.{}.{}", name, ext));
        if p2.exists() {
            return p2;
        }
    }
    dir.join(format!("{}.gr", name))
}
