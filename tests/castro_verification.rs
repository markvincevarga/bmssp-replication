use duan::generators::castro_d3::{generate_seeded as gen_d3, CastroD3Config, D3_DEFAULT_SEED};
use duan::generators::castro_grid::{
    generate_seeded as gen_grid, grid_dimensions, CastroGridConfig, CastroGridShape,
    CastroGridWeights, CASTRO_GRID_DEFAULT_SEED,
};
use duan::generators::castro_h3::{generate_seeded as gen_h3, CastroH3Config, H3_DEFAULT_SEED};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::collections::VecDeque;

fn forward_bfs_count(g: &Graph<(), f64>, src: usize) -> usize {
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

fn reverse_bfs_count(g: &Graph<(), f64>, src: usize) -> usize {
    let n = g.node_count();
    let mut rev: Vec<Vec<usize>> = vec![Vec::new(); n];
    for u in 0..n {
        for e in g.edges(NodeIndex::new(u)) {
            rev[e.target().index()].push(u);
        }
    }
    let mut seen = vec![false; n];
    let mut q = VecDeque::new();
    q.push_back(src);
    seen[src] = true;
    let mut count = 1;
    while let Some(u) = q.pop_front() {
        for &v in &rev[u] {
            if !seen[v] {
                seen[v] = true;
                count += 1;
                q.push_back(v);
            }
        }
    }
    count
}

fn max_out_degree(g: &Graph<(), f64>) -> usize {
    (0..g.node_count())
        .map(|i| g.edges(NodeIndex::new(i)).count())
        .max()
        .unwrap_or(0)
}

#[test]
fn castro_d3_full_table() {
    for &n in &[128usize, 256, 512, 1024, 2048, 4096, 8192, 16384] {
        let config = CastroD3Config {
            node_count: n,
            ..Default::default()
        };
        let g = gen_d3(&config, D3_DEFAULT_SEED);
        assert_eq!(g.node_count(), n, "n mismatch for D3 size {}", n);
        assert_eq!(g.edge_count(), 3 * n, "m mismatch for D3 size {}", n);
        assert!(
            max_out_degree(&g) <= 4,
            "max out-degree exceeds 4 for D3 size {}",
            n
        );
        assert_eq!(
            forward_bfs_count(&g, 0),
            n,
            "vertex 0 must reach all for D3 size {}",
            n
        );
    }
}

#[test]
fn castro_h3_full_table() {
    for &n in &[128usize, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536] {
        let config = CastroH3Config {
            node_count: n,
            ..Default::default()
        };
        let g = gen_h3(&config, H3_DEFAULT_SEED);
        assert_eq!(g.node_count(), n, "n mismatch for H3 size {}", n);
        assert_eq!(g.edge_count(), 3 * n, "m mismatch for H3 size {}", n);
        assert_eq!(
            forward_bfs_count(&g, 0),
            n,
            "vertex 0 must reach all (forward) for H3 size {}",
            n
        );
        assert_eq!(
            reverse_bfs_count(&g, 0),
            n,
            "vertex 0 must reach all (reverse) for H3 size {}",
            n
        );
    }
}

#[test]
fn castro_grid_dimensions_match_table() {
    let cases = vec![
        (256usize, CastroGridShape::Square, (16usize, 16usize)),
        (256, CastroGridShape::Rectangular, (8, 32)),
        (1024, CastroGridShape::Square, (32, 32)),
        (1024, CastroGridShape::Rectangular, (16, 64)),
        (4096, CastroGridShape::Square, (64, 64)),
        (4096, CastroGridShape::Rectangular, (32, 128)),
        (16_384, CastroGridShape::Square, (128, 128)),
        (16_384, CastroGridShape::Rectangular, (64, 256)),
        (65_536, CastroGridShape::Square, (256, 256)),
        (65_536, CastroGridShape::Rectangular, (128, 512)),
        (262_144, CastroGridShape::Square, (512, 512)),
        (262_144, CastroGridShape::Rectangular, (256, 1024)),
        (1_048_576, CastroGridShape::Square, (1024, 1024)),
        (1_048_576, CastroGridShape::Rectangular, (512, 2048)),
        (4_194_304, CastroGridShape::Square, (2048, 2048)),
        (4_194_304, CastroGridShape::Rectangular, (1024, 4096)),
        (16_777_216, CastroGridShape::Square, (4096, 4096)),
        (16_777_216, CastroGridShape::Rectangular, (2048, 8192)),
    ];
    for (n, shape, expected) in cases {
        let got = grid_dimensions(n, shape);
        assert_eq!(got, expected, "dimensions for n={}, {:?}", n, shape);
    }
}

#[test]
fn castro_grid_edge_count_8connectivity() {
    // For an R×C 8-connected grid (no wallcol effect on count), interior cells
    // contribute 4 unique undirected edges each via the (i,j+1), (i+1,j),
    // (i+1,j+1), (i+1,j-1) pattern; boundaries omit out-of-bounds neighbors.
    // Exact count: count distinct undirected edges = R*(C-1) [horizontal]
    //                                              + (R-1)*C [vertical]
    //                                              + (R-1)*(C-1) [↘ diagonals]
    //                                              + (R-1)*(C-1) [↙ diagonals]
    // Each undirected edge becomes 2 directed edges in the petgraph output.
    let r = 16usize;
    let c = 16usize;
    let cfg = CastroGridConfig {
        rows: r,
        cols: c,
        weights: CastroGridWeights::Euclidean,
        max_weight: 0,
    };
    let g = gen_grid(&cfg, CASTRO_GRID_DEFAULT_SEED);
    assert_eq!(g.node_count(), r * c);
    let undirected_edges = r * (c - 1) + (r - 1) * c + 2 * (r - 1) * (c - 1);
    assert_eq!(g.edge_count(), 2 * undirected_edges);
}

#[test]
fn castro_grid_wall_weight_present_in_random() {
    let cfg = CastroGridConfig {
        rows: 16,
        cols: 16,
        weights: CastroGridWeights::RandomInt,
        max_weight: 100_000,
    };
    let g = gen_grid(&cfg, CASTRO_GRID_DEFAULT_SEED);
    let wall = ((cfg.cols + cfg.rows + 1) as u64 * (cfg.max_weight + 1)) as f64;
    let saw_wall = g.raw_edges().iter().any(|e| (e.weight - wall).abs() < 1e-9);
    assert!(saw_wall, "expected at least one wall-weight edge");
}

#[test]
fn castro_grid_wall_weight_present_in_euclidean() {
    let cfg = CastroGridConfig {
        rows: 16,
        cols: 16,
        weights: CastroGridWeights::Euclidean,
        max_weight: 0,
    };
    let g = gen_grid(&cfg, CASTRO_GRID_DEFAULT_SEED);
    let wall = ((cfg.cols + cfg.rows + 1) as u64 * 1) as f64;
    let saw_wall = g.raw_edges().iter().any(|e| (e.weight - wall).abs() < 1e-9);
    assert!(saw_wall, "expected at least one wall-weight edge");
}

#[test]
fn castro_grid_random_weights_in_paper_range() {
    let cfg = CastroGridConfig {
        rows: 32,
        cols: 32,
        weights: CastroGridWeights::RandomInt,
        max_weight: 100_000,
    };
    let g = gen_grid(&cfg, CASTRO_GRID_DEFAULT_SEED);
    let wall = ((cfg.cols + cfg.rows + 1) as u64 * (cfg.max_weight + 1)) as f64;
    for e in g.raw_edges() {
        let w = e.weight;
        let in_range = w >= 1.0 && w <= 100_000.0;
        let is_wall = (w - wall).abs() < 1e-9;
        assert!(
            in_range || is_wall,
            "weight {} out of paper range [1, 100000] and not wall {}",
            w,
            wall
        );
    }
}

#[test]
fn castro_grid_euclidean_weights_only_1_or_sqrt2_or_wall() {
    let cfg = CastroGridConfig {
        rows: 16,
        cols: 16,
        weights: CastroGridWeights::Euclidean,
        max_weight: 0,
    };
    let g = gen_grid(&cfg, CASTRO_GRID_DEFAULT_SEED);
    let wall = ((cfg.cols + cfg.rows + 1) as u64 * 1) as f64;
    for e in g.raw_edges() {
        let w = e.weight;
        let ok = (w - 1.0).abs() < 1e-9
            || (w - std::f64::consts::SQRT_2).abs() < 1e-9
            || (w - wall).abs() < 1e-9;
        assert!(ok, "unexpected Euclidean grid weight: {}", w);
    }
}

#[test]
fn castro_d3_h3_seed_constants_match_paper_scripts() {
    assert_eq!(D3_DEFAULT_SEED, 1, "Castro's D3 bash uses seed=1");
    assert_eq!(H3_DEFAULT_SEED, 1971, "Castro's H3 bash uses seed=1971");
    assert_eq!(
        CASTRO_GRID_DEFAULT_SEED, 1971,
        "Castro's grid bash uses seed=1971"
    );
}

#[test]
fn castro_d3_default_max_weight_matches_paper() {
    let cfg = CastroD3Config::default();
    assert_eq!(
        cfg.max_weight, 100_000,
        "Castro's D3 bash uses max_weight=100000"
    );
}

#[test]
fn castro_h3_default_max_weight_and_density_match_paper() {
    let cfg = CastroH3Config::default();
    assert_eq!(
        cfg.max_weight, 100_000,
        "Castro's H3 bash uses C=max_weight=100000"
    );
    assert_eq!(cfg.density, 3, "paper reports H3 density");
}
