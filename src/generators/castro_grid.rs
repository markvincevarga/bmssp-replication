use petgraph::graph::Graph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub const CASTRO_GRID_MAX_WEIGHT: u64 = 100_000;
pub const CASTRO_GRID_DEFAULT_SEED: u64 = 1971;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastroGridShape {
    Square,
    Rectangular,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastroGridWeights {
    Euclidean,
    RandomInt,
}

#[derive(Debug, Clone)]
pub struct CastroGridConfig {
    pub rows: usize,
    pub cols: usize,
    pub weights: CastroGridWeights,
    pub max_weight: u64,
}

impl CastroGridConfig {
    pub fn from_node_count(n: usize, shape: CastroGridShape, weights: CastroGridWeights) -> Self {
        let (rows, cols) = grid_dimensions(n, shape);
        Self {
            rows,
            cols,
            weights,
            max_weight: CASTRO_GRID_MAX_WEIGHT,
        }
    }
}

pub fn grid_dimensions(n: usize, shape: CastroGridShape) -> (usize, usize) {
    match shape {
        CastroGridShape::Square => {
            let s = (n as f64).sqrt().round() as usize;
            assert!(
                s * s == n,
                "Square grid requires perfect-square node_count, got {}",
                n
            );
            (s, s)
        }
        CastroGridShape::Rectangular => {
            let s = (n as f64).sqrt().round() as usize;
            assert!(
                s * s == n,
                "Rectangular grid requires perfect-square node_count, got {}",
                n
            );
            assert!(
                s % 2 == 0,
                "Rectangular grid requires even sqrt(node_count) (rows = N/2, cols = 2N), got sqrt = {}",
                s
            );
            (s / 2, s * 2)
        }
    }
}

pub fn generate_seeded(config: &CastroGridConfig, seed: u64) -> Graph<(), f64> {
    let n_row = config.rows as i64;
    let n_col = config.cols as i64;
    assert!(n_row > 0 && n_col > 0);
    let n = (n_row * n_col) as usize;
    let c = match config.weights {
        CastroGridWeights::Euclidean => 0u64,
        CastroGridWeights::RandomInt => config.max_weight,
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut graph: Graph<(), f64> = Graph::with_capacity(n, n * 8);
    let nodes: Vec<_> = (0..n).map(|_| graph.add_node(())).collect();

    let wallcol: i64 = n_col / 2;
    let wall_weight: f64 = ((n_col + n_row + 1) as u64 * (c + 1)) as f64;

    let add = |graph: &mut Graph<(), f64>, i: i64, j: i64, ii: i64, jj: i64, mut w: f64| {
        if i.max(ii) >= n_row || j.max(jj) >= n_col || i.min(ii).min(j).min(jj) < 0 {
            return;
        }
        let id1 = (i * n_col + j) as usize;
        let id2 = (ii * n_col + jj) as usize;
        if jj == wallcol && ii >= 5 {
            w = wall_weight;
        }
        graph.add_edge(nodes[id1], nodes[id2], w);
        graph.add_edge(nodes[id2], nodes[id1], w);
    };

    for i in 0..n_row {
        for j in 0..n_col {
            match config.weights {
                CastroGridWeights::RandomInt => {
                    let w = rng.gen_range(1..=config.max_weight) as f64;
                    add(&mut graph, i, j, i, j + 1, w);
                    let w = rng.gen_range(1..=config.max_weight) as f64;
                    add(&mut graph, i, j, i + 1, j, w);
                    let w = rng.gen_range(1..=config.max_weight) as f64;
                    add(&mut graph, i, j, i + 1, j + 1, w);
                    let w = rng.gen_range(1..=config.max_weight) as f64;
                    add(&mut graph, i, j, i + 1, j - 1, w);
                }
                CastroGridWeights::Euclidean => {
                    add(&mut graph, i, j, i, j + 1, 1.0);
                    add(&mut graph, i, j, i + 1, j, 1.0);
                    add(&mut graph, i, j, i + 1, j + 1, std::f64::consts::SQRT_2);
                    add(&mut graph, i, j, i + 1, j - 1, std::f64::consts::SQRT_2);
                }
            }
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::graph::NodeIndex;
    use petgraph::visit::EdgeRef;
    use std::collections::VecDeque;

    fn bfs_reach_count(g: &Graph<(), f64>, src: usize) -> usize {
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

    #[test]
    fn square_dimensions() {
        let cfg = CastroGridConfig::from_node_count(
            256,
            CastroGridShape::Square,
            CastroGridWeights::Euclidean,
        );
        assert_eq!((cfg.rows, cfg.cols), (16, 16));
    }

    #[test]
    fn rectangular_dimensions_4to1() {
        let cfg = CastroGridConfig::from_node_count(
            256,
            CastroGridShape::Rectangular,
            CastroGridWeights::Euclidean,
        );
        assert_eq!((cfg.rows, cfg.cols), (8, 32));
    }

    #[test]
    fn euclidean_weights_use_sqrt2_for_diagonal() {
        let cfg = CastroGridConfig {
            rows: 4,
            cols: 4,
            weights: CastroGridWeights::Euclidean,
            max_weight: 0,
        };
        let g = generate_seeded(&cfg, 1971);
        let mut saw_orth = false;
        let mut saw_diag = false;
        for e in g.raw_edges() {
            if (e.weight - 1.0).abs() < 1e-9 {
                saw_orth = true;
            } else if (e.weight - std::f64::consts::SQRT_2).abs() < 1e-9 {
                saw_diag = true;
            }
        }
        assert!(saw_orth);
        assert!(saw_diag);
    }

    #[test]
    fn random_weights_within_range() {
        let cfg = CastroGridConfig {
            rows: 4,
            cols: 4,
            weights: CastroGridWeights::RandomInt,
            max_weight: 100,
        };
        let g = generate_seeded(&cfg, 1971);
        let wall_weight = ((4 + 4 + 1) as u64 * (100 + 1)) as f64;
        for e in g.raw_edges() {
            let w = e.weight;
            let ok_normal = w >= 1.0 && w <= 100.0;
            let ok_wall = (w - wall_weight).abs() < 1e-9;
            assert!(
                ok_normal || ok_wall,
                "weight {} out of [1, 100] and not wall {}",
                w,
                wall_weight
            );
        }
    }

    #[test]
    fn fully_reachable_from_zero() {
        let cfg = CastroGridConfig::from_node_count(
            1024,
            CastroGridShape::Square,
            CastroGridWeights::Euclidean,
        );
        let g = generate_seeded(&cfg, 1971);
        assert_eq!(bfs_reach_count(&g, 0), 1024);
    }

    #[test]
    fn deterministic_random() {
        let cfg = CastroGridConfig {
            rows: 8,
            cols: 8,
            weights: CastroGridWeights::RandomInt,
            max_weight: 100,
        };
        let g1 = generate_seeded(&cfg, 1971);
        let g2 = generate_seeded(&cfg, 1971);
        let e1: Vec<_> = g1
            .raw_edges()
            .iter()
            .map(|e| (e.source().index(), e.target().index(), e.weight))
            .collect();
        let e2: Vec<_> = g2
            .raw_edges()
            .iter()
            .map(|e| (e.source().index(), e.target().index(), e.weight))
            .collect();
        assert_eq!(e1, e2);
    }
}
