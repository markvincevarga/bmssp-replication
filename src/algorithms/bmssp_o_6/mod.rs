mod base_case;
mod bmssp_core;
mod d_struct;
mod dist_value;
mod epoch_map;
mod find_pivots;
mod state;
mod transform;

use super::traits::{PathFinder, PathResult, PreprocessResult};
use crate::graph_repr::CsrAosGraph;
use petgraph::graph::{Graph, NodeIndex};
use rustc_hash::FxHashMap as HashMap;

pub struct BmsspO6 {
    use_transform: bool,
    max_degree: usize,
}

impl BmsspO6 {
    pub fn new() -> Self {
        Self {
            use_transform: false,
            max_degree: 4,
        }
    }

    pub fn with_transformation(max_degree: usize) -> Self {
        Self {
            use_transform: true,
            max_degree,
        }
    }
}

impl Default for BmsspO6 {
    fn default() -> Self {
        Self::new()
    }
}

impl PathFinder for BmsspO6 {
    fn preprocess(&self, graph: &Graph<(), f64>, source: NodeIndex) -> PreprocessResult {
        if !self.use_transform {
            return PreprocessResult {
                graph: graph.clone(),
                source,
                node_mapping: None,
            };
        }
        let result = transform::constant_degree_transform(graph, source, self.max_degree);
        PreprocessResult {
            graph: result.graph,
            source: result.source,
            node_mapping: Some(result.node_mapping),
        }
    }

    fn map_result(
        &self,
        result: PathResult,
        node_mapping: &Option<HashMap<NodeIndex, NodeIndex>>,
    ) -> PathResult {
        let mapping = match node_mapping {
            Some(m) => m,
            None => return result,
        };

        let mut distances: HashMap<NodeIndex, f64> = HashMap::default();
        let mut predecessors: HashMap<NodeIndex, NodeIndex> = HashMap::default();

        for (&new_node, &dist) in &result.distances {
            if let Some(&orig_node) = mapping.get(&new_node) {
                let entry = distances.entry(orig_node).or_insert(f64::INFINITY);
                if dist < *entry {
                    *entry = dist;
                    if let Some(&pred_new) = result.predecessors.get(&new_node) {
                        if let Some(&pred_orig) = mapping.get(&pred_new) {
                            if pred_orig != orig_node {
                                predecessors.insert(orig_node, pred_orig);
                            }
                        }
                    }
                }
            }
        }

        PathResult {
            distances,
            predecessors,
        }
    }

    fn shortest_paths(&self, graph: &Graph<(), f64>, source: NodeIndex) -> PathResult {
        let n = graph.node_count();
        if n == 0 {
            return PathResult {
                distances: HashMap::default(),
                predecessors: HashMap::default(),
            };
        }

        let csr = CsrAosGraph::from_petgraph(graph);
        let source_u32 = source.index() as u32;
        let mut state = state::BmsspState::new(n, source_u32);

        let max_level = if n <= 1 {
            0
        } else {
            ((n as f64).log2() / state.t as f64).ceil() as usize
        };
        let mut queues: Vec<d_struct::DStruct> = Vec::new();
        let mut epoch_map = epoch_map::EpochMap::new(n);
        let bound = dist_value::safe_infinity();

        let sources = vec![source_u32];
        bmssp_core::bmssp_iterative(
            &csr,
            &mut state,
            max_level,
            bound,
            &sources,
            &mut queues,
            &mut epoch_map,
        );

        let inf = dist_value::safe_infinity();
        let mut distances: HashMap<NodeIndex, f64> = HashMap::with_capacity_and_hasher(n, Default::default());
        let mut predecessors: HashMap<NodeIndex, NodeIndex> = HashMap::with_capacity_and_hasher(n, Default::default());

        let src_idx = source.index();
        for i in 0..n {
            let d = state.d_hat[i];
            if d >= inf {
                continue;
            }
            distances.insert(NodeIndex::new(i), d);
            if i != src_idx && state.pred[i] != u32::MAX {
                predecessors.insert(NodeIndex::new(i), NodeIndex::new(state.pred[i] as usize));
            }
        }

        PathResult {
            distances,
            predecessors,
        }
    }

    fn reconstruct_path(
        &self,
        result: &PathResult,
        source: NodeIndex,
        target: NodeIndex,
    ) -> Option<Vec<NodeIndex>> {
        if !result.distances.contains_key(&target) {
            return None;
        }

        let mut path = Vec::new();
        let mut current = target;

        path.push(current);

        while current != source {
            match result.predecessors.get(&current) {
                Some(&pred) => {
                    path.push(pred);
                    current = pred;
                }
                None => return None,
            }
        }

        path.reverse();
        Some(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::dijkstra::Dijkstra;
    use petgraph::graph::Graph;

    fn assert_distances_match(bmssp: &PathResult, dijkstra: &PathResult) {
        for (&node, &dij_dist) in &dijkstra.distances {
            let bmssp_dist = bmssp
                .distances
                .get(&node)
                .copied()
                .unwrap_or(f64::INFINITY);
            assert!(
                (bmssp_dist - dij_dist).abs() < 1e-9,
                "Mismatch at node {:?}: bmssp={}, dijkstra={}",
                node,
                bmssp_dist,
                dij_dist
            );
        }
        for (&node, &bmssp_dist) in &bmssp.distances {
            let dij_dist = dijkstra
                .distances
                .get(&node)
                .copied()
                .unwrap_or(f64::INFINITY);
            assert!(
                (bmssp_dist - dij_dist).abs() < 1e-9,
                "Extra node {:?}: bmssp={}, dijkstra={}",
                node,
                bmssp_dist,
                dij_dist
            );
        }
    }

    fn create_simple_graph() -> Graph<(), f64> {
        let mut graph = Graph::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());

        graph.add_edge(n0, n1, 1.0);
        graph.add_edge(n0, n2, 4.0);
        graph.add_edge(n1, n2, 2.0);
        graph.add_edge(n1, n3, 5.0);
        graph.add_edge(n2, n3, 1.0);

        graph
    }

    #[test]
    fn test_basic_distances() {
        let graph = create_simple_graph();
        let bmssp = BmsspO6::new();
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        let bmssp_result = bmssp.shortest_paths(&graph, source);
        let dij_result = dijkstra.shortest_paths(&graph, source);

        assert_distances_match(&bmssp_result, &dij_result);
    }

    #[test]
    fn test_single_node() {
        let mut graph = Graph::new();
        graph.add_node(());

        let bmssp = BmsspO6::new();
        let result = bmssp.shortest_paths(&graph, NodeIndex::new(0));

        assert_eq!(result.distances.get(&NodeIndex::new(0)), Some(&0.0));
    }

    #[test]
    fn test_unreachable_node() {
        let mut graph = Graph::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let _n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1.0);

        let bmssp = BmsspO6::new();
        let result = bmssp.shortest_paths(&graph, n0);

        assert!(result.distances.contains_key(&n0));
        assert!(result.distances.contains_key(&n1));
        assert!(!result.distances.contains_key(&NodeIndex::new(2)));
    }

    #[test]
    fn test_path_reconstruction() {
        let graph = create_simple_graph();
        let bmssp = BmsspO6::new();
        let source = NodeIndex::new(0);
        let result = bmssp.shortest_paths(&graph, source);

        let path = bmssp
            .reconstruct_path(&result, source, NodeIndex::new(3))
            .expect("Path should exist");

        assert_eq!(path[0], source);
        assert_eq!(*path.last().unwrap(), NodeIndex::new(3));

        let mut cost = 0.0;
        for w in path.windows(2) {
            let edge = graph.find_edge(w[0], w[1]).expect("Edge should exist");
            cost += graph.edge_weight(edge).unwrap();
        }
        assert!((cost - 4.0).abs() < 1e-9);
    }

    fn make_chain(n: usize) -> Graph<(), f64> {
        let mut graph = Graph::new();
        let nodes: Vec<NodeIndex> = (0..n).map(|_| graph.add_node(())).collect();
        for i in 0..n - 1 {
            graph.add_edge(nodes[i], nodes[i + 1], 1.0);
        }
        graph
    }

    #[test]
    fn test_chain_5() {
        let graph = make_chain(5);
        let bmssp = BmsspO6::new();
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        assert_distances_match(
            &bmssp.shortest_paths(&graph, source),
            &dijkstra.shortest_paths(&graph, source),
        );
    }

    #[test]
    fn test_chain_10() {
        let graph = make_chain(10);
        let bmssp = BmsspO6::new();
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        assert_distances_match(
            &bmssp.shortest_paths(&graph, source),
            &dijkstra.shortest_paths(&graph, source),
        );
    }

    #[test]
    fn test_chain_20() {
        let graph = make_chain(20);
        let bmssp = BmsspO6::new();
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        assert_distances_match(
            &bmssp.shortest_paths(&graph, source),
            &dijkstra.shortest_paths(&graph, source),
        );
    }

    fn make_diamond() -> Graph<(), f64> {
        let mut graph = Graph::new();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());

        graph.add_edge(a, b, 1.0);
        graph.add_edge(a, c, 2.0);
        graph.add_edge(b, d, 3.0);
        graph.add_edge(c, d, 1.0);

        graph
    }

    #[test]
    fn test_diamond() {
        let graph = make_diamond();
        let bmssp = BmsspO6::new();
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        assert_distances_match(
            &bmssp.shortest_paths(&graph, source),
            &dijkstra.shortest_paths(&graph, source),
        );
    }

    fn make_complete(n: usize) -> Graph<(), f64> {
        let mut graph = Graph::new();
        let nodes: Vec<NodeIndex> = (0..n).map(|_| graph.add_node(())).collect();
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    graph.add_edge(nodes[i], nodes[j], (i + j + 1) as f64);
                }
            }
        }
        graph
    }

    #[test]
    fn test_complete_4() {
        let graph = make_complete(4);
        let bmssp = BmsspO6::new();
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        assert_distances_match(
            &bmssp.shortest_paths(&graph, source),
            &dijkstra.shortest_paths(&graph, source),
        );
    }

    #[test]
    fn test_complete_10() {
        let graph = make_complete(10);
        let bmssp = BmsspO6::new();
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        assert_distances_match(
            &bmssp.shortest_paths(&graph, source),
            &dijkstra.shortest_paths(&graph, source),
        );
    }

    #[test]
    fn test_erdos_renyi_100() {
        use crate::generators::{ErdosRenyiConfig, ErdosRenyiGenerator, GraphGenerator};

        let config = ErdosRenyiConfig::new(100, 0.15, 0.1, 10.0);
        let generator = ErdosRenyiGenerator::new();
        let graph = generator.generate_weighted(&config);
        let source = NodeIndex::new(0);

        let bmssp = BmsspO6::new();
        let dijkstra = Dijkstra::new();

        assert_distances_match(
            &bmssp.shortest_paths(&graph, source),
            &dijkstra.shortest_paths(&graph, source),
        );
    }

    #[test]
    fn test_stress_complete() {
        let bmssp = BmsspO6::new();
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        for n in [5, 10, 20, 50] {
            let graph = make_complete(n);
            assert_distances_match(
                &bmssp.shortest_paths(&graph, source),
                &dijkstra.shortest_paths(&graph, source),
            );
        }
    }

    #[test]
    fn test_with_transformation() {
        let graph = make_complete(10);
        let bmssp = BmsspO6::with_transformation(2);
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        let preprocessed = bmssp.preprocess(&graph, source);
        let raw_result = bmssp.shortest_paths(&preprocessed.graph, preprocessed.source);
        let bmssp_result = bmssp.map_result(raw_result, &preprocessed.node_mapping);
        let dij_result = dijkstra.shortest_paths(&graph, source);

        assert_distances_match(&bmssp_result, &dij_result);
    }

    #[test]
    fn test_zero_weight_cycle() {
        let mut graph = Graph::new();
        let nodes: Vec<NodeIndex> = (0..6).map(|_| graph.add_node(())).collect();
        graph.add_edge(nodes[0], nodes[1], 0.0);
        graph.add_edge(nodes[1], nodes[2], 0.0);
        graph.add_edge(nodes[2], nodes[0], 0.0);
        graph.add_edge(nodes[1], nodes[3], 5.0);
        graph.add_edge(nodes[2], nodes[4], 3.0);
        graph.add_edge(nodes[4], nodes[5], 1.0);

        let bmssp = BmsspO6::new();
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        assert_distances_match(
            &bmssp.shortest_paths(&graph, source),
            &dijkstra.shortest_paths(&graph, source),
        );
    }

    #[test]
    fn test_transform_fixed_edge() {
        use crate::generators::{FixedEdgeCountConfig, FixedEdgeCountGenerator};

        let generator = FixedEdgeCountGenerator::new();
        let bmssp = BmsspO6::with_transformation(2);
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        for &n in &[50, 100, 200] {
            let config = FixedEdgeCountConfig::new(n, n * 3, 0.1, 10.0);
            let graph = generator.generate_seeded(&config, 42);

            let preprocessed = bmssp.preprocess(&graph, source);
            let raw_result = bmssp.shortest_paths(&preprocessed.graph, preprocessed.source);
            let bmssp_result = bmssp.map_result(raw_result, &preprocessed.node_mapping);
            let dij_result = dijkstra.shortest_paths(&graph, source);

            assert_distances_match(&bmssp_result, &dij_result);
        }
    }

    #[test]
    fn test_transform_chain() {
        let graph = make_chain(20);
        let bmssp = BmsspO6::with_transformation(2);
        let dijkstra = Dijkstra::new();
        let source = NodeIndex::new(0);

        let preprocessed = bmssp.preprocess(&graph, source);
        let raw_result = bmssp.shortest_paths(&preprocessed.graph, preprocessed.source);
        let bmssp_result = bmssp.map_result(raw_result, &preprocessed.node_mapping);
        let dij_result = dijkstra.shortest_paths(&graph, source);

        assert_distances_match(&bmssp_result, &dij_result);
    }

    #[test]
    fn test_erdos_renyi_1000() {
        use crate::generators::{ErdosRenyiConfig, ErdosRenyiGenerator, GraphGenerator};

        let config = ErdosRenyiConfig::new(1000, 0.05, 0.1, 10.0);
        let generator = ErdosRenyiGenerator::new();
        let graph = generator.generate_weighted(&config);
        let source = NodeIndex::new(0);

        let bmssp = BmsspO6::new();
        let dijkstra = Dijkstra::new();

        assert_distances_match(
            &bmssp.shortest_paths(&graph, source),
            &dijkstra.shortest_paths(&graph, source),
        );
    }
}
