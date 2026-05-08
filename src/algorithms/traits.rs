use petgraph::graph::{Graph, NodeIndex};
use rustc_hash::FxHashMap;

pub type HashMap<K, V> = FxHashMap<K, V>;

#[derive(Debug, Clone)]
pub struct PathResult {
    pub distances: HashMap<NodeIndex, f64>,
    pub predecessors: HashMap<NodeIndex, NodeIndex>,
}

pub struct PreprocessResult {
    pub graph: Graph<(), f64>,
    pub source: NodeIndex,
    pub node_mapping: Option<HashMap<NodeIndex, NodeIndex>>,
}

pub trait PathFinder {
    fn preprocess(&self, graph: &Graph<(), f64>, source: NodeIndex) -> PreprocessResult {
        PreprocessResult {
            graph: graph.clone(),
            source,
            node_mapping: None,
        }
    }

    fn map_result(
        &self,
        result: PathResult,
        _node_mapping: &Option<HashMap<NodeIndex, NodeIndex>>,
    ) -> PathResult {
        result
    }

    fn shortest_paths(
        &self,
        graph: &Graph<(), f64>,
        source: NodeIndex,
    ) -> PathResult;

    fn reconstruct_path(
        &self,
        result: &PathResult,
        source: NodeIndex,
        target: NodeIndex,
    ) -> Option<Vec<NodeIndex>>;
}
