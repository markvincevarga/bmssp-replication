use super::traits::{PathFinder, PathResult};
use crate::graph_repr::CsrAosGraph;
use petgraph::graph::{Graph, NodeIndex};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use rustc_hash::FxHashMap as HashMap;

#[derive(Copy, Clone)]
struct State {
    cost: f64,
    node: u32,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

pub struct DijkstraOpt;

impl DijkstraOpt {
    pub fn new() -> Self {
        DijkstraOpt
    }
}

impl Default for DijkstraOpt {
    fn default() -> Self {
        Self::new()
    }
}

impl PathFinder for DijkstraOpt {
    #[hotpath::measure]
    fn shortest_paths(&self, graph: &Graph<(), f64>, source: NodeIndex) -> PathResult {
        let n = graph.node_count();
        let csr = CsrAosGraph::from_petgraph(graph);

        let mut dist = vec![f64::INFINITY; n];
        let mut pred = vec![u32::MAX; n];
        let mut heap = BinaryHeap::with_capacity(n);

        let src = source.index();
        dist[src] = 0.0;
        heap.push(State {
            cost: 0.0,
            node: src as u32,
        });

        while let Some(State { cost, node }) = heap.pop() {
            let u = node as usize;
            unsafe {
                if cost > *dist.get_unchecked(u) {
                    continue;
                }

                for edge in csr.edges(u) {
                    let v = edge.target as usize;
                    let next_cost = cost + edge.weight;

                    if next_cost < *dist.get_unchecked(v) {
                        *dist.get_unchecked_mut(v) = next_cost;
                        *pred.get_unchecked_mut(v) = node;
                        heap.push(State {
                            cost: next_cost,
                            node: edge.target,
                        });
                    }
                }
            }
        }

        let mut distances = HashMap::default();
        let mut predecessors = HashMap::default();

        for i in 0..n {
            if dist[i] < f64::INFINITY {
                distances.insert(NodeIndex::new(i), dist[i]);
                if i != src && pred[i] != u32::MAX {
                    predecessors.insert(NodeIndex::new(i), NodeIndex::new(pred[i] as usize));
                }
            }
        }

        PathResult {
            distances,
            predecessors,
        }
    }

    #[hotpath::measure]
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
    fn test_shortest_path_distances() {
        let graph = create_simple_graph();
        let dijkstra = DijkstraOpt::new();
        let result = dijkstra.shortest_paths(&graph, NodeIndex::new(0));

        assert_eq!(result.distances.get(&NodeIndex::new(0)), Some(&0.0));
        assert_eq!(result.distances.get(&NodeIndex::new(1)), Some(&1.0));
        assert_eq!(result.distances.get(&NodeIndex::new(2)), Some(&3.0));
        assert_eq!(result.distances.get(&NodeIndex::new(3)), Some(&4.0));
    }

    #[test]
    fn test_path_reconstruction() {
        let graph = create_simple_graph();
        let dijkstra = DijkstraOpt::new();
        let result = dijkstra.shortest_paths(&graph, NodeIndex::new(0));

        let path = dijkstra
            .reconstruct_path(&result, NodeIndex::new(0), NodeIndex::new(3))
            .expect("Path should exist");

        assert_eq!(
            path,
            vec![
                NodeIndex::new(0),
                NodeIndex::new(1),
                NodeIndex::new(2),
                NodeIndex::new(3)
            ]
        );
    }

    #[test]
    fn test_unreachable_node() {
        let mut graph = Graph::new();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let _n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1.0);

        let dijkstra = DijkstraOpt::new();
        let result = dijkstra.shortest_paths(&graph, n0);

        assert!(result.distances.get(&NodeIndex::new(2)).is_none());
        assert!(dijkstra.reconstruct_path(&result, n0, NodeIndex::new(2)).is_none());
    }

    #[test]
    fn test_source_to_source() {
        let graph = create_simple_graph();
        let dijkstra = DijkstraOpt::new();
        let result = dijkstra.shortest_paths(&graph, NodeIndex::new(0));

        let path = dijkstra
            .reconstruct_path(&result, NodeIndex::new(0), NodeIndex::new(0))
            .expect("Path to self should exist");

        assert_eq!(path, vec![NodeIndex::new(0)]);
    }
}
