use petgraph::graph::{Graph, NodeIndex};

pub fn create_test_graph() -> Graph<(), f64> {
    let mut graph = Graph::new();

    let n0 = graph.add_node(());
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());
    let n3 = graph.add_node(());
    let n4 = graph.add_node(());

    graph.add_edge(n0, n1, 1.0);
    graph.add_edge(n0, n2, 4.0);
    graph.add_edge(n1, n3, 2.0);
    graph.add_edge(n1, n4, 2.0);
    graph.add_edge(n2, n4, 3.0);
    graph.add_edge(n3, n4, 1.0);

    graph
}

pub fn verify_path(graph: &Graph<(), f64>, path: &[NodeIndex]) -> Option<f64> {
    if path.is_empty() {
        return None;
    }

    if path.len() == 1 {
        return Some(0.0);
    }

    let mut total_cost = 0.0;

    for window in path.windows(2) {
        let from = window[0];
        let to = window[1];

        let edge = graph.find_edge(from, to)?;
        let weight = graph.edge_weight(edge)?;

        total_cost += weight;
    }

    Some(total_cost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_graph() {
        let graph = create_test_graph();
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 6);
    }

    #[test]
    fn test_verify_path_valid() {
        let graph = create_test_graph();
        let path = vec![NodeIndex::new(0), NodeIndex::new(1), NodeIndex::new(3)];

        let cost = verify_path(&graph, &path);
        assert_eq!(cost, Some(3.0)); // 1.0 + 2.0
    }

    #[test]
    fn test_verify_path_empty() {
        let graph = create_test_graph();
        let path = vec![];

        assert_eq!(verify_path(&graph, &path), None);
    }

    #[test]
    fn test_verify_path_single_node() {
        let graph = create_test_graph();
        let path = vec![NodeIndex::new(0)];

        assert_eq!(verify_path(&graph, &path), Some(0.0));
    }

    #[test]
    fn test_verify_path_invalid() {
        let graph = create_test_graph();
        let path = vec![NodeIndex::new(0), NodeIndex::new(3)];

        assert_eq!(verify_path(&graph, &path), None);
    }
}
