use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use std::collections::HashSet;
use rustc_hash::FxHashMap as HashMap;

pub struct TransformResult {
    pub graph: Graph<(), f64>,
    pub node_mapping: HashMap<NodeIndex, NodeIndex>,
    pub source: NodeIndex,
}

pub fn constant_degree_transform(
    graph: &Graph<(), f64>,
    source: NodeIndex,
    _max_degree: usize,
) -> TransformResult {
    let mut new_graph = Graph::new();
    let mut node_mapping: HashMap<NodeIndex, NodeIndex> = HashMap::default();
    let mut proxy_map: HashMap<(usize, usize), NodeIndex> = HashMap::default();

    for old_node in graph.node_indices() {
        let v = old_node.index();

        let mut neighbors: Vec<usize> = Vec::new();
        let mut seen: HashSet<usize> = HashSet::new();

        for e in graph.edges_directed(old_node, Direction::Outgoing) {
            let w = e.target().index();
            if seen.insert(w) {
                neighbors.push(w);
            }
        }
        for e in graph.edges_directed(old_node, Direction::Incoming) {
            let w = e.source().index();
            if seen.insert(w) {
                neighbors.push(w);
            }
        }

        if neighbors.is_empty() {
            let proxy = new_graph.add_node(());
            node_mapping.insert(proxy, old_node);
            continue;
        }

        let mut cycle_nodes: Vec<NodeIndex> = Vec::with_capacity(neighbors.len());
        for &w in &neighbors {
            let proxy = new_graph.add_node(());
            node_mapping.insert(proxy, old_node);
            proxy_map.insert((v, w), proxy);
            cycle_nodes.push(proxy);
        }

        for i in 0..cycle_nodes.len() {
            let next = (i + 1) % cycle_nodes.len();
            new_graph.add_edge(cycle_nodes[i], cycle_nodes[next], 0.0);
        }
    }

    for edge in graph.edge_references() {
        let u = edge.source().index();
        let v = edge.target().index();
        let w = *edge.weight();

        let from = proxy_map[&(u, v)];
        let to = proxy_map[&(v, u)];
        new_graph.add_edge(from, to, w);
    }

    let source_v = source.index();
    let new_source = graph
        .edges_directed(source, Direction::Outgoing)
        .next()
        .map(|e| proxy_map[&(source_v, e.target().index())])
        .or_else(|| {
            graph
                .edges_directed(source, Direction::Incoming)
                .next()
                .map(|e| proxy_map[&(source_v, e.source().index())])
        })
        .unwrap_or_else(|| {
            *node_mapping
                .iter()
                .find(|(_, &orig)| orig == source)
                .unwrap()
                .0
        });

    TransformResult {
        graph: new_graph,
        node_mapping,
        source: new_source,
    }
}
