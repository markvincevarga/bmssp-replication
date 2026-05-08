use duan::algorithms::{
    BmsspBase, BmsspO1, BmsspO2, BmsspO3, BmsspO4, BmsspO5, BmsspO6, Dijkstra, PathFinder,
    PathResult,
};
use duan::graph_source::GraphSource;
use duan::graph_store::load_or_generate;
use petgraph::graph::{Graph, NodeIndex};

fn assert_matches(name: &str, source: &GraphSource, n: usize, dij: &PathResult, got: &PathResult) {
    let mut wrong: Vec<(usize, f64, f64)> = Vec::new();
    for (k, dv) in &dij.distances {
        match got.distances.get(k) {
            Some(bv) => {
                if (dv - bv).abs() > 1e-6 {
                    wrong.push((k.index(), *dv, *bv));
                }
            }
            None => wrong.push((k.index(), *dv, f64::INFINITY)),
        }
    }
    assert!(
        wrong.is_empty(),
        "{} on {:?} n={}: disagrees with dijkstra on {} of {} reachable vertices. \
         First 5: {:?}",
        name,
        source,
        n,
        wrong.len(),
        dij.distances.len(),
        &wrong[..wrong.len().min(5)],
    );
}

fn check_all(source: GraphSource, n: usize) {
    let graph: Graph<(), f64> = load_or_generate(n, &source);
    let src = NodeIndex::new(0);
    let dij = Dijkstra::new().shortest_paths(&graph, src);
    assert_matches("bmssp_base", &source, n, &dij, &BmsspBase::new().shortest_paths(&graph, src));
    assert_matches("bmssp_o_1",  &source, n, &dij, &BmsspO1::new().shortest_paths(&graph, src));
    assert_matches("bmssp_o_2",  &source, n, &dij, &BmsspO2::new().shortest_paths(&graph, src));
    assert_matches("bmssp_o_3",  &source, n, &dij, &BmsspO3::new().shortest_paths(&graph, src));
    assert_matches("bmssp_o_4",  &source, n, &dij, &BmsspO4::new().shortest_paths(&graph, src));
    assert_matches("bmssp_o_5",  &source, n, &dij, &BmsspO5::new().shortest_paths(&graph, src));
    assert_matches("bmssp_o_6",  &source, n, &dij, &BmsspO6::new().shortest_paths(&graph, src));
}

#[test]
fn sgrided_n256_all_variants_match_dijkstra() {
    // 16x16 Euclidean grid — Castro Table A.3 smallest. Lots of equal-weight ties
    // (axis=1.0, diagonal=√2) which expose ordering bugs in the BMSSP recursion.
    check_all(GraphSource::CastroSGridED, 256);
}

#[test]
fn rgrided_n256_all_variants_match_dijkstra() {
    check_all(GraphSource::CastroRGridED, 256);
}

#[test]
fn sgridr_n256_all_variants_match_dijkstra() {
    check_all(GraphSource::CastroSGridR, 256);
}

#[test]
fn sgrided_n1024_all_variants_match_dijkstra() {
    check_all(GraphSource::CastroSGridED, 1024);
}

#[test]
fn d3_n65536_all_variants_match_dijkstra() {
    check_all(GraphSource::CastroD3, 65_536);
}
