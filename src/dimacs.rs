use petgraph::graph::{Graph, NodeIndex};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

pub fn read_gr_file<P: AsRef<Path>>(path: P) -> io::Result<Graph<(), f64>> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if ext == "gz" {
        read_gr_gz(file)
    } else {
        read_gr_reader(BufReader::new(file))
    }
}

fn read_gr_gz<R: Read>(r: R) -> io::Result<Graph<(), f64>> {
    let decoder = flate2::read::GzDecoder::new(r);
    read_gr_reader(BufReader::new(decoder))
}

fn read_gr_reader<R: BufRead>(reader: R) -> io::Result<Graph<(), f64>> {
    let mut graph: Graph<(), f64> = Graph::new();
    let mut nodes: Vec<NodeIndex> = Vec::new();
    let mut declared_n: Option<usize> = None;
    let mut declared_m: Option<usize> = None;
    let mut shift: Option<usize> = None;
    let mut deferred: Vec<(usize, usize, f64)> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim_start();
        if trimmed.is_empty() {
            continue;
        }
        let first = trimmed.as_bytes()[0];
        match first {
            b'c' => continue,
            b'p' => {
                let mut parts = trimmed.split_ascii_whitespace();
                parts.next();
                parts.next();
                let n = parts.next().and_then(|s| s.parse::<usize>().ok());
                let m = parts.next().and_then(|s| s.parse::<usize>().ok());
                if let Some(n) = n {
                    declared_n = Some(n);
                    declared_m = m;
                    if nodes.is_empty() {
                        graph.reserve_nodes(n);
                        if let Some(m) = m {
                            graph.reserve_edges(m);
                        }
                        nodes = (0..n).map(|_| graph.add_node(())).collect();
                    }
                }
            }
            b'a' | b'e' => {
                let mut parts = trimmed.split_ascii_whitespace();
                parts.next();
                let u = match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                    Some(v) => v,
                    None => continue,
                };
                let v = match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                    Some(v) => v,
                    None => continue,
                };
                let w = parts
                    .next()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0);

                if shift.is_none() {
                    if u == 0 || v == 0 {
                        shift = Some(0);
                        if !deferred.is_empty() {
                            for &(du, dv, dw) in &deferred {
                                push_edge(&mut graph, &mut nodes, du, dv, dw, 0);
                            }
                            deferred.clear();
                        }
                    } else {
                        deferred.push((u, v, w));
                        if deferred.len() < 4 {
                            continue;
                        }
                        shift = Some(1);
                        for &(du, dv, dw) in &deferred {
                            push_edge(&mut graph, &mut nodes, du, dv, dw, 1);
                        }
                        deferred.clear();
                        continue;
                    }
                }
                push_edge(&mut graph, &mut nodes, u, v, w, shift.unwrap());
            }
            _ => continue,
        }
    }

    if !deferred.is_empty() {
        let s = shift.unwrap_or(1);
        for &(du, dv, dw) in &deferred {
            push_edge(&mut graph, &mut nodes, du, dv, dw, s);
        }
    }

    if let Some(n) = declared_n {
        if graph.node_count() < n {
            while graph.node_count() < n {
                nodes.push(graph.add_node(()));
            }
        }
    }
    let _ = declared_m;
    Ok(graph)
}

#[inline]
fn push_edge(
    graph: &mut Graph<(), f64>,
    nodes: &mut Vec<NodeIndex>,
    u: usize,
    v: usize,
    w: f64,
    shift: usize,
) {
    if u < shift || v < shift {
        return;
    }
    let u0 = u - shift;
    let v0 = v - shift;
    let need = u0.max(v0) + 1;
    while nodes.len() < need {
        nodes.push(graph.add_node(()));
    }
    graph.add_edge(nodes[u0], nodes[v0], w);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_one_indexed() {
        let txt = "c comment\np sp 4 5\na 1 2 10\na 2 3 5\na 3 4 1\na 1 4 100\na 2 4 7\n";
        let g = read_gr_reader(Cursor::new(txt)).unwrap();
        assert_eq!(g.node_count(), 4);
        assert_eq!(g.edge_count(), 5);
    }

    #[test]
    fn parses_zero_indexed() {
        let txt = "p sp 3 2\na 0 1 1\na 1 2 1\n";
        let g = read_gr_reader(Cursor::new(txt)).unwrap();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn ignores_comments_blanks() {
        let txt = "c hello\n\nc world\np sp 2 1\na 1 2 4\n";
        let g = read_gr_reader(Cursor::new(txt)).unwrap();
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn missing_p_line_uses_inferred_n() {
        let txt = "a 1 5 1\na 2 3 1\n";
        let g = read_gr_reader(Cursor::new(txt)).unwrap();
        assert_eq!(g.node_count(), 5);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn streams_without_buffer_explosion() {

        let mut s = String::from("p sp 10000 100000\n");
        for i in 0..100_000u32 {
            let u = (i % 10_000) + 1;
            let v = ((i + 1) % 10_000) + 1;
            s.push_str(&format!("a {} {} 1\n", u, v));
        }
        let g = read_gr_reader(Cursor::new(s)).unwrap();
        assert_eq!(g.node_count(), 10_000);
        assert_eq!(g.edge_count(), 100_000);
    }
}
