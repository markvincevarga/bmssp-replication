use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::io::{self, Read, Write};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AosEdge {
    pub weight: f64,
    pub target: u32,
}

pub struct CsrAosGraph {
    pub offsets: Vec<u32>,
    pub edges: Vec<AosEdge>,
    pub node_count: usize,
}

impl CsrAosGraph {
    pub fn from_petgraph(graph: &Graph<(), f64>) -> Self {
        let n = graph.node_count();
        let m = graph.edge_count();
        let mut offsets = vec![0u32; n + 1];
        let mut edges = Vec::with_capacity(m);

        for u in 0..n {
            let node = petgraph::graph::NodeIndex::new(u);
            for edge in graph.edges(node) {
                edges.push(AosEdge {
                    weight: *edge.weight(),
                    target: edge.target().index() as u32,
                });
            }
            offsets[u + 1] = edges.len() as u32;
        }

        Self { offsets, edges, node_count: n }
    }

    #[inline(always)]
    pub fn edges(&self, u: usize) -> &[AosEdge] {
        let start = self.offsets[u] as usize;
        let end = self.offsets[u + 1] as usize;
        &self.edges[start..end]
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(b"DUAN")?;
        w.write_all(&1u32.to_le_bytes())?;
        w.write_all(&(self.node_count as u64).to_le_bytes())?;
        w.write_all(&(self.edges.len() as u64).to_le_bytes())?;

        for &off in &self.offsets {
            w.write_all(&off.to_le_bytes())?;
        }

        for e in &self.edges {
            w.write_all(&e.target.to_le_bytes())?;
            w.write_all(&e.weight.to_le_bytes())?;
        }

        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"DUAN" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unknown version"));
        }

        r.read_exact(&mut buf8)?;
        let node_count = u64::from_le_bytes(buf8) as usize;
        r.read_exact(&mut buf8)?;
        let edge_count = u64::from_le_bytes(buf8) as usize;

        let mut offsets = Vec::with_capacity(node_count + 1);
        for _ in 0..=node_count {
            r.read_exact(&mut buf4)?;
            offsets.push(u32::from_le_bytes(buf4));
        }

        let mut edges = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            r.read_exact(&mut buf4)?;
            let target = u32::from_le_bytes(buf4);
            r.read_exact(&mut buf8)?;
            let weight = f64::from_le_bytes(buf8);
            edges.push(AosEdge { weight, target });
        }

        Ok(Self { offsets, edges, node_count })
    }

    pub fn to_petgraph(&self) -> Graph<(), f64> {
        let mut graph = Graph::new();
        let nodes: Vec<_> = (0..self.node_count).map(|_| graph.add_node(())).collect();

        for u in 0..self.node_count {
            for e in self.edges(u) {
                graph.add_edge(nodes[u], NodeIndex::new(e.target as usize), e.weight);
            }
        }

        graph
    }
}
