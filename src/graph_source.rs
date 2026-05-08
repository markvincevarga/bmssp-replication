use std::fmt;

#[derive(Debug, Clone)]
pub enum GraphSource {
    FixedEdgeCount { edge_factor: usize },
    BarabasiAlbert { m: usize },
    WattsStrogatz { k: usize, beta: f64 },
    RandomGeometric { radius: f64 },
    Grid,
    CastroD3,
    CastroH3,
    CastroSGridED,
    CastroRGridED,
    CastroSGridR,
    CastroRGridR,
    DimacsRoad { name: String },
}

impl GraphSource {
    pub fn cache_key(&self) -> String {
        match self {
            Self::FixedEdgeCount { edge_factor } => format!("ef{}", edge_factor),
            Self::BarabasiAlbert { m } => format!("ba_m{}", m),
            Self::WattsStrogatz { k, beta } => {
                format!("ws_k{}_b{}", k, (beta * 100.0) as u32)
            }
            Self::RandomGeometric { radius } => {
                format!("rg_r{}", (radius * 1000.0) as u32)
            }
            Self::Grid => "grid".to_string(),
            Self::CastroD3 => "castro_d3".to_string(),
            Self::CastroH3 => "castro_h3".to_string(),
            Self::CastroSGridED => "castro_sgrided".to_string(),
            Self::CastroRGridED => "castro_rgrided".to_string(),
            Self::CastroSGridR => "castro_sgridr".to_string(),
            Self::CastroRGridR => "castro_rgridr".to_string(),
            Self::DimacsRoad { name } => format!("dimacs_{}", sanitize(name)),
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim();
        match s {
            "grid" => return Ok(Self::Grid),
            "castro:d3" | "d3" => return Ok(Self::CastroD3),
            "castro:h3" | "h3" => return Ok(Self::CastroH3),
            "castro:sgrided" | "sgrided" => return Ok(Self::CastroSGridED),
            "castro:rgrided" | "rgrided" => return Ok(Self::CastroRGridED),
            "castro:sgridr" | "sgridr" => return Ok(Self::CastroSGridR),
            "castro:rgridr" | "rgridr" => return Ok(Self::CastroRGridR),
            _ => {}
        }

        let (name, params) = s
            .split_once(':')
            .ok_or_else(|| format!("expected 'type:params', got '{}'", s))?;

        match name {
            "fixed" => {
                let ef = parse_param(params, "ef")?;
                Ok(Self::FixedEdgeCount {
                    edge_factor: ef.parse().map_err(|_| format!("invalid ef value: {}", ef))?,
                })
            }
            "ba" => {
                let m = parse_param(params, "m")?;
                Ok(Self::BarabasiAlbert {
                    m: m.parse().map_err(|_| format!("invalid m value: {}", m))?,
                })
            }
            "ws" => {
                let k_str = parse_param(params, "k")?;
                let beta_str = parse_param(params, "beta")?;
                Ok(Self::WattsStrogatz {
                    k: k_str
                        .parse()
                        .map_err(|_| format!("invalid k value: {}", k_str))?,
                    beta: beta_str
                        .parse()
                        .map_err(|_| format!("invalid beta value: {}", beta_str))?,
                })
            }
            "rg" => {
                let r = parse_param(params, "radius")?;
                Ok(Self::RandomGeometric {
                    radius: r
                        .parse()
                        .map_err(|_| format!("invalid radius value: {}", r))?,
                })
            }
            "dimacs" => {
                let n = parse_param(params, "name")?;
                if n.is_empty() {
                    return Err("dimacs name cannot be empty".to_string());
                }
                Ok(Self::DimacsRoad {
                    name: n.to_string(),
                })
            }
            _ => Err(format!("unknown graph source: {}", name)),
        }
    }

    pub fn ignores_node_count(&self) -> bool {
        matches!(self, Self::DimacsRoad { .. })
    }
}

fn sanitize(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn parse_param<'a>(params: &'a str, key: &str) -> Result<&'a str, String> {
    for part in params.split(':') {
        if let Some((k, v)) = part.split_once('=') {
            if k == key {
                return Ok(v);
            }
        }
    }
    Err(format!("missing parameter '{}'", key))
}

impl fmt::Display for GraphSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FixedEdgeCount { edge_factor } => write!(f, "fixed(ef={})", edge_factor),
            Self::BarabasiAlbert { m } => write!(f, "barabasi-albert(m={})", m),
            Self::WattsStrogatz { k, beta } => write!(f, "watts-strogatz(k={},β={})", k, beta),
            Self::RandomGeometric { radius } => write!(f, "random-geometric(r={})", radius),
            Self::Grid => write!(f, "grid"),
            Self::CastroD3 => write!(f, "castro-d3"),
            Self::CastroH3 => write!(f, "castro-h3"),
            Self::CastroSGridED => write!(f, "castro-sgrided"),
            Self::CastroRGridED => write!(f, "castro-rgrided"),
            Self::CastroSGridR => write!(f, "castro-sgridr"),
            Self::CastroRGridR => write!(f, "castro-rgridr"),
            Self::DimacsRoad { name } => write!(f, "dimacs({})", name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_fixed() {
        let src = GraphSource::parse("fixed:ef=8").unwrap();
        assert!(matches!(src, GraphSource::FixedEdgeCount { edge_factor: 8 }));
        assert_eq!(src.cache_key(), "ef8");
    }

    #[test]
    fn parse_ba() {
        let src = GraphSource::parse("ba:m=5").unwrap();
        assert!(matches!(src, GraphSource::BarabasiAlbert { m: 5 }));
        assert_eq!(src.cache_key(), "ba_m5");
    }

    #[test]
    fn parse_ws() {
        let src = GraphSource::parse("ws:k=6:beta=0.3").unwrap();
        if let GraphSource::WattsStrogatz { k, beta } = src {
            assert_eq!(k, 6);
            assert!((beta - 0.3).abs() < 1e-9);
        } else {
            panic!("expected WattsStrogatz");
        }
    }

    #[test]
    fn parse_rg() {
        let src = GraphSource::parse("rg:radius=0.1").unwrap();
        if let GraphSource::RandomGeometric { radius } = src {
            assert!((radius - 0.1).abs() < 1e-9);
        } else {
            panic!("expected RandomGeometric");
        }
    }

    #[test]
    fn parse_grid() {
        let src = GraphSource::parse("grid").unwrap();
        assert!(matches!(src, GraphSource::Grid));
        assert_eq!(src.cache_key(), "grid");
    }

    #[test]
    fn parse_castro_random() {
        assert!(matches!(
            GraphSource::parse("castro:d3").unwrap(),
            GraphSource::CastroD3
        ));
        assert!(matches!(
            GraphSource::parse("d3").unwrap(),
            GraphSource::CastroD3
        ));
        assert!(matches!(
            GraphSource::parse("h3").unwrap(),
            GraphSource::CastroH3
        ));
    }

    #[test]
    fn parse_castro_grids() {
        assert!(matches!(
            GraphSource::parse("sgrided").unwrap(),
            GraphSource::CastroSGridED
        ));
        assert!(matches!(
            GraphSource::parse("rgrided").unwrap(),
            GraphSource::CastroRGridED
        ));
        assert!(matches!(
            GraphSource::parse("sgridr").unwrap(),
            GraphSource::CastroSGridR
        ));
        assert!(matches!(
            GraphSource::parse("rgridr").unwrap(),
            GraphSource::CastroRGridR
        ));
    }

    #[test]
    fn parse_dimacs() {
        let s = GraphSource::parse("dimacs:name=USA-road-t.NY").unwrap();
        if let GraphSource::DimacsRoad { name } = &s {
            assert_eq!(name, "USA-road-t.NY");
        } else {
            panic!("expected DimacsRoad");
        }
        assert_eq!(s.cache_key(), "dimacs_USA-road-t_NY");
        assert!(s.ignores_node_count());
    }

    #[test]
    fn parse_unknown_fails() {
        assert!(GraphSource::parse("foo:x=1").is_err());
    }
}
