use duan::algorithms::bmssp_base::d_struct::DStruct;
use duan::algorithms::bmssp_base::dist_value::DistValue;
use std::collections::{HashMap, HashSet};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn range(&mut self, lo: u64, hi_exc: u64) -> u64 {
        lo + (self.next_u64() % (hi_exc - lo))
    }

    fn float_in(&mut self, lo: f64, hi: f64) -> f64 {
        let pct = (self.next_u64() % 1_000_001) as f64 / 1_000_000.0;
        let raw = lo + pct * (hi - lo);
        sanitize(raw)
    }
}

fn sanitize(w: f64) -> f64 {
    (w * 1e10).round() / 1e10
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Insert(u32, f64),
    BatchPrepend(usize, [u32; 8], [f64; 8]),
    Pull,
    Erase(u32),
}

const UPPER: f64 = 1_000_000.0;

fn run_property(seed: u64, n_ops: usize, m: usize, n_nodes: u32) -> Result<(), String> {
    let mut d = DStruct::new();
    d.initialize(m, 0.0, UPPER);

    let mut reference: HashMap<u32, f64> = HashMap::new();
    let mut rng = Rng::new(seed);
    let mut history: Vec<Op> = Vec::with_capacity(n_ops);
    let mut floor: f64 = 0.0;

    for op_i in 0..n_ops {
        let pick = rng.range(0, 100);
        let cur_min = reference.values().cloned().fold(UPPER, f64::min);
        let op = build_op(pick, &mut rng, n_nodes, floor, cur_min);
        history.push(op);

        if let Op::Pull = op {
            let before = reference.clone();
            let (sep, batch) = d.pull();
            check_pull(sep, &batch, &before).map_err(|e| {
                format!(
                    "FAIL at op {} (Pull):\n  {}\n\nseed={:#x} m={} floor={}\nfull history ({} ops):\n{:#?}",
                    op_i,
                    e,
                    seed,
                    m,
                    floor,
                    history.len(),
                    &history
                )
            })?;
            for n in &batch {
                reference.remove(n);
            }
            check_size(&d, &reference, "after Pull").map_err(|e| {
                format!(
                    "FAIL at op {} (Pull): {}\nseed={:#x} m={}",
                    op_i, e, seed, m
                )
            })?;
            if !batch.is_empty() {
                floor = sep;
            }
            if reference.is_empty() {
                floor = 0.0;
            }
        } else if let Err(msg) = apply_op(&op, &mut d, &mut reference) {
            return Err(format!(
                "FAIL at op {} ({:?}):\n  {}\n\nseed={:#x} m={} floor={}\nfull history ({} ops):\n{:#?}",
                op_i,
                op,
                msg,
                seed,
                m,
                floor,
                history.len(),
                &history
            ));
        }
    }

    drain_and_check(&mut d, &mut reference, &history, seed, m)
}

fn build_op(pick: u64, rng: &mut Rng, n_nodes: u32, floor: f64, cur_min: f64) -> Op {
    if pick < 55 {
        let node = rng.range(0, n_nodes as u64) as u32;
        let lo = floor.max(0.0);
        let hi = (UPPER * 0.999).max(lo + 0.001);
        let dist = rng.float_in(lo, hi);
        Op::Insert(node, dist)
    } else if pick < 65 {
        if cur_min <= 0.001 {
            return Op::Pull;
        }
        let count = rng.range(1, 8) as usize;
        let mut nodes = [0u32; 8];
        let mut dists = [0.0f64; 8];
        for i in 0..count {
            nodes[i] = rng.range(0, n_nodes as u64) as u32;
            dists[i] = rng.float_in(0.0, cur_min - 0.001);
        }
        Op::BatchPrepend(count, nodes, dists)
    } else if pick < 95 {
        Op::Pull
    } else {
        let node = rng.range(0, n_nodes as u64) as u32;
        Op::Erase(node)
    }
}

fn apply_op(
    op: &Op,
    d: &mut DStruct,
    reference: &mut HashMap<u32, f64>,
) -> Result<(), String> {
    match *op {
        Op::Insert(node, dist) => {
            let dv = DistValue::new(dist, 0, node);
            let canonical = dv.dist;
            d.insert(dv);
            let entry = reference.entry(node).or_insert(f64::INFINITY);
            if *entry > canonical {
                *entry = canonical;
            }
            check_size(d, reference, "after Insert")
        }
        Op::BatchPrepend(count, nodes, dists) => {
            let mut entries: Vec<DistValue> = Vec::with_capacity(count);
            for i in 0..count {
                entries.push(DistValue::new(dists[i], 0, nodes[i]));
            }
            d.batch_prepend(&entries);
            for dv in &entries {
                let entry = reference.entry(dv.node).or_insert(f64::INFINITY);
                if *entry > dv.dist {
                    *entry = dv.dist;
                }
            }
            check_size(d, reference, "after BatchPrepend")
        }
        Op::Pull => {
            let before = reference.clone();
            let (sep, batch) = d.pull();
            check_pull(sep, &batch, &before)?;
            for n in &batch {
                reference.remove(n);
            }
            check_size(d, reference, "after Pull")
        }
        Op::Erase(node) => {
            d.erase(node);
            reference.remove(&node);
            check_size(d, reference, "after Erase")
        }
    }
}

fn check_pull(sep: f64, batch: &[u32], before: &HashMap<u32, f64>) -> Result<(), String> {
    let mut seen = HashSet::new();
    for n in batch {
        if !seen.insert(*n) {
            return Err(format!("duplicate node {} in pull batch {:?}", n, batch));
        }
    }
    for n in batch {
        if !before.contains_key(n) {
            return Err(format!(
                "pull returned node {} not in reference; sep={}, batch={:?}",
                n, sep, batch
            ));
        }
    }

    if before.is_empty() {
        if !batch.is_empty() {
            return Err(format!("pull from empty returned batch {:?}", batch));
        }
        return Ok(());
    }

    if batch.is_empty() {
        return Err(format!(
            "pull returned empty batch but reference has {} entries (smallest={})",
            before.len(),
            before.values().cloned().fold(f64::INFINITY, f64::min)
        ));
    }

    let returned_dists: Vec<f64> = batch.iter().map(|n| before[n]).collect();
    let max_ret = returned_dists
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let returned_set: HashSet<u32> = batch.iter().copied().collect();
    let mut min_rem = f64::INFINITY;
    let mut min_rem_node: Option<u32> = None;
    for (&n, &dist) in before {
        if !returned_set.contains(&n) && dist < min_rem {
            min_rem = dist;
            min_rem_node = Some(n);
        }
    }

    if !(sep > max_ret) {
        return Err(format!(
            "separator not > max(returned): sep={} max_ret={} returned_dists={:?} batch={:?}",
            sep, max_ret, returned_dists, batch
        ));
    }
    if min_rem_node.is_some() && !(sep <= min_rem) {
        return Err(format!(
            "separator not <= min(remaining): sep={} min_rem={} (node {}) batch={:?}",
            sep,
            min_rem,
            min_rem_node.unwrap(),
            batch
        ));
    }
    for (&n, &dist) in before {
        if !returned_set.contains(&n) && dist < sep {
            return Err(format!(
                "node {} with dist {} < sep {} was NOT returned; batch={:?}",
                n, dist, sep, batch
            ));
        }
    }
    Ok(())
}

fn check_size(d: &DStruct, reference: &HashMap<u32, f64>, ctx: &str) -> Result<(), String> {
    let dsize = d.size();
    let rsize = reference.len();
    if dsize != rsize {
        return Err(format!(
            "{}: dstruct.size()={}, reference.len()={}",
            ctx, dsize, rsize
        ));
    }
    Ok(())
}

fn drain_and_check(
    d: &mut DStruct,
    reference: &mut HashMap<u32, f64>,
    history: &[Op],
    seed: u64,
    m: usize,
) -> Result<(), String> {
    let mut iters = 0usize;
    let limit = reference.len() * 4 + 1000;
    while d.size() > 0 {
        iters += 1;
        if iters > limit {
            return Err(format!(
                "drain exceeded {} iters; d.size()={} reference.len()={} seed={:#x} m={}",
                iters,
                d.size(),
                reference.len(),
                seed,
                m
            ));
        }
        let before = reference.clone();
        let (sep, batch) = d.pull();
        check_pull(sep, &batch, &before).map_err(|e| {
            let tail_start = history.len().saturating_sub(50);
            format!(
                "drain pull failed: {}\nseed={:#x} m={}\nlast {} ops:\n{:#?}",
                e,
                seed,
                m,
                history.len() - tail_start,
                &history[tail_start..]
            )
        })?;
        for n in &batch {
            reference.remove(n);
        }
    }
    if !reference.is_empty() {
        let leftover: Vec<(u32, f64)> = reference.iter().take(10).map(|(n, d)| (*n, *d)).collect();
        return Err(format!(
            "after drain: reference has {} leftover nodes (lost in queue): {:?}\nseed={:#x} m={}",
            reference.len(),
            leftover,
            seed,
            m
        ));
    }
    Ok(())
}

#[test]
fn property_random_m4_seed_a() {
    if let Err(e) = run_property(0xDEAD_BEEF_DEAD_BEEF, 10_000, 4, 50) {
        panic!("{}", e);
    }
}

#[test]
fn property_random_m8_seed_b() {
    if let Err(e) = run_property(0xC0FF_EE00_BAAD_F00D, 10_000, 8, 100) {
        panic!("{}", e);
    }
}

#[test]
fn property_random_m16_seed_c() {
    if let Err(e) = run_property(0xFEED_FACE_CAFE_BABE, 10_000, 16, 200) {
        panic!("{}", e);
    }
}

#[test]
fn property_random_m1_stress() {
    if let Err(e) = run_property(0x1234_5678_ABCD_1234, 5_000, 1, 30) {
        panic!("{}", e);
    }
}

#[test]
fn property_random_long_m4() {
    if let Err(e) = run_property(0xABCD_1234_5678_DEAD, 50_000, 4, 50) {
        panic!("{}", e);
    }
}

#[test]
fn property_random_long_m8() {
    if let Err(e) = run_property(0xBADD_CAFE_F00D_1010, 50_000, 8, 100) {
        panic!("{}", e);
    }
}

#[test]
fn property_random_seed_d() {
    if let Err(e) = run_property(0x1111_2222_3333_4444, 10_000, 4, 50) {
        panic!("{}", e);
    }
}

#[test]
fn property_random_seed_e() {
    if let Err(e) = run_property(0x9999_AAAA_BBBB_CCCC, 10_000, 8, 100) {
        panic!("{}", e);
    }
}

#[test]
fn property_random_high_collision_m4() {
    // Small node range relative to ops → many duplicate node inserts → exercises
    // stale-filter and re-insert paths.
    if let Err(e) = run_property(0xDEAD_BEEF_DEAD_FACE, 20_000, 4, 10) {
        panic!("{}", e);
    }
}

#[test]
fn property_insert_pull_only_m4() {
    if let Err(e) = run_property_simple(0x55AA_55AA_55AA_55AA, 10_000, 4, 50) {
        panic!("{}", e);
    }
}

#[test]
fn property_insert_pull_only_m1() {
    if let Err(e) = run_property_simple(0xA5A5_A5A5_A5A5_A5A5, 10_000, 1, 30) {
        panic!("{}", e);
    }
}

#[test]
fn property_insert_prepend_pull_m4() {
    if let Err(e) = run_property_no_erase(0x77AA_77AA_77AA_77AA, 5_000, 4, 50) {
        panic!("{}", e);
    }
}

fn run_property_no_erase(seed: u64, n_ops: usize, m: usize, n_nodes: u32) -> Result<(), String> {
    let mut d = DStruct::new();
    d.initialize(m, 0.0, UPPER);

    let mut reference: HashMap<u32, f64> = HashMap::new();
    let mut rng = Rng::new(seed);
    let mut floor: f64 = 0.0;
    let mut history: Vec<Op> = Vec::new();

    for op_i in 0..n_ops {
        let pick = rng.range(0, 100);
        let cur_min = reference.values().cloned().fold(UPPER, f64::min);
        let op = if pick < 55 {
            let node = rng.range(0, n_nodes as u64) as u32;
            let lo = floor.max(0.0);
            let hi = (UPPER * 0.999).max(lo + 0.001);
            Op::Insert(node, rng.float_in(lo, hi))
        } else if pick < 70 {
            if cur_min <= 0.001 {
                Op::Pull
            } else {
                let count = rng.range(1, 8) as usize;
                let mut nodes = [0u32; 8];
                let mut dists = [0.0f64; 8];
                for i in 0..count {
                    nodes[i] = rng.range(0, n_nodes as u64) as u32;
                    dists[i] = rng.float_in(0.0, cur_min - 0.001);
                }
                Op::BatchPrepend(count, nodes, dists)
            }
        } else {
            Op::Pull
        };
        history.push(op);

        if let Op::Pull = op {
            let before = reference.clone();
            let (sep, batch) = d.pull();
            check_pull(sep, &batch, &before).map_err(|e| {
                format!(
                    "FAIL (no-erase) at op {} Pull: {}\nseed={:#x} m={}\nhistory ({} ops):\n{:#?}",
                    op_i, e, seed, m, history.len(), &history
                )
            })?;
            for n in &batch {
                reference.remove(n);
            }
            check_size(&d, &reference, "after Pull")
                .map_err(|e| format!("FAIL at op {} Pull: {}", op_i, e))?;
            if !batch.is_empty() {
                floor = sep;
            }
            if reference.is_empty() {
                floor = 0.0;
            }
        } else {
            apply_op(&op, &mut d, &mut reference)
                .map_err(|e| format!("FAIL at op {} ({:?}): {}", op_i, op, e))?;
        }
    }

    drain_and_check(&mut d, &mut reference, &history, seed, m)
}

#[test]
fn property_insert_pull_erase_m4() {
    if let Err(e) = run_property_no_prepend(0x33CC_33CC_33CC_33CC, 10_000, 4, 50) {
        panic!("{}", e);
    }
}

fn run_property_no_prepend(seed: u64, n_ops: usize, m: usize, n_nodes: u32) -> Result<(), String> {
    let mut d = DStruct::new();
    d.initialize(m, 0.0, UPPER);

    let mut reference: HashMap<u32, f64> = HashMap::new();
    let mut rng = Rng::new(seed);
    let mut floor: f64 = 0.0;
    let mut history: Vec<Op> = Vec::new();

    for op_i in 0..n_ops {
        let pick = rng.range(0, 100);
        let op = if pick < 60 {
            let node = rng.range(0, n_nodes as u64) as u32;
            let lo = floor.max(0.0);
            let hi = (UPPER * 0.999).max(lo + 0.001);
            Op::Insert(node, rng.float_in(lo, hi))
        } else if pick < 90 {
            Op::Pull
        } else {
            Op::Erase(rng.range(0, n_nodes as u64) as u32)
        };
        history.push(op);

        if let Op::Pull = op {
            let before = reference.clone();
            let (sep, batch) = d.pull();
            check_pull(sep, &batch, &before).map_err(|e| {
                format!(
                    "FAIL (no-prepend) at op {} Pull: {}\nseed={:#x} m={}\nhistory ({} ops):\n{:#?}",
                    op_i, e, seed, m, history.len(), &history
                )
            })?;
            for n in &batch {
                reference.remove(n);
            }
            check_size(&d, &reference, "after Pull")
                .map_err(|e| format!("FAIL at op {} Pull: {}", op_i, e))?;
            if !batch.is_empty() {
                floor = sep;
            }
            if reference.is_empty() {
                floor = 0.0;
            }
        } else {
            apply_op(&op, &mut d, &mut reference)
                .map_err(|e| format!("FAIL at op {} ({:?}): {}", op_i, op, e))?;
        }
    }

    drain_and_check(&mut d, &mut reference, &history, seed, m)
}

fn run_property_simple(seed: u64, n_ops: usize, m: usize, n_nodes: u32) -> Result<(), String> {
    let mut d = DStruct::new();
    d.initialize(m, 0.0, UPPER);

    let mut reference: HashMap<u32, f64> = HashMap::new();
    let mut rng = Rng::new(seed);
    let mut floor: f64 = 0.0;
    let mut history: Vec<Op> = Vec::new();

    for op_i in 0..n_ops {
        let pick = rng.range(0, 100);
        let op = if pick < 70 {
            let node = rng.range(0, n_nodes as u64) as u32;
            let lo = floor.max(0.0);
            let hi = (UPPER * 0.999).max(lo + 0.001);
            Op::Insert(node, rng.float_in(lo, hi))
        } else {
            Op::Pull
        };
        history.push(op);

        if let Op::Pull = op {
            let before = reference.clone();
            let (sep, batch) = d.pull();
            check_pull(sep, &batch, &before).map_err(|e| {
                format!(
                    "FAIL (insert+pull only) at op {} Pull: {}\nseed={:#x} m={} floor={}\nhistory ({} ops):\n{:#?}",
                    op_i, e, seed, m, floor, history.len(), &history
                )
            })?;
            for n in &batch {
                reference.remove(n);
            }
            check_size(&d, &reference, "after Pull")
                .map_err(|e| format!("FAIL at op {} Pull: {}", op_i, e))?;
            if !batch.is_empty() {
                floor = sep;
            }
            if reference.is_empty() {
                floor = 0.0;
            }
        } else {
            apply_op(&op, &mut d, &mut reference)
                .map_err(|e| format!("FAIL at op {} ({:?}): {}", op_i, op, e))?;
        }
    }

    drain_and_check(&mut d, &mut reference, &history, seed, m)
}

#[test]
fn deterministic_erase_then_reinsert_larger_dist() {
    let mut d = DStruct::new();
    d.initialize(4, 0.0, 1000.0);

    d.insert(DistValue::new(3.0, 0, 7));
    assert_eq!(d.size(), 1);

    d.erase(7);
    assert_eq!(d.size(), 0);

    d.insert(DistValue::new(5.0, 0, 7));
    assert_eq!(d.size(), 1, "node should reappear after erase + larger reinsert");

    let (sep, batch) = d.pull();
    assert_eq!(batch, vec![7], "node 7 must be returned, not lost; sep={}", sep);
    assert_eq!(d.size(), 0);
}

#[test]
fn deterministic_erase_then_reinsert_smaller_dist() {
    let mut d = DStruct::new();
    d.initialize(4, 0.0, 1000.0);

    d.insert(DistValue::new(8.0, 0, 3));
    d.erase(3);
    d.insert(DistValue::new(2.0, 0, 3));
    assert_eq!(d.size(), 1);

    let (_, batch) = d.pull();
    assert_eq!(batch, vec![3]);
}

#[test]
fn deterministic_many_ties_pull() {
    let mut d = DStruct::new();
    d.initialize(3, 0.0, 1000.0);

    for n in 0u32..10 {
        d.insert(DistValue::new(5.0, 0, n));
    }
    assert_eq!(d.size(), 10);

    let mut total_pulled = 0usize;
    let mut iters = 0;
    while d.size() > 0 && iters < 50 {
        iters += 1;
        let (_, batch) = d.pull();
        if batch.is_empty() {
            break;
        }
        total_pulled += batch.len();
    }
    assert_eq!(total_pulled, 10, "all 10 tied items must be returned");
    assert_eq!(d.size(), 0);
}

#[test]
fn deterministic_d1_first_block_all_stale_later_block_fresh() {
    let mut d = DStruct::new();
    d.initialize(1, 0.0, 1_000_000.0);

    d.insert(DistValue::new(100.0, 0, 1));
    d.insert(DistValue::new(200.0, 0, 2));
    d.insert(DistValue::new(50.0, 0, 3));
    d.erase(3);
    d.batch_prepend(&[DistValue::new(10.0, 0, 4)]);
    assert_eq!(d.size(), 3, "expect nodes 1, 2, 4 live");

    let (sep, batch) = d.pull();
    assert_eq!(batch, vec![4], "first pull returns smallest D₀ item; got batch={:?} sep={}", batch, sep);
    assert!(
        sep <= 100.0,
        "separator must be ≤ next-smallest 100 (node 1); got sep={} (D₁ first block is all-stale; later block has node 1 fresh)",
        sep
    );
}

#[test]
fn deterministic_d0_unwalked_tail_smaller_than_d1_first() {
    let mut d = DStruct::new();
    d.initialize(1, 0.0, 1_000_000.0);

    d.insert(DistValue::new(170069.0, 0, 1));
    d.batch_prepend(&[
        DistValue::new(12763.0, 0, 2),
        DistValue::new(20919.0, 0, 3),
    ]);
    assert_eq!(d.size(), 3);

    let (sep, batch) = d.pull();
    assert_eq!(
        batch,
        vec![2],
        "first pull must return only smallest; got batch={:?} sep={}",
        batch,
        sep
    );
    assert!(
        sep <= 20919.0,
        "separator must be <= next-smallest 20919; got sep={}",
        sep
    );
}

#[test]
fn deterministic_walked_d1_tied_with_unwalked_d1() {
    // D₁ contains three blocks all with items at the same dist (e.g. uniform-
    // weight grid graphs produce many tied d_hat values). With M=1, step 2 of
    // pull walks the first block, then unwalked has another block at the
    // same value. Without tie resolution, cap-filter would yield empty
    // eligible and the fallback would emit an invalid separator. Tie
    // resolution must walk the tied unwalked blocks until cap rises strictly.
    let mut d = DStruct::new();
    d.initialize(1, 0.0, 1_000_000.0);

    d.insert(DistValue::new(100.0, 0, 1));
    d.insert(DistValue::new(100.0, 0, 2));
    d.insert(DistValue::new(100.0, 0, 3));
    d.insert(DistValue::new(200.0, 0, 4));
    assert_eq!(d.size(), 4);

    let (sep, batch) = d.pull();
    assert!(
        !batch.is_empty(),
        "pull must return ≥1 item from {{1,2,3}} all at 100; got sep={}",
        sep
    );
    let returned: std::collections::HashSet<u32> = batch.iter().copied().collect();
    let valid: std::collections::HashSet<u32> = [1, 2, 3].into_iter().collect();
    assert!(
        returned.is_subset(&valid),
        "pull must return only nodes from the tied 100-cluster; got {:?}",
        batch
    );
    assert!(
        sep > 100.0,
        "separator must be strictly > max(returned)=100; got sep={}",
        sep
    );
    assert!(
        sep <= 200.0,
        "separator must be ≤ min(remaining); only node 4 at 200 left after pulling some 100s. got sep={}",
        sep
    );
}

#[test]
fn deterministic_walked_d0_tied_with_unwalked_d1() {
    // Cross-list tie: D₀ has an item at 50, D₁ has unwalked items at 50 too.
    // Without tie resolution, walked D₀'s 50 would tie with unwalked D₁'s 50,
    // and cap-filter would defer the walked item, eligible empty.
    let mut d = DStruct::new();
    d.initialize(1, 0.0, 1_000_000.0);

    // D₁ gets items at 50 and 200.
    d.insert(DistValue::new(50.0, 0, 10));
    d.insert(DistValue::new(200.0, 0, 11));

    // Pull once to set floor, then we'll set up the cross-list tie.
    let (sep1, _) = d.pull();
    assert!(sep1 > 0.0);
    // Now D contains node 11 at 200. best={11:200}.

    // BatchPrepend with 50 < cur_min(200). Now D₀ has item at 50.
    d.batch_prepend(&[DistValue::new(50.0, 0, 12)]);
    // Insert item at 50 too — this updates best[12] check (12 already has 50, same value, rejected).
    // Use a different node:
    d.insert(DistValue::new(50.0, 0, 13));
    // best = {11:200, 12:50, 13:50}. D₀ has [(12, 50)]. D₁ has [(13, 50), (11, 200)] across blocks.
    assert_eq!(d.size(), 3);

    let (sep, batch) = d.pull();
    let returned: std::collections::HashSet<u32> = batch.iter().copied().collect();
    // Both 12 and 13 are at 50 (smallest); 11 at 200.
    // Pull must return at least one of {12, 13} and not 11.
    assert!(
        !returned.contains(&11),
        "node 11 (dist 200) must not be returned before tied 50s; got {:?} sep={}",
        batch,
        sep
    );
    assert!(
        sep > 50.0,
        "separator must be > 50; got sep={}",
        sep
    );
}

#[test]
fn deterministic_tie_extension_when_max_in_returned_equals_pivot() {
    // Crafted to expose the bug where `select_nth_unstable_by` puts a non-tied
    // item at index M-1 even though another item in eligible[..M] ties with
    // pivot's dist. With M=2 and items [(5, X), (10, Y), (10, Z)], select_nth(2)
    // makes items[2] = pivot at dist 10; items[..2] contain the dist-5 item AND
    // the dist-10 item (in some unspecified order). A tie-extension check that
    // only inspects items[M-1] would miss the tie when items[M-1] happens to be
    // the dist-5 item; the strict M-partition would then return one dist-10 and
    // leave another dist-10 in leftover, producing sep ≤ max(returned).
    //
    // Because the failure depends on select_nth's internal ordering, run many
    // permutations to maximise chance of triggering the unsound branch.
    for trial in 0..32u32 {
        let mut d = DStruct::new();
        d.initialize(2, 0.0, 1000.0);

        // Distinct nodes so dedupe doesn't merge items. dist=5 for one node,
        // dist=10 for two nodes. Vary node ordering each trial via simple
        // rotation to perturb select_nth's internal pivot choice.
        let nodes_5 = trial % 7;
        let nodes_10_a = (trial + 1) % 7 + 7;
        let nodes_10_b = (trial + 2) % 7 + 14;
        d.insert(DistValue::new(5.0, 0, nodes_5));
        d.insert(DistValue::new(10.0, 0, nodes_10_a));
        d.insert(DistValue::new(10.0, 1, nodes_10_b));

        let (sep, batch) = d.pull();
        let max_ret = batch
            .iter()
            .map(|&n| if n == nodes_5 { 5.0 } else { 10.0 })
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            sep > max_ret,
            "trial {}: separator {} must be strictly > max(returned)={} batch={:?}",
            trial,
            sep,
            max_ret,
            batch
        );
    }
}

#[test]
fn deterministic_pull_separator_strict_above_returned() {
    let mut d = DStruct::new();
    d.initialize(2, 0.0, 1000.0);

    for (n, v) in [(0u32, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)] {
        d.insert(DistValue::new(v, 0, n));
    }

    let (sep, batch) = d.pull();
    assert!(!batch.is_empty());
    let max_ret = batch
        .iter()
        .map(|n| match *n {
            0 => 1.0,
            1 => 2.0,
            2 => 3.0,
            3 => 4.0,
            _ => f64::NAN,
        })
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(sep > max_ret, "separator {} must be strictly > max returned {}", sep, max_ret);
}
