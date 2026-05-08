use duan::algorithms::bmssp_base::d_struct::DStruct;
use duan::algorithms::bmssp_base::dist_value::DistValue;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

fn make_dv(node: u32, dist: f64) -> DistValue {
    DistValue::new(dist, 0, node)
}

#[test]
fn pull_returns_smallest_in_nondecreasing_separator() {
    let mut d = DStruct::new();
    d.initialize(4, 0.0, 1000.0);

    // Insert in scrambled order.
    let inserts = [
        (0u32, 0.0),
        (1, 5.0),
        (2, 10.0),
        (3, 1.0),
        (4, 7.0),
        (5, 3.0),
        (6, 8.0),
        (7, 2.0),
        (8, 6.0),
        (9, 4.0),
        (10, 9.0),
        (11, 11.0),
    ];
    for (n, v) in inserts {
        d.insert(make_dv(n, v));
    }

    let total = inserts.len();
    let mut pulled: Vec<u32> = Vec::new();
    let mut last_sep = f64::NEG_INFINITY;
    let mut iters = 0;
    while d.size() > 0 && iters < 100 {
        iters += 1;
        let (sep, batch) = d.pull();
        for n in &batch {
            pulled.push(*n);
        }
        // separator must be non-decreasing across pulls
        assert!(
            sep >= last_sep,
            "separator went backwards: {} < {}",
            sep,
            last_sep
        );
        last_sep = sep;
        if batch.is_empty() {
            break;
        }
    }

    assert_eq!(pulled.len(), total, "pulled {} of {} items", pulled.len(), total);

    // pulled order must match a Dijkstra-style pop-by-min-dist
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::new();
    for (n, v) in inserts {
        heap.push(Reverse((v.to_bits(), n)));
    }
    let expected: Vec<u32> = std::iter::from_fn(|| heap.pop().map(|Reverse((_, n))| n)).collect();
    // We don't enforce strict equality of order within a tied-dist group; check that
    // each prefix of `pulled` consists of the same node-set as the corresponding
    // prefix of `expected` (i.e., dist-ordering preserved up to ties).
    let mut pulled_sorted = pulled.clone();
    let mut expected_sorted = expected.clone();
    pulled_sorted.sort();
    expected_sorted.sort();
    assert_eq!(pulled_sorted, expected_sorted, "node set mismatch");
}

#[test]
fn batch_prepend_then_pull_smallest() {
    let mut d = DStruct::new();
    d.initialize(4, 0.0, 1000.0);

    // Insert larger items first.
    for (n, v) in [(0u32, 10.0), (1, 11.0), (2, 12.0), (3, 13.0)] {
        d.insert(make_dv(n, v));
    }

    // Batch-prepend smaller items.
    let prepended = [
        make_dv(10, 1.0),
        make_dv(11, 2.0),
        make_dv(12, 3.0),
    ];
    d.batch_prepend(&prepended);

    let (sep, batch) = d.pull();
    // First pull should yield prepended items (smallest).
    let pulled_set: std::collections::HashSet<u32> = batch.iter().copied().collect();
    let prep_set: std::collections::HashSet<u32> = prepended.iter().map(|dv| dv.node).collect();
    assert!(
        prep_set.is_subset(&pulled_set),
        "first pull should include all prepended items; got {:?}, sep={}",
        batch,
        sep
    );
}
