use criterion::{criterion_group, criterion_main, Criterion};
use duan::algorithms::*;

mod benchmark_framework;
use benchmark_framework::{
    benchmark_algorithms_comprehensive, AlgorithmRegistry, BenchmarkConfig,
};

#[cfg(feature = "track-alloc")]
#[global_allocator]
static GLOBAL: duan::tracking_alloc::TrackingAllocator = duan::tracking_alloc::TrackingAllocator;

fn benches(c: &mut Criterion) {
    let config = BenchmarkConfig::from_env();

    let registry = AlgorithmRegistry::new(config.algorithms.clone())
        .register("dijkstra", Dijkstra::new())
        .register("dijkstra_opt", DijkstraOpt::new())
        .register("bmssp_base", BmsspBase::new())
        .register("bmssp_base_transform", BmsspBase::with_transformation(2))
        .register("bmssp_o_1", BmsspO1::new())
        .register("bmssp_o_2", BmsspO2::new())
        .register("bmssp_o_3", BmsspO3::new())
        .register("bmssp_o_4", BmsspO4::new())
        .register("bmssp_o_5", BmsspO5::new())
        .register("bmssp_o_6", BmsspO6::new());

    benchmark_algorithms_comprehensive(c, &config, &registry);

    #[cfg(feature = "track-alloc")]
    benchmark_framework::benchmark_memory(&config, &registry);
}

criterion_group!(benchmark_group, benches);
criterion_main!(benchmark_group);
