use criterion::{black_box, criterion_group, criterion_main, Criterion};
use my_project::hello;

fn bench_hello(c: &mut Criterion) {
    c.bench_function("hello", |b| {
        b.iter(|| hello(black_box("criterion")));
    });
}

criterion_group!(benches, bench_hello);
criterion_main!(benches);
