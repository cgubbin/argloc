use apm::{
    ArgumentOracle, HolomorphicFunction, IntegratorConfig, MaxWeightedScorePolicy, QuadTreeConfig,
    Rect, ShiftSplitOnBoundary, run_with_policy,
};

use num_complex::Complex;

#[derive(Debug, Clone, Copy)]
struct Linear {
    root: Complex<f64>,
}

impl HolomorphicFunction for Linear {
    type Complex = Complex<f64>;

    fn value(&self, z: Complex<f64>) -> Complex<f64> {
        z - self.root
    }

    fn derivative(&self, _z: Complex<f64>) -> Complex<f64> {
        Complex::new(1.0, 0.0)
    }
}

#[test]
fn root_on_uniform_split_line_recovers_with_shifted_subdivision() {
    let domain = Rect::new(0.0, 1.0, 0.0, 1.0).unwrap();

    // This root lies on the first vertical uniform split line x = 0.5.
    let root = Complex::new(0.5, 0.25);

    let oracle = ArgumentOracle::new(Linear { root }, IntegratorConfig::default(), 1e-10, 1e-6);

    let config = QuadTreeConfig::new(1e-3)
        .with_max_iter(20)
        .with_max_depth(8)
        .with_max_leaves(1_000);

    let result = run_with_policy(
        domain,
        oracle,
        MaxWeightedScorePolicy,
        ShiftSplitOnBoundary {
            shift_fraction: 0.1,
        },
        config,
    )
    .unwrap();

    let root_cells: Vec<_> = result
        .iter()
        .filter(|leaf| leaf.data().root_count > 0)
        .collect();

    assert!(!root_cells.is_empty());

    let total_count: isize = root_cells.iter().map(|leaf| leaf.data().root_count).sum();

    assert_eq!(total_count, 1);
}
