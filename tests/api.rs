use argloc::{ArgumentConfig, ComplexFunction, SearchTarget, find_poles, find_singularities};
use num_complex::Complex;
use quad_rs::IntegratorConfig;
use quadtree_core::{QuadTreeConfig, Rect};

const TOL: f64 = 1e-2;

fn config() -> ArgumentConfig<f64> {
    let tree = QuadTreeConfig::new(1e-3)
        .with_max_iter(250)
        .with_max_depth(12)
        .with_max_leaves(20_000)
        .with_target_window(1);

    let integrator = IntegratorConfig::default()
        .with_absolute_tolerance(1e-10)
        .with_relative_tolerance(1e-10);

    ArgumentConfig::new(1e-3)
        .with_quad_tree(tree)
        .with_integrator(integrator)
        .with_zero_tol(1e-10)
        .with_residual_tolerance(1e-6)
        .with_boundary_shift_fraction(0.1)
}

fn assert_root_near(roots: &[argloc::SingularPointEstimate<Complex<f64>>], expected: Complex<f64>) {
    let best = roots
        .iter()
        .min_by(|a, b| {
            let da = (a.location - expected).norm();
            let db = (b.location - expected).norm();
            da.partial_cmp(&db).unwrap()
        })
        .expect("expected at least one root");

    assert!(
        (best.location - expected).norm() < TOL,
        "no root near {expected}; closest was {:?}",
        best.location
    );
}

#[derive(Debug, Clone, Copy)]
struct Quadratic;

impl ComplexFunction for Quadratic {
    type Complex = Complex<f64>;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        z * z - Complex::new(1.0, 0.0)
    }

    fn derivative(&self, z: Self::Complex) -> Self::Complex {
        Complex::new(2.0, 0.0) * z
    }
}

#[test]
fn finds_two_real_roots_of_quadratic() {
    let domain = Rect::new(-2.0, 2.0, -1.5, 1.5).unwrap();

    let result = find_singularities(Quadratic, domain, SearchTarget::Zeros, config()).unwrap();

    assert_eq!(
        result.points.iter().map(|r| r.multiplicity).sum::<usize>(),
        2
    );

    assert_root_near(&result.points[..], Complex::new(-1.0, 0.0));
    assert_root_near(&result.points[..], Complex::new(1.0, 0.0));
}

#[derive(Debug, Clone, Copy)]
struct CubicUnity;

impl ComplexFunction for CubicUnity {
    type Complex = Complex<f64>;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        z * z * z - Complex::new(1.0, 0.0)
    }

    fn derivative(&self, z: Self::Complex) -> Self::Complex {
        Complex::new(3.0, 0.0) * z * z
    }
}

#[test]
fn finds_roots_of_unity() {
    let domain = Rect::new(-1.5, 1.5, -1.5, 1.5).unwrap();

    let result = find_singularities(CubicUnity, domain, SearchTarget::Zeros, config()).unwrap();

    assert_eq!(
        result.points.iter().map(|r| r.multiplicity).sum::<usize>(),
        3
    );

    assert_root_near(&result.points[..], Complex::new(1.0, 0.0));
    assert_root_near(&result.points[..], Complex::new(-0.5, 3.0_f64.sqrt() / 2.0));
    assert_root_near(
        &result.points[..],
        Complex::new(-0.5, -3.0_f64.sqrt() / 2.0),
    );
}

#[derive(Debug, Clone, Copy)]
struct ShiftedCluster;

impl ComplexFunction for ShiftedCluster {
    type Complex = Complex<f64>;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        (z - Complex::new(0.20, 0.20))
            * (z - Complex::new(0.23, 0.22))
            * (z - Complex::new(-0.65, 0.40))
    }

    fn derivative(&self, z: Self::Complex) -> Self::Complex {
        let a = Complex::new(0.20, 0.20);
        let b = Complex::new(0.23, 0.22);
        let c = Complex::new(-0.65, 0.40);

        (z - b) * (z - c) + (z - a) * (z - c) + (z - a) * (z - b)
    }
}

#[test]
fn handles_clustered_roots_and_an_isolated_root() {
    let domain = Rect::new(-1.0, 0.6, -0.2, 0.8).unwrap();

    let result = find_singularities(ShiftedCluster, domain, SearchTarget::Zeros, config()).unwrap();

    assert_eq!(
        result.points.iter().map(|r| r.multiplicity).sum::<usize>(),
        3
    );

    assert_root_near(&result.points[..], Complex::new(-0.65, 0.40));

    assert!(
        result.points.iter().any(|r| {
            r.multiplicity >= 1 && (r.location - Complex::new(0.215, 0.21)).norm() < 0.08
        }),
        "expected a localised estimate or cluster around the close pair"
    );
}

#[derive(Debug, Clone, Copy)]
struct DoubleRoot;

impl ComplexFunction for DoubleRoot {
    type Complex = Complex<f64>;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        let dz = z - Complex::new(0.3, -0.2);
        dz * dz
    }

    fn derivative(&self, z: Self::Complex) -> Self::Complex {
        Complex::new(2.0, 0.0) * (z - Complex::new(0.3, -0.2))
    }
}

#[test]
fn reports_multiplicity_for_repeated_root() {
    let domain = Rect::new(-0.5, 1.0, -1.0, 0.5).unwrap();

    let result = find_singularities(DoubleRoot, domain, SearchTarget::Zeros, config()).unwrap();

    let total_multiplicity = result.points.iter().map(|r| r.multiplicity).sum::<usize>();

    assert_eq!(total_multiplicity, 2);
    assert_root_near(&result.points[..], Complex::new(0.3, -0.2));
}

#[derive(Debug, Clone, Copy)]
struct BoundaryRoot;

impl ComplexFunction for BoundaryRoot {
    type Complex = Complex<f64>;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        z - Complex::new(0.5, 0.25)
    }

    fn derivative(&self, _z: Self::Complex) -> Self::Complex {
        Complex::new(1.0, 0.0)
    }
}

#[test]
fn recovers_when_root_lies_on_uniform_split_line() {
    let domain = Rect::new(0.0, 1.0, 0.0, 1.0).unwrap();

    let result = find_singularities(BoundaryRoot, domain, SearchTarget::Zeros, config()).unwrap();

    assert_eq!(
        result.points.iter().map(|r| r.multiplicity).sum::<usize>(),
        1
    );
    assert_root_near(&result.points[..], Complex::new(0.5, 0.25));
}

#[derive(Debug, Clone, Copy)]
struct NoRoots;

impl ComplexFunction for NoRoots {
    type Complex = Complex<f64>;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        z * z + Complex::new(4.0, 0.0)
    }

    fn derivative(&self, z: Self::Complex) -> Self::Complex {
        Complex::new(2.0, 0.0) * z
    }
}

#[test]
fn returns_no_roots_when_region_contains_none() {
    let domain = Rect::new(-1.0, 1.0, -1.0, 1.0).unwrap();

    let result = find_singularities(NoRoots, domain, SearchTarget::Zeros, config()).unwrap();

    assert!(result.points.is_empty());
    assert_eq!(
        result
            .leaves
            .iter()
            .map(|l| l.data.root_count)
            .sum::<isize>(),
        0
    );
}

#[derive(Debug, Clone, Copy)]
struct DoublePole;

impl ComplexFunction for DoublePole {
    type Complex = Complex<f64>;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        let dz = z - Complex::new(0.25, -0.4);
        Complex::new(1.0, 0.0) / (dz * dz)
    }

    fn derivative(&self, z: Self::Complex) -> Self::Complex {
        let dz = z - Complex::new(0.25, -0.4);
        -Complex::new(2.0, 0.0) / (dz * dz * dz)
    }
}

#[test]
fn reports_multiplicity_for_double_pole() {
    let domain = Rect::new(-1.0, 1.0, -1.0, 0.5).unwrap();

    let result = find_poles(DoublePole, domain, config()).unwrap();

    assert_eq!(
        result.points.iter().map(|r| r.multiplicity).sum::<usize>(),
        2
    );
    assert_root_near(&result.points, Complex::new(0.25, -0.4));
}
