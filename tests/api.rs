use std::convert::Infallible;

use argloc::{
    ArgumentConfig, ComplexFunction, FunctionBundle, SearchTarget, find_poles, find_singularities,
};
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
    type Error = Infallible;

    fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error> {
        Ok(FunctionBundle::new(
            z * z - Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0) * z,
        ))
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
    type Error = Infallible;

    fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error> {
        Ok(FunctionBundle::new(
            z * z * z - Complex::new(1.0, 0.0),
            Complex::new(3.0, 0.0) * z * z,
        ))
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
    type Error = Infallible;

    fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error> {
        let a = Complex::new(0.20, 0.20);
        let b = Complex::new(0.23, 0.22);
        let c = Complex::new(-0.65, 0.40);
        Ok(FunctionBundle::new(
            (z - Complex::new(0.20, 0.20))
                * (z - Complex::new(0.23, 0.22))
                * (z - Complex::new(-0.65, 0.40)),
            (z - b) * (z - c) + (z - a) * (z - c) + (z - a) * (z - b),
        ))
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
    type Error = Infallible;

    fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error> {
        let dz = z - Complex::new(0.3, -0.2);
        Ok(FunctionBundle::new(
            dz * dz,
            Complex::new(2.0, 0.0) * (z - Complex::new(0.3, -0.2)),
        ))
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
    type Error = Infallible;

    fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error> {
        Ok(FunctionBundle::new(
            z - Complex::new(0.5, 0.25),
            Complex::new(1.0, 0.0),
        ))
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
    type Error = Infallible;

    fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error> {
        Ok(FunctionBundle::new(
            z * z + Complex::new(4.0, 0.0),
            Complex::new(2.0, 0.0) * z,
        ))
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
    type Error = Infallible;

    fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error> {
        let dz = z - Complex::new(0.25, -0.4);
        Ok(FunctionBundle::new(
            Complex::new(1.0, 0.0) / (dz * dz),
            -Complex::new(2.0, 0.0) / (dz * dz * dz),
        ))
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

/// `z^2 - 1` scaled far below the residual tolerance but far above the zero
/// tolerance: the contour is not near-singular, so both roots are found.
#[derive(Debug, Clone, Copy)]
struct SmallQuadratic;

impl ComplexFunction for SmallQuadratic {
    type Complex = Complex<f64>;
    type Error = Infallible;

    fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error> {
        let scale = 1e-9;
        Ok(FunctionBundle::new(
            (z * z - Complex::new(1.0, 0.0)) * scale,
            Complex::new(2.0 * scale, 0.0) * z,
        ))
    }
}

#[test]
fn zero_tolerance_alone_decides_near_singular_contours() {
    let config = config().with_zero_tol(1e-14).with_residual_tolerance(1e-6);
    let domain = Rect::new(-2.0, 2.0, -0.7, 0.9).unwrap();
    let result = find_singularities(SmallQuadratic, domain, SearchTarget::Zeros, config)
        .expect("a function far above the zero tolerance is not near-singular");
    assert_root_near(&result.points[..], Complex::new(1.0, 0.0));
    assert_root_near(&result.points[..], Complex::new(-1.0, 0.0));
}

/// `z − 10`, counting its evaluations.
#[derive(Debug, Clone)]
struct Counted {
    evaluations: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl ComplexFunction for Counted {
    type Complex = Complex<f64>;
    type Error = Infallible;

    fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error> {
        self.evaluations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(FunctionBundle::new(
            z - Complex::new(10.0, 0.0),
            Complex::new(1.0, 0.0),
        ))
    }
}

#[test]
fn an_empty_domain_is_resolved_by_its_own_contour() {
    use std::sync::atomic::Ordering::Relaxed;
    let domain = Rect::new(-2.0, 2.0, -1.5, 1.5).unwrap();
    // The default convergence window, which would otherwise split the domain
    // once per step.
    let config = ArgumentConfig::new(1e-3);
    let function = Counted {
        evaluations: Default::default(),
    };
    let result = find_singularities(
        function.clone(),
        domain,
        SearchTarget::Zeros,
        config.clone(),
    )
    .unwrap();
    assert!(result.points.is_empty());
    assert_eq!(result.leaves.len(), 1);
    assert_eq!(result.leaves[0].bounds, domain);
    assert_eq!(result.leaves[0].data.root_count, 0);
    assert_eq!(result.termination, argloc::Termination::Converged);

    // One contour: as many evaluations as counting the domain alone.
    let alone = Counted {
        evaluations: Default::default(),
    };
    let mut oracle = argloc::ArgumentOracle::new(
        alone.clone(),
        config.integrator.clone(),
        config.residual_tolerance,
        config.zero_tol,
        SearchTarget::Zeros,
    );
    let scaler = quadtree_core::Scaler2D::unit_square(domain).unwrap();
    quadtree_core::QuadOracle::evaluate(
        &mut oracle,
        domain,
        quadtree_core::EvaluationContext::new(&scaler),
    )
    .unwrap();
    assert_eq!(
        function.evaluations.load(Relaxed),
        alone.evaluations.load(Relaxed)
    );
}

#[test]
fn a_domain_with_a_root_is_still_refined_and_localised() {
    let domain = Rect::new(8.0, 12.5, -1.5, 1.5).unwrap();
    let function = Counted {
        evaluations: Default::default(),
    };
    let result = find_singularities(
        function,
        domain,
        SearchTarget::Zeros,
        ArgumentConfig::new(1e-3),
    )
    .unwrap();
    assert_eq!(
        result.points.iter().map(|r| r.multiplicity).sum::<usize>(),
        1
    );
    assert_root_near(&result.points[..], Complex::new(10.0, 0.0));
}

/// `x_min + 1 · (x_max − x_min)` rounds one ulp above `x_max` for this domain;
/// the reported leaves and enclosures still lie inside it, with boundary edges
/// equal to the domain's.
#[test]
fn leaves_and_enclosures_lie_inside_an_outward_rounding_domain() {
    let (x_min, x_max) = (-0.14175649807409718, -0.06175649807409717);
    assert!(x_min + 1.0 * (x_max - x_min) > x_max);
    let domain = Rect::new(x_min, x_max, -0.04, 0.04).unwrap();

    #[derive(Debug, Clone, Copy)]
    struct NearEdge;
    impl ComplexFunction for NearEdge {
        type Complex = Complex<f64>;
        type Error = Infallible;
        fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error> {
            let root = Complex::new(-0.0701, 0.013);
            Ok(FunctionBundle::new(z - root, Complex::new(1.0, 0.0)))
        }
    }

    let result = find_singularities(NearEdge, domain, SearchTarget::Zeros, config()).unwrap();
    let inside = |r: &Rect<f64>| {
        domain.x_min <= r.x_min
            && r.x_max <= domain.x_max
            && domain.y_min <= r.y_min
            && r.y_max <= domain.y_max
    };
    assert!(result.leaves.iter().all(|leaf| inside(&leaf.bounds)));
    assert!(result.points().iter().all(|point| inside(&point.enclosure)));
    assert!(
        result
            .leaves
            .iter()
            .any(|leaf| leaf.bounds.x_max == domain.x_max)
    );
    assert_root_near(result.points(), Complex::new(-0.0701, 0.013));
}
