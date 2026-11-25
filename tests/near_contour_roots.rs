//! Roots that lie just off a bisection line must still be found.
//!
//! With a root 1e-12 from a cell edge and |f'| ~ 1e3 there, |f| on the contour
//! stays above `zero_tol`, so the contour does not count as singular, but the
//! adaptive quadrature cannot resolve the peak in `f'/f` without refining
//! below its minimum piece width.

use std::convert::Infallible;

use argloc::{ArgumentConfig, ComplexFunction, FunctionBundle, find_zeros};
use num_complex::Complex;
use quadtree_core::{QuadTreeConfig, Rect};

const SCALE: f64 = 1e3;
const OFFSET: f64 = 1e-12;

/// `SCALE * (z - r_1) * ... * (z - r_n)`.
#[derive(Debug, Clone)]
struct ScaledPolynomial {
    roots: Vec<Complex<f64>>,
}

impl ComplexFunction for ScaledPolynomial {
    type Complex = Complex<f64>;
    type Error = Infallible;

    fn bundle(&self, z: Complex<f64>) -> Result<FunctionBundle<Complex<f64>>, Self::Error> {
        let factors: Vec<_> = self.roots.iter().map(|r| z - r).collect();
        let value = factors.iter().product::<Complex<f64>>() * SCALE;
        let derivative = (0..factors.len())
            .map(|i| {
                factors
                    .iter()
                    .enumerate()
                    .filter(|&(j, _)| j != i)
                    .map(|(_, f)| f)
                    .product::<Complex<f64>>()
            })
            .sum::<Complex<f64>>()
            * SCALE;
        Ok(FunctionBundle::new(value, derivative))
    }
}

fn config() -> ArgumentConfig<f64> {
    let tree = QuadTreeConfig::new(1e-3)
        .with_max_iter(250)
        .with_max_depth(12)
        .with_max_leaves(20_000);

    // Default-like tolerances: zero 1e-12, residual 1e-8.
    ArgumentConfig::new(1e-3)
        .with_quad_tree(tree)
        .with_zero_tol(1e-12)
        .with_residual_tolerance(1e-8)
}

fn unit_square() -> Rect<f64> {
    Rect::new(0.0, 1.0, 0.0, 1.0).unwrap()
}

fn assert_finds(roots: Vec<Complex<f64>>) {
    let result = find_zeros(
        ScaledPolynomial {
            roots: roots.clone(),
        },
        unit_square(),
        config(),
    )
    .unwrap_or_else(|err| panic!("search failed: {err}"));

    let total: usize = result.points.iter().map(|p| p.multiplicity).sum();
    assert_eq!(total, roots.len(), "points: {:?}", result.points);

    for expected in roots {
        assert!(
            result
                .points
                .iter()
                .any(|p| (p.location - expected).norm() < 1e-6),
            "no root near {expected}; points: {:?}",
            result.points
        );
    }
}

#[test]
fn root_just_off_first_bisection_line_is_found() {
    assert_finds(vec![
        Complex::new(0.5 + OFFSET, 0.3),
        Complex::new(0.2, 0.7),
    ]);
}

#[test]
fn root_just_off_second_level_bisection_line_is_found() {
    assert_finds(vec![
        Complex::new(0.25 + OFFSET, 0.1),
        Complex::new(0.8, 0.7),
    ]);
}

#[test]
fn root_just_off_the_shifted_split_too_reports_a_boundary_singularity() {
    // The first root forces a shifted split at x = 0.4, y = 0.6; the second
    // lies just off that shifted line, so the single retry fails as well.
    let function = ScaledPolynomial {
        roots: vec![
            Complex::new(0.5 + OFFSET, 0.3),
            Complex::new(0.4 + OFFSET, 0.8),
        ],
    };

    let err = find_zeros(function, unit_square(), config())
        .expect_err("expected the retried subdivision to fail too");

    let message = err.to_string();
    assert!(
        message.contains("boundary singularity"),
        "unexpected error: {message}"
    );
}
