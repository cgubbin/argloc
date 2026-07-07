//! Quadtree oracle for argument-principle root counting.
//!
//! This module connects the mathematical argument-principle machinery to the
//! generic quadtree refinement engine.
//!
//! The quadtree engine asks an oracle to evaluate each rectangular cell. For
//! this crate, evaluating a cell means:
//!
//! 1. constructing a closed contour around the rectangle,
//! 2. integrating `f'(z) / f(z)` around that contour,
//! 3. converting the winding number into an integer root count,
//! 4. assigning a refinement score.
//!
//! Expensive contour moments are intentionally not computed in this oracle.
//! The first pass should only count roots and decide whether cells need further
//! subdivision. Moment-based localisation happens later, only on relevant
//! root-containing leaves.

use crate::{
    SearchTarget,
    argument::{ArgumentError, LogDerivativeError, compute_winding},
    cell::ArgumentCell,
    function::HolomorphicFunction,
};

use nalgebra::ComplexField;
use num_traits::{Float, FromPrimitive};
use quad_rs::{ComplexScalar, Contour, IntegrableFloat, IntegrationOutput, IntegratorConfig};
use quadtree_core::{EvaluationContext, geometry::Rect, oracle::QuadOracle};

/// Oracle that evaluates quadtree cells using the Argument Principle.
#[derive(Debug, Clone)]
pub struct ArgumentOracle<F, T: ComplexScalar> {
    function: F,
    config: IntegratorConfig<T>,
    residual_tolerance: T,
    zero_tol: T,
    search_target: SearchTarget,
}

impl<F, T> ArgumentOracle<F, T>
where
    T: ComplexScalar + Float + FromPrimitive,
{
    /// Creates a new argument-principle oracle.
    pub fn new(
        function: F,
        config: IntegratorConfig<T>,
        residual_tolerance: T,
        zero_tol: T,
        search_target: SearchTarget,
    ) -> Self {
        Self {
            function,
            config,
            residual_tolerance,
            zero_tol,
            search_target,
        }
    }

    fn score(&self, root_count: isize, residual: T, integration_error: T) -> T {
        if root_count == 0 {
            return T::zero();
        }

        residual.max(integration_error)
    }

    fn map_argument_error<C>(
        &self,
        err: ArgumentError<C>,
        scaled_bounds: Rect<T>,
        raw_bounds: Rect<T>,
        ctx: EvaluationContext<'_, T>,
    ) -> ArgumentError<C>
    where
        C: ComplexField<RealField = T> + Copy,
        T: ComplexScalar<Complex = C>,
    {
        match err {
            ArgumentError::IntegrandEvaluation(
                source @ LogDerivativeError::NearSingularContour { z: raw_z, .. },
            ) => {
                let p = quadtree_core::Point {
                    x: raw_z.real(),
                    y: raw_z.imaginary(),
                };

                match ctx.scaler.to_scaled(p) {
                    Ok(sp) => {
                        let scaled_z = T::complex(sp.x, sp.y);

                        ArgumentError::BoundarySingularity {
                            raw_z,
                            scaled_z,
                            raw_bounds,
                            scaled_bounds,
                            source,
                        }
                    }
                    Err(e) => ArgumentError::Scaling(e),
                }
            }
            other => other,
        }
    }
}

pub(crate) fn rectangle_contour<T: ComplexScalar>(bounds: Rect<T>) -> Contour<T> {
    let z00 = T::complex(bounds.x_min, bounds.y_min);
    let z10 = T::complex(bounds.x_max, bounds.y_min);
    let z11 = T::complex(bounds.x_max, bounds.y_max);
    let z01 = T::complex(bounds.x_min, bounds.y_max);

    Contour::piecewise_linear(vec![z00, z10, z11, z01, z00])
}

impl<F, T> QuadOracle<T> for ArgumentOracle<F, T>
where
    F: HolomorphicFunction,
    F::Complex: ComplexField<RealField = T> + Copy + IntegrationOutput<F::Complex, Float = T>,
    T: ComplexScalar<Complex = F::Complex> + Float + FromPrimitive + IntegrableFloat,
{
    type Data = ArgumentCell<T>;
    type Error = ArgumentError<F::Complex>;

    fn evaluate(
        &mut self,
        raw_bounds: Rect<T>,
        ctx: EvaluationContext<'_, T>,
    ) -> Result<Self::Data, Self::Error> {
        let contour = rectangle_contour(raw_bounds);

        let winding = compute_winding(
            &self.function,
            contour,
            self.config.clone(),
            self.zero_tol,
            self.search_target,
        )
        .map_err(|err| self.map_argument_error(err, ctx.scaler.scaled_domain(), raw_bounds, ctx))?;

        let score = if winding.residual <= self.residual_tolerance {
            self.score(
                winding.root_count,
                winding.residual,
                winding.integration_error,
            )
        } else {
            Float::max(winding.residual, winding.integration_error)
        };

        Ok(ArgumentCell::new(
            winding.root_count,
            winding.residual,
            winding.integration_error,
            score,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        argument::{ArgumentError, LogDerivativeError},
        function::HolomorphicFunction,
    };

    use approx::assert_relative_eq;
    use num_complex::Complex;
    use quad_rs::IntegratorConfig;
    use quadtree_core::{Scaler2D, geometry::Rect, oracle::QuadOracle};

    const TOL: f64 = 1e-8;

    #[derive(Debug, Clone, Copy)]
    struct Linear {
        root: Complex<f64>,
    }

    impl HolomorphicFunction for Linear {
        type Complex = Complex<f64>;

        fn value(&self, z: Self::Complex) -> Self::Complex {
            z - self.root
        }

        fn derivative(&self, _z: Self::Complex) -> Self::Complex {
            Complex::new(1.0, 0.0)
        }
    }

    fn config() -> IntegratorConfig<f64> {
        IntegratorConfig::default()
            .with_absolute_tolerance(1e-10)
            .with_relative_tolerance(1e-10)
    }

    fn oracle(root: Complex<f64>) -> ArgumentOracle<Linear, f64> {
        ArgumentOracle::new(
            Linear { root },
            config(),
            1e-10, // zero_tol
            1e-6,  // residual_tolerance
            SearchTarget::Zeros,
        )
    }

    #[test]
    fn rectangle_with_no_root_returns_zero_count() {
        let bounds = Rect::new(0.0, 1.0, 0.0, 1.0).unwrap();

        let mut oracle = oracle(Complex::new(2.0, 2.0));

        let scaler = Scaler2D::unit_square(bounds).unwrap();
        let ctx = EvaluationContext::new(&scaler);

        let cell = oracle.evaluate(bounds, ctx).unwrap();

        assert_eq!(cell.root_count, 0);
        assert_relative_eq!(cell.residual, 0.0, epsilon = TOL);
        assert_relative_eq!(cell.score, 0.0, epsilon = TOL);
    }

    #[test]
    fn rectangle_with_one_root_returns_count_one() {
        let bounds = Rect::new(0.0, 1.0, 0.0, 1.0).unwrap();

        let mut oracle = oracle(Complex::new(0.25, 0.75));

        let scaler = Scaler2D::unit_square(bounds).unwrap();
        let ctx = EvaluationContext::new(&scaler);

        let cell = oracle.evaluate(bounds, ctx).unwrap();

        assert_eq!(cell.root_count, 1);
        assert!(cell.residual < TOL);
        assert!(cell.integration_error >= 0.0);
    }

    #[test]
    fn rectangle_with_two_roots_returns_count_two() {
        #[derive(Debug, Clone, Copy)]
        struct Quadratic {
            a: Complex<f64>,
            b: Complex<f64>,
        }

        impl HolomorphicFunction for Quadratic {
            type Complex = Complex<f64>;

            fn value(&self, z: Self::Complex) -> Self::Complex {
                (z - self.a) * (z - self.b)
            }

            fn derivative(&self, z: Self::Complex) -> Self::Complex {
                (z - self.a) + (z - self.b)
            }
        }

        let bounds = Rect::new(0.0, 1.0, 0.0, 1.0).unwrap();

        let mut oracle = ArgumentOracle::new(
            Quadratic {
                a: Complex::new(0.25, 0.25),
                b: Complex::new(0.75, 0.75),
            },
            config(),
            1e-10,
            1e-6,
            SearchTarget::Zeros,
        );

        let scaler = Scaler2D::unit_square(bounds).unwrap();
        let ctx = EvaluationContext::new(&scaler);

        let cell = oracle.evaluate(bounds, ctx).unwrap();

        assert_eq!(cell.root_count, 2);
        assert!(cell.residual < TOL);
    }

    #[test]
    fn root_on_boundary_returns_boundary_singularity_error() {
        let bounds = Rect::new(0.0, 1.0, 0.0, 1.0).unwrap();

        let mut oracle = oracle(Complex::new(0.5, 0.0));

        let scaler = Scaler2D::unit_square(bounds).unwrap();
        let ctx = EvaluationContext::new(&scaler);

        let err = oracle.evaluate(bounds, ctx).unwrap_err();

        assert!(matches!(err, ArgumentError::BoundarySingularity { .. }));
    }

    #[test]
    fn root_on_corner_returns_boundary_singularity_error() {
        let bounds = Rect::new(0.0, 1.0, 0.0, 1.0).unwrap();

        let mut oracle = oracle(Complex::new(0.0, 0.0));

        let scaler = Scaler2D::unit_square(bounds).unwrap();
        let ctx = EvaluationContext::new(&scaler);

        let err = oracle.evaluate(bounds, ctx).unwrap_err();

        assert!(matches!(err, ArgumentError::BoundarySingularity { .. }));
    }

    #[test]
    fn large_residual_produces_nonzero_refinement_score() {
        let bounds = Rect::new(0.0, 1.0, 0.0, 1.0).unwrap();

        let mut oracle = ArgumentOracle::new(
            Linear {
                root: Complex::new(0.25, 0.75),
            },
            config(),
            1e-10,
            0.0, // deliberately strict residual tolerance
            SearchTarget::Zeros,
        );

        let scaler = Scaler2D::unit_square(bounds).unwrap();
        let ctx = EvaluationContext::new(&scaler);

        let cell = oracle.evaluate(bounds, ctx).unwrap();

        assert_eq!(cell.root_count, 1);
        assert!(cell.score >= cell.residual);
    }

    #[test]
    fn boundary_singularity_error_contains_raw_and_scaled_coordinates() {
        let raw_domain = Rect::new(10.0, 20.0, -5.0, 5.0).unwrap();

        let scaler = Scaler2D::unit_square(raw_domain.clone()).unwrap();
        let scaled_bounds = scaler.scaled_domain();
        let ctx = EvaluationContext { scaler: &scaler };

        // This root lies on the lower boundary of the raw rectangle.
        let raw_root = Complex::new(15.0, -5.0);

        let mut oracle = oracle(raw_root);

        let err = oracle.evaluate(raw_domain, ctx).unwrap_err();

        match err {
            ArgumentError::BoundarySingularity {
                raw_z,
                scaled_z,
                raw_bounds,
                scaled_bounds: err_scaled_bounds,
                ..
            } => {
                assert_relative_eq!(raw_z.re, 15.0, epsilon = TOL);
                assert_relative_eq!(raw_z.im, -5.0, epsilon = TOL);

                assert_relative_eq!(scaled_z.re, 0.5, epsilon = TOL);
                assert_relative_eq!(scaled_z.im, 0.0, epsilon = TOL);

                assert_eq!(raw_bounds, raw_domain);
                assert_eq!(err_scaled_bounds, scaled_bounds);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn oracle_evaluates_raw_contour_from_scaled_bounds() {
        let raw_domain = Rect::new(10.0, 20.0, -5.0, 5.0).unwrap();

        let scaler = Scaler2D::unit_square(raw_domain.clone()).unwrap();
        let ctx = EvaluationContext { scaler: &scaler };

        let raw_root = Complex::new(15.0, 0.0);

        let mut oracle = oracle(raw_root);

        let cell = oracle.evaluate(raw_domain, ctx).unwrap();

        assert_eq!(cell.root_count, 1);
        assert!(cell.residual < TOL);
    }
}
