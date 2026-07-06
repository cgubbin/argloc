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
    argument::{ArgumentError, compute_winding},
    cell::ArgumentCell,
    function::HolomorphicFunction,
};

use nalgebra::ComplexField;
use num_traits::{Float, FromPrimitive};
use quad_rs::{ComplexScalar, Contour, IntegrableFloat, IntegrationOutput, IntegratorConfig};
use quadtree_core::{geometry::Rect, oracle::QuadOracle};

/// Oracle that evaluates quadtree cells using the Argument Principle.
#[derive(Debug, Clone)]
pub struct ArgumentOracle<F, T: ComplexScalar> {
    function: F,
    config: IntegratorConfig<T>,
    residual_tolerance: T,
    zero_tol: T,
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
    ) -> Self {
        Self {
            function,
            config,
            residual_tolerance,
            zero_tol,
        }
    }

    fn rectangle_contour(bounds: Rect<T>) -> Contour<T> {
        let z00 = T::complex(bounds.x_min, bounds.y_min);
        let z10 = T::complex(bounds.x_max, bounds.y_min);
        let z11 = T::complex(bounds.x_max, bounds.y_max);
        let z01 = T::complex(bounds.x_min, bounds.y_max);

        Contour::piecewise_linear(vec![z00, z10, z11, z01, z00])
    }

    fn score(&self, root_count: isize, residual: T, integration_error: T) -> T {
        if root_count == 0 {
            return T::zero();
        }

        residual.max(integration_error)
    }
}

impl<F, T> QuadOracle<T> for ArgumentOracle<F, T>
where
    F: HolomorphicFunction,
    F::Complex: ComplexField<RealField = T> + Copy + IntegrationOutput<F::Complex, Float = T>,
    T: ComplexScalar<Complex = F::Complex> + Float + FromPrimitive + IntegrableFloat,
{
    type Data = ArgumentCell<T>;
    type Error = ArgumentError<F::Complex>;

    fn evaluate(&mut self, bounds: Rect<T>) -> Result<Self::Data, Self::Error> {
        let contour = Self::rectangle_contour(bounds);

        let winding = compute_winding(&self.function, contour, self.config.clone(), self.zero_tol)?;

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
