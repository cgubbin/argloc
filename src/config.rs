// src/config.rs

use num_traits::{Float, FromPrimitive};
use quad_rs::{ComplexScalar, IntegratorConfig};
use quadtree_core::QuadTreeConfig;

#[derive(Debug, Clone)]
pub struct ArgumentConfig<T: ComplexScalar> {
    pub quad_tree: QuadTreeConfig<T>,
    pub integrator: IntegratorConfig<T>,
    pub zero_tol: T,
    pub residual_tolerance: T,
    pub boundary_shift_fraction: T,
}

impl<T> ArgumentConfig<T>
where
    T: Float + FromPrimitive + ComplexScalar,
{
    pub fn new(score_tolerance: T) -> Self {
        Self {
            quad_tree: QuadTreeConfig::new(score_tolerance),
            integrator: IntegratorConfig::default(),
            zero_tol: T::from_f64(1e-12).unwrap(),
            residual_tolerance: T::from_f64(1e-8).unwrap(),
            boundary_shift_fraction: T::from_f64(0.1).unwrap(),
        }
    }

    pub fn with_quad_tree(mut self, quad_tree: QuadTreeConfig<T>) -> Self {
        self.quad_tree = quad_tree;
        self
    }

    pub fn with_integrator(mut self, integrator: IntegratorConfig<T>) -> Self {
        self.integrator = integrator;
        self
    }

    pub fn with_zero_tol(mut self, zero_tol: T) -> Self {
        self.zero_tol = zero_tol;
        self
    }

    pub fn with_residual_tolerance(mut self, residual_tolerance: T) -> Self {
        self.residual_tolerance = residual_tolerance;
        self
    }

    pub fn with_boundary_shift_fraction(mut self, boundary_shift_fraction: T) -> Self {
        self.boundary_shift_fraction = boundary_shift_fraction;
        self
    }
}
