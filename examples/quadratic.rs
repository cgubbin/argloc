use std::convert::Infallible;

use argloc::{ArgumentConfig, ComplexFunction, FunctionBundle, SearchTarget, find_singularities};
use num_complex::Complex;
use quad_rs::IntegratorConfig;
use quadtree_core::{QuadTreeConfig, Rect};

#[derive(Debug, Clone, Copy)]
struct Quadratic;

impl ComplexFunction for Quadratic {
    type Complex = Complex<f64>;
    type Error = Infallible;

    fn bundle(
        &self,
        z: Self::Complex,
    ) -> Result<argloc::FunctionBundle<Self::Complex>, Self::Error> {
        Ok(FunctionBundle::new(
            z * z - Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0) * z,
        ))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Rect::new(-2.0, 2.0, -1.5, 1.5)?;

    let tree = QuadTreeConfig::new(1e-3)
        .with_max_iter(250)
        .with_max_depth(12)
        .with_max_leaves(20_000)
        .with_target_window(1);

    let integrator = IntegratorConfig::default()
        .with_absolute_tolerance(1e-10)
        .with_relative_tolerance(1e-10);

    let config = ArgumentConfig::new(1e-3)
        .with_quad_tree(tree)
        .with_integrator(integrator)
        .with_zero_tol(1e-10)
        .with_residual_tolerance(1e-6)
        .with_boundary_shift_fraction(0.1);

    let result = find_singularities(Quadratic, domain, SearchTarget::Zeros, config)?;

    println!("Found {} root estimates", result.points.len());

    for root in &result.points {
        println!(
            "{:?}: z = {}, multiplicity = {}, enclosure = {}",
            root.kind, root.location, root.multiplicity, root.enclosure
        );
    }

    println!("leaves: {}", result.leaves.len());
    println!("termination: {:?}", result.termination);

    Ok(())
}
