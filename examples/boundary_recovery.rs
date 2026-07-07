use argloc::{ArgumentConfig, ComplexFunction, find_singularities};
use num_complex::Complex;
use quad_rs::IntegratorConfig;
use quadtree_core::{QuadTreeConfig, Rect};

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Rect::new(0.0, 1.0, 0.0, 1.0)?;

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

    let result = find_singularities(BoundaryRoot, domain, argloc::SearchTarget::Zeros, config)?;

    println!("Boundary-recovery example");
    println!("The root lies on the first uniform split line x = 0.5.");

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
