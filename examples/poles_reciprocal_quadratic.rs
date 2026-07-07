use argloc::{ArgumentConfig, ComplexFunction, find_poles};
use num_complex::Complex;
use quad_rs::IntegratorConfig;
use quadtree_core::{QuadTreeConfig, Rect};

#[derive(Debug, Clone, Copy)]
struct ReciprocalQuadratic;

impl ComplexFunction for ReciprocalQuadratic {
    type Complex = Complex<f64>;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        Complex::new(1.0, 0.0) / (z * z - Complex::new(5.0, 0.0))
    }

    fn derivative(&self, z: Self::Complex) -> Self::Complex {
        let denom = z * z - Complex::new(5.0, 0.0);
        -Complex::new(2.0, 0.0) * z / (denom * denom)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Rect::new(-3.0, 3.0, -1.0, 1.0)?;

    let tree = QuadTreeConfig::new(1e-3)
        .with_max_iter(300)
        .with_max_depth(12)
        .with_max_leaves(20_000)
        .with_target_window(1);

    let integrator = IntegratorConfig::default()
        .with_absolute_tolerance(1e-10)
        .with_relative_tolerance(1e-10);

    let config = ArgumentConfig::new(1e-3)
        .with_quad_tree(tree)
        .with_integrator(integrator);

    let result = find_poles(ReciprocalQuadratic, domain, config)?;

    for pole in &result.points {
        println!(
            "{:?}: z = {}, multiplicity = {}, enclosure = {}",
            pole.kind, pole.location, pole.multiplicity, pole.enclosure
        );
    }

    Ok(())
}
