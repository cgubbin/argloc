use argloc::{ArgumentConfig, ComplexFunction, find_poles};
use num_complex::Complex;
use quadtree_core::Rect;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Rect::new(-1.0, 1.0, -1.0, 0.5)?;

    let result = find_poles(DoublePole, domain, ArgumentConfig::new(1e-3))?;

    for pole in &result.roots {
        println!(
            "{:?}: z = {}, multiplicity = {}, enclosure = {}",
            pole.kind, pole.location, pole.multiplicity, pole.enclosure
        );
    }

    Ok(())
}
