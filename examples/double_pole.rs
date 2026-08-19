use std::convert::Infallible;

use argloc::{ArgumentConfig, ComplexFunction, FunctionBundle, find_poles};
use num_complex::Complex;
use quadtree_core::Rect;

#[derive(Debug, Clone, Copy)]
struct DoublePole;

impl ComplexFunction for DoublePole {
    type Complex = Complex<f64>;
    type Error = Infallible;

    fn bundle(
        &self,
        z: Self::Complex,
    ) -> Result<argloc::FunctionBundle<Self::Complex>, Self::Error> {
        let dz = z - Complex::new(0.25, -0.4);
        Ok(FunctionBundle::new(
            Complex::new(1.0, 0.0) / (dz * dz),
            -Complex::new(2.0, 0.0) / (dz * dz * dz),
        ))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Rect::new(-1.0, 1.0, -1.0, 0.5)?;

    let result = find_poles(DoublePole, domain, ArgumentConfig::new(1e-3))?;

    for pole in &result.points {
        println!(
            "{:?}: z = {}, multiplicity = {}, enclosure = {}",
            pole.kind, pole.location, pole.multiplicity, pole.enclosure
        );
    }

    Ok(())
}
