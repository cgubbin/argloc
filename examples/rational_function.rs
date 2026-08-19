use std::convert::Infallible;

use argloc::{ArgumentConfig, ComplexFunction, FunctionBundle, find_poles, find_zeros};
use num_complex::Complex;
use quadtree_core::Rect;

#[derive(Debug, Clone, Copy)]
struct RationalFunction;

impl ComplexFunction for RationalFunction {
    type Complex = Complex<f64>;
    type Error = Infallible;

    fn bundle(
        &self,
        z: Self::Complex,
    ) -> Result<argloc::FunctionBundle<Self::Complex>, Self::Error> {
        let numerator = z - Complex::new(0.5, 0.25);
        let denominator = (z - Complex::new(-0.6, 0.0)) * (z - Complex::new(0.8, -0.3));

        let a = Complex::new(0.5, 0.25);
        let p = Complex::new(-0.6, 0.0);
        let q = Complex::new(0.8, -0.3);

        let n = z - a;
        let d = (z - p) * (z - q);

        let dn = Complex::new(1.0, 0.0);
        let dd = (z - p) + (z - q);

        Ok(FunctionBundle::new(
            numerator / denominator,
            (dn * d - n * dd) / (d * d),
        ))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Rect::new(-1.0, 1.2, -0.8, 0.8)?;
    let config = ArgumentConfig::new(1e-3);

    let zeros = find_zeros(RationalFunction, domain, config.clone())?;
    let poles = find_poles(RationalFunction, domain, config)?;

    println!("Zeros:");
    for root in &zeros.points {
        println!(
            "  z = {}, multiplicity = {}",
            root.location, root.multiplicity
        );
    }

    println!("Poles:");
    for pole in &poles.points {
        println!(
            "  z = {}, multiplicity = {}",
            pole.location, pole.multiplicity
        );
    }

    Ok(())
}
