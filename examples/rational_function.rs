use argloc::{ArgumentConfig, HolomorphicFunction, find_poles, find_zeros};
use num_complex::Complex;
use quadtree_core::Rect;

#[derive(Debug, Clone, Copy)]
struct RationalFunction;

impl HolomorphicFunction for RationalFunction {
    type Complex = Complex<f64>;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        let numerator = z - Complex::new(0.5, 0.25);
        let denominator = (z - Complex::new(-0.6, 0.0)) * (z - Complex::new(0.8, -0.3));
        numerator / denominator
    }

    fn derivative(&self, z: Self::Complex) -> Self::Complex {
        let a = Complex::new(0.5, 0.25);
        let p = Complex::new(-0.6, 0.0);
        let q = Complex::new(0.8, -0.3);

        let n = z - a;
        let d = (z - p) * (z - q);

        let dn = Complex::new(1.0, 0.0);
        let dd = (z - p) + (z - q);

        (dn * d - n * dd) / (d * d)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let domain = Rect::new(-1.0, 1.2, -0.8, 0.8)?;
    let config = ArgumentConfig::new(1e-3);

    let zeros = find_zeros(RationalFunction, domain, config.clone())?;
    let poles = find_poles(RationalFunction, domain, config)?;

    println!("Zeros:");
    for root in &zeros.roots {
        println!(
            "  z = {}, multiplicity = {}",
            root.location, root.multiplicity
        );
    }

    println!("Poles:");
    for pole in &poles.roots {
        println!(
            "  z = {}, multiplicity = {}",
            pole.location, pole.multiplicity
        );
    }

    Ok(())
}
