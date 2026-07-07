//! Root localisation from argument-principle moments.
//!
//! This module interprets contour moments as root-location information.
//!
//! The argument module computes raw mathematical quantities such as winding
//! numbers and logarithmic-derivative moments. This module turns those
//! quantities into root estimates.
//!
//! For a contour enclosing roots `z₁, …, zₙ`, counted with multiplicity,
//!
//! ```text
//! S₁ = (1 / 2πi) ∮ z f'(z)/f(z) dz = z₁ + ... + zₙ
//! ```
//!
//! Therefore `S₁ / n` gives the multiplicity-weighted centroid of the enclosed
//! roots.
//!
//! For a single enclosed root this centroid is a root estimate. For multiple
//! enclosed roots it is only a cluster summary and should not be interpreted as
//! an individual root location.

use nalgebra::ComplexField;
use num_traits::{Float, FromPrimitive};
use quad_rs::{ComplexScalar, IntegrableFloat, IntegrationOutput, IntegratorConfig};
use quadtree_core::Rect;

use crate::{
    ArgumentError, ArgumentLeaf, ComplexFunction, SearchTarget, argument::compute_moment,
    oracle::rectangle_contour,
};

/// A root or root-cluster estimate produced from contour moments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingularPointEstimate<C: ComplexField> {
    /// Estimated location.
    ///
    /// For a single root this is the root estimate. For multiple roots this is
    /// the multiplicity-weighted cluster centroid.
    pub location: C,

    /// Number of enclosed roots counted with multiplicity.
    pub multiplicity: usize,

    pub enclosure: Rect<C::RealField>,

    /// Whether this estimate represents one root or a cluster centroid.
    pub kind: SingularPointEstimateKind,

    pub target: SearchTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingularPointEstimateKind {
    SinglePoint,
    ClusterCentroid,
}

pub fn localise_from_cells<F, C>(
    function: &F,
    leaves: &[ArgumentLeaf<C::RealField>],
    integrator_config: IntegratorConfig<C::RealField>,
    zero_tol: C::RealField,
    target: SearchTarget,
) -> Result<Vec<SingularPointEstimate<C>>, ArgumentError<C>>
where
    F: ComplexFunction<Complex = C>,
    C: ComplexField + Copy + IntegrationOutput<C, Float = C::RealField>,
    C::RealField: Float + FromPrimitive + IntegrableFloat + ComplexScalar<Complex = C>,
{
    let mut roots = Vec::new();

    for leaf in leaves {
        if leaf.data.root_count <= 0 {
            continue;
        }

        let contour = rectangle_contour(leaf.bounds);

        let first_moment = compute_moment(
            function,
            contour,
            integrator_config.clone(),
            1,
            zero_tol,
            target,
        )?;

        let n = leaf.data.root_count;
        let n_complex = C::from_real(C::RealField::from_isize(n).unwrap());

        let location = first_moment.moment / n_complex;

        let kind = if n == 1 {
            SingularPointEstimateKind::SinglePoint
        } else {
            SingularPointEstimateKind::ClusterCentroid
        };

        roots.push(SingularPointEstimate {
            location,
            multiplicity: n as usize,
            enclosure: leaf.bounds,
            kind,
            target,
        });
    }

    Ok(roots)
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use approx::assert_relative_eq;
//     use num_complex::Complex;

//     const TOL: f64 = 1e-12;

//     #[test]
//     fn centroid_returns_none_for_zero_count() {
//         let moment = Complex::new(1.0, 2.0);

//         assert_eq!(centroid(0, moment), None);
//     }

//     #[test]
//     fn centroid_returns_none_for_negative_count() {
//         let moment = Complex::new(1.0, 2.0);

//         assert_eq!(centroid(-1, moment), None);
//     }

//     #[test]
//     fn centroid_of_single_root_is_first_moment() {
//         let moment = Complex::new(0.25, -0.5);

//         let c = centroid(1, moment).unwrap();

//         assert_relative_eq!(c.re, moment.re, epsilon = TOL);
//         assert_relative_eq!(c.im, moment.im, epsilon = TOL);
//     }

//     #[test]
//     fn centroid_of_two_roots_is_mean_location() {
//         let a = Complex::new(0.2, 0.1);
//         let b = Complex::new(-0.4, 0.3);

//         let first_moment = a + b;

//         let c = centroid(2, first_moment).unwrap();
//         let expected = (a + b) / Complex::new(2.0, 0.0);

//         assert_relative_eq!(c.re, expected.re, epsilon = TOL);
//         assert_relative_eq!(c.im, expected.im, epsilon = TOL);
//     }

//     #[test]
//     fn centroid_counts_multiplicity() {
//         let root = Complex::new(0.2, -0.1);
//         let first_moment = Complex::new(3.0, 0.0) * root;

//         let c = centroid(3, first_moment).unwrap();

//         assert_relative_eq!(c.re, root.re, epsilon = TOL);
//         assert_relative_eq!(c.im, root.im, epsilon = TOL);
//     }

//     #[test]
//     fn estimate_from_first_moment_returns_single_root_estimate() {
//         let root = Complex::new(0.3, -0.2);

//         let estimate = estimate_from_first_moment(1, root).unwrap();

//         assert_eq!(estimate.multiplicity, 1);
//         assert_eq!(estimate.kind, RootEstimateKind::SingleRoot);
//         assert_relative_eq!(estimate.location.re, root.re, epsilon = TOL);
//         assert_relative_eq!(estimate.location.im, root.im, epsilon = TOL);
//     }

//     #[test]
//     fn estimate_from_first_moment_returns_cluster_centroid_for_multiple_roots() {
//         let a = Complex::new(0.2, 0.1);
//         let b = Complex::new(-0.4, 0.3);

//         let estimate = estimate_from_first_moment(2, a + b).unwrap();

//         assert_eq!(estimate.multiplicity, 2);
//         assert_eq!(estimate.kind, RootEstimateKind::ClusterCentroid);

//         let expected = (a + b) / Complex::new(2.0, 0.0);

//         assert_relative_eq!(estimate.location.re, expected.re, epsilon = TOL);
//         assert_relative_eq!(estimate.location.im, expected.im, epsilon = TOL);
//     }

//     #[test]
//     fn estimate_from_first_moment_rejects_zero_count() {
//         let err = estimate_from_first_moment(0, Complex::<f64>::new(1.0, 0.0)).unwrap_err();

//         assert_eq!(err, LocalisationError::ZeroRootCount);
//     }

//     #[test]
//     fn estimate_from_first_moment_rejects_negative_count() {
//         let err = estimate_from_first_moment(-1, Complex::<f64>::new(1.0, 0.0)).unwrap_err();

//         assert_eq!(err, LocalisationError::NegativeRootCount(-1));
//     }

//     #[test]
//     fn single_root_from_first_moment_accepts_one_root() {
//         let root = Complex::new(0.3, 0.4);

//         let estimate = single_root_from_first_moment(1, root).unwrap();

//         assert_eq!(estimate.kind, RootEstimateKind::SingleRoot);
//         assert_eq!(estimate.multiplicity, 1);
//         assert_relative_eq!(estimate.location.re, root.re, epsilon = TOL);
//         assert_relative_eq!(estimate.location.im, root.im, epsilon = TOL);
//     }

//     #[test]
//     fn single_root_from_first_moment_rejects_multiple_roots() {
//         let err = single_root_from_first_moment(2, Complex::<f64>::new(1.0, 0.0)).unwrap_err();

//         assert_eq!(err, LocalisationError::MultipleRoots(2));
//     }
// }
