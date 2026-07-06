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
use num_traits::{Float, FromPrimitive, Zero};
use quad_rs::ComplexScalar;

#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum LocalisationError {
    #[error("cannot localise roots when root count is zero")]
    ZeroRootCount,

    #[error("cannot localise individual root from a multi-root cluster with count {0}")]
    MultipleRoots(isize),

    #[error("negative root count {0}; poles are not supported by this localiser")]
    NegativeRootCount(isize),
}

/// A root or root-cluster estimate produced from contour moments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootEstimate<C> {
    /// Estimated location.
    ///
    /// For a single root this is the root estimate. For multiple roots this is
    /// the multiplicity-weighted cluster centroid.
    pub location: C,

    /// Number of enclosed roots counted with multiplicity.
    pub multiplicity: usize,

    /// Whether this estimate represents one root or a cluster centroid.
    pub kind: RootEstimateKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootEstimateKind {
    SingleRoot,
    ClusterCentroid,
}

/// Computes the multiplicity-weighted centroid of enclosed roots.
///
/// Returns `None` if `root_count <= 0`.
pub fn centroid<C>(root_count: isize, first_moment: C) -> Option<C>
where
    C: ComplexField,
    C::RealField: ComplexScalar<Complex = C> + Float + FromPrimitive + Zero,
{
    if root_count <= 0 {
        return None;
    }

    Some(
        first_moment
            / <<C as ComplexField>::RealField as ComplexScalar>::complex(
                <<C as ComplexField>::RealField as FromPrimitive>::from_isize(root_count).unwrap(),
                <<C as ComplexField>::RealField as Zero>::zero(),
            ),
    )
}

/// Builds a root or cluster estimate from the first moment.
pub fn estimate_from_first_moment<C>(
    root_count: isize,
    first_moment: C,
) -> Result<RootEstimate<C>, LocalisationError>
where
    C: ComplexField,
    C::RealField: ComplexScalar<Complex = C> + Float + FromPrimitive + Zero,
{
    if root_count < 0 {
        return Err(LocalisationError::NegativeRootCount(root_count));
    }

    if root_count == 0 {
        return Err(LocalisationError::ZeroRootCount);
    }

    let location = centroid(root_count, first_moment).expect("root_count already checked positive");

    let multiplicity = root_count as usize;

    let kind = if root_count == 1 {
        RootEstimateKind::SingleRoot
    } else {
        RootEstimateKind::ClusterCentroid
    };

    Ok(RootEstimate {
        location,
        multiplicity,
        kind,
    })
}

/// Builds a single-root estimate from the first moment.
///
/// This is stricter than [`estimate_from_first_moment`]: it rejects multi-root
/// contours instead of returning a cluster centroid.
pub fn single_root_from_first_moment<C>(
    root_count: isize,
    first_moment: C,
) -> Result<RootEstimate<C>, LocalisationError>
where
    C: ComplexField,
    C::RealField: ComplexScalar<Complex = C> + Float + FromPrimitive + Zero,
{
    if root_count == 1 {
        return estimate_from_first_moment(root_count, first_moment);
    }

    if root_count < 0 {
        return Err(LocalisationError::NegativeRootCount(root_count));
    }

    if root_count == 0 {
        return Err(LocalisationError::ZeroRootCount);
    }

    Err(LocalisationError::MultipleRoots(root_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex;

    const TOL: f64 = 1e-12;

    #[test]
    fn centroid_returns_none_for_zero_count() {
        let moment = Complex::new(1.0, 2.0);

        assert_eq!(centroid(0, moment), None);
    }

    #[test]
    fn centroid_returns_none_for_negative_count() {
        let moment = Complex::new(1.0, 2.0);

        assert_eq!(centroid(-1, moment), None);
    }

    #[test]
    fn centroid_of_single_root_is_first_moment() {
        let moment = Complex::new(0.25, -0.5);

        let c = centroid(1, moment).unwrap();

        assert_relative_eq!(c.re, moment.re, epsilon = TOL);
        assert_relative_eq!(c.im, moment.im, epsilon = TOL);
    }

    #[test]
    fn centroid_of_two_roots_is_mean_location() {
        let a = Complex::new(0.2, 0.1);
        let b = Complex::new(-0.4, 0.3);

        let first_moment = a + b;

        let c = centroid(2, first_moment).unwrap();
        let expected = (a + b) / Complex::new(2.0, 0.0);

        assert_relative_eq!(c.re, expected.re, epsilon = TOL);
        assert_relative_eq!(c.im, expected.im, epsilon = TOL);
    }

    #[test]
    fn centroid_counts_multiplicity() {
        let root = Complex::new(0.2, -0.1);
        let first_moment = Complex::new(3.0, 0.0) * root;

        let c = centroid(3, first_moment).unwrap();

        assert_relative_eq!(c.re, root.re, epsilon = TOL);
        assert_relative_eq!(c.im, root.im, epsilon = TOL);
    }

    #[test]
    fn estimate_from_first_moment_returns_single_root_estimate() {
        let root = Complex::new(0.3, -0.2);

        let estimate = estimate_from_first_moment(1, root).unwrap();

        assert_eq!(estimate.multiplicity, 1);
        assert_eq!(estimate.kind, RootEstimateKind::SingleRoot);
        assert_relative_eq!(estimate.location.re, root.re, epsilon = TOL);
        assert_relative_eq!(estimate.location.im, root.im, epsilon = TOL);
    }

    #[test]
    fn estimate_from_first_moment_returns_cluster_centroid_for_multiple_roots() {
        let a = Complex::new(0.2, 0.1);
        let b = Complex::new(-0.4, 0.3);

        let estimate = estimate_from_first_moment(2, a + b).unwrap();

        assert_eq!(estimate.multiplicity, 2);
        assert_eq!(estimate.kind, RootEstimateKind::ClusterCentroid);

        let expected = (a + b) / Complex::new(2.0, 0.0);

        assert_relative_eq!(estimate.location.re, expected.re, epsilon = TOL);
        assert_relative_eq!(estimate.location.im, expected.im, epsilon = TOL);
    }

    #[test]
    fn estimate_from_first_moment_rejects_zero_count() {
        let err = estimate_from_first_moment(0, Complex::<f64>::new(1.0, 0.0)).unwrap_err();

        assert_eq!(err, LocalisationError::ZeroRootCount);
    }

    #[test]
    fn estimate_from_first_moment_rejects_negative_count() {
        let err = estimate_from_first_moment(-1, Complex::<f64>::new(1.0, 0.0)).unwrap_err();

        assert_eq!(err, LocalisationError::NegativeRootCount(-1));
    }

    #[test]
    fn single_root_from_first_moment_accepts_one_root() {
        let root = Complex::new(0.3, 0.4);

        let estimate = single_root_from_first_moment(1, root).unwrap();

        assert_eq!(estimate.kind, RootEstimateKind::SingleRoot);
        assert_eq!(estimate.multiplicity, 1);
        assert_relative_eq!(estimate.location.re, root.re, epsilon = TOL);
        assert_relative_eq!(estimate.location.im, root.im, epsilon = TOL);
    }

    #[test]
    fn single_root_from_first_moment_rejects_multiple_roots() {
        let err = single_root_from_first_moment(2, Complex::<f64>::new(1.0, 0.0)).unwrap_err();

        assert_eq!(err, LocalisationError::MultipleRoots(2));
    }
}
