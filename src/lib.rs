//! # argloc
//!
//! `argloc` locates zeros and poles of complex-valued functions using the
//! Argument Principle, contour integration, and adaptive quadtree refinement.
//!
//! The crate searches rectangular regions of the complex plane. Each region is
//! surrounded by a contour, the logarithmic derivative is integrated around
//! that contour, and the resulting winding number determines how many target
//! singularities lie inside the region.
//!
//! Adaptive refinement is provided by `quadtree_core`; contour integration is
//! provided by `quad_rs`.
//!
//! # Mathematical basis
//!
//! For a meromorphic function `f(z)` and a closed contour `Γ` containing no
//! zeros or poles on the contour, the Argument Principle states:
//!
//! ```text
//!              1
//! N - P = ------------ ∮Γ f′(z) / f(z) dz
//!          2πi
//! ```
//!
//! where:
//!
//! - `N` is the number of zeros inside `Γ`;
//! - `P` is the number of poles inside `Γ`;
//! - both are counted with multiplicity.
//!
//! `argloc` applies this identity to rectangular cell boundaries. Cells with
//! non-zero target count, uncertain winding, or unresolved numerical error are
//! refined adaptively.
//!
//! # Zeros, poles, and API assumptions
//!
//! The Argument Principle computes a **signed** count, `N - P`. It does not
//! separately report `N` and `P`.
//!
//! This is a fundamental limitation. A region containing
//!
//! ```text
//! 3 zeros and 2 poles
//! ```
//!
//! has the same signed argument count as a region containing
//!
//! ```text
//! 1 zero and 0 poles.
//! ```
//!
//! Therefore a single logarithmic-derivative contour integral cannot, in
//! general, robustly separate zeros from poles in a mixed meromorphic function.
//!
//! ## `find_zeros`
//!
//! [`find_zeros`] locates zeros of a function in a search domain.
//!
//! It assumes that the supplied function has no poles in the search domain.
//! This is the usual holomorphic root-finding case, for example:
//!
//! ```text
//! f(z) = z^3 - 1
//! ```
//!
//! If poles are present in the search region, the signed argument count may
//! become negative. In that case the assumptions of [`find_zeros`] have been
//! violated and the solver returns an error rather than silently producing an
//! incorrect result.
//!
//! ## `find_poles`
//!
//! [`find_poles`] locates poles of a function in a search domain.
//!
//! Internally, this reverses the sign convention so that poles contribute
//! positive counts. It is suitable for functions such as:
//!
//! ```text
//! f(z) = 1 / (z^2 - 5)
//! ```
//!
//! where the poles are the objects of interest.
//!
//! If zeros are also present in the same search region, they subtract from the
//! pole count. The solver cannot generally distinguish this from a smaller
//! number of poles using the argument-principle integral alone.
//!
//! ## Mixed rational functions
//!
//! For a rational function
//!
//! ```text
//! f(z) = g(z) / h(z)
//! ```
//!
//! zeros and poles should be handled separately when possible:
//!
//! - use [`find_zeros`] on `g` to locate zeros;
//! - use [`find_zeros`] on `h`, or [`find_poles`] on `f`, to locate poles.
//!
//! This is the robust approach because it avoids asking the signed count
//! `N - P` to recover two unknown quantities.
//!
//! A future lower-level API may expose raw signed argument-count regions for
//! users who explicitly want to analyse `N - P` directly.
//!
//! # Localisation
//!
//! After refinement, target locations are estimated using the first
//! logarithmic-derivative moment:
//!
//! ```text
//! S₁ = (1 / 2πi) ∮Γ z f′(z) / f(z) dz
//! ```
//!
//! For zero searches this gives the sum of enclosed zeros. For pole searches
//! the sign convention is reversed, giving the sum of enclosed poles.
//!
//! If a contour encloses target points `z₁, …, zₙ`, counted with multiplicity,
//! then:
//!
//! ```text
//! S₁ = z₁ + ... + zₙ
//! ```
//!
//! and:
//!
//! ```text
//! S₁ / n
//! ```
//!
//! is the multiplicity-weighted centroid.
//!
//! For a single-target cell this is a point estimate. For a multi-target cell
//! this is a cluster centroid, not an individual root or pole location.
//!
//! # Boundary singularities
//!
//! A zero or pole on a cell boundary makes `f′(z) / f(z)` singular on the
//! integration contour. This commonly occurs during adaptive subdivision when a
//! target lies exactly on a split line.
//!
//! `argloc` treats this as a recoverable refinement event. The integrand reports
//! a near-singular contour point, the oracle maps that point into the quadtree
//! coordinate system, and the subdivision policy retries with a shifted split.
//!
//! This preserves the rectangular partition and avoids contour indentation,
//! whose inclusion/exclusion semantics can make root-count bookkeeping
//! ambiguous.
//!
//! # Basic usage
//!
//! ```no_run
//! use argloc::{find_zeros, ArgumentConfig, ComplexFunction};
//! use num_complex::Complex;
//! use quadtree_core::Rect;
//!
//! #[derive(Debug, Clone, Copy)]
//! struct Cubic;
//!
//! impl ComplexFunction for Cubic {
//!     type Complex = Complex<f64>;
//!
//!     fn value(&self, z: Self::Complex) -> Self::Complex {
//!         z * z * z - Complex::new(1.0, 0.0)
//!     }
//!
//!     fn derivative(&self, z: Self::Complex) -> Self::Complex {
//!         Complex::new(3.0, 0.0) * z * z
//!     }
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let domain = Rect::new(-1.5, 1.5, -1.5, 1.5)?;
//! let config = ArgumentConfig::new(1e-3);
//!
//! let result = find_zeros(Cubic, domain, config)?;
//!
//! for estimate in result.points {
//!     println!(
//!         "{:?}: z = {}, multiplicity = {}",
//!         estimate.kind,
//!         estimate.location,
//!         estimate.multiplicity
//!     );
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Finding poles
//!
//! ```no_run
//! use argloc::{find_poles, ArgumentConfig, ComplexFunction};
//! use num_complex::Complex;
//! use quadtree_core::Rect;
//!
//! #[derive(Debug, Clone, Copy)]
//! struct ReciprocalQuadratic;
//!
//! impl ComplexFunction for ReciprocalQuadratic {
//!     type Complex = Complex<f64>;
//!
//!     fn value(&self, z: Self::Complex) -> Self::Complex {
//!         Complex::new(1.0, 0.0) / (z * z - Complex::new(5.0, 0.0))
//!     }
//!
//!     fn derivative(&self, z: Self::Complex) -> Self::Complex {
//!         let d = z * z - Complex::new(5.0, 0.0);
//!         -Complex::new(2.0, 0.0) * z / (d * d)
//!     }
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let domain = Rect::new(-3.0, 3.0, -1.0, 1.0)?;
//! let config = ArgumentConfig::new(1e-3);
//!
//! let result = find_poles(ReciprocalQuadratic, domain, config)?;
//!
//! for pole in result.points {
//!     println!("pole at {}, multiplicity {}", pole.location, pole.multiplicity);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Main components
//!
//! - [`ComplexFunction`] defines the user-supplied complex function and
//!   derivative.
//! - [`ArgumentConfig`] controls quadtree refinement, integration tolerances,
//!   singularity thresholds, and boundary recovery.
//! - [`find_zeros`] locates zeros under the no-poles-in-domain assumption.
//! - [`find_poles`] locates poles under the corresponding pole-search
//!   convention.
//! - [`SingularPointEstimate`] describes a target estimate or cluster centroid.
//! - [`FindSingularitiesError`] is the public error type for the high-level API.
//!
//! # Design philosophy
//!
//! The crate separates the algorithm into independent layers:
//!
//! - contour integration computes winding numbers and moments;
//! - the oracle turns rectangular cells into argument-principle data;
//! - the quadtree engine handles adaptive refinement;
//! - localisation interprets final cells as target estimates.
//!
//! This keeps the complex-analysis code independent of the adaptive refinement
//! engine, while still allowing domain-specific recovery such as shifted
//! subdivision near boundary singularities.
//!
//! # Limitations
//!
//! Mixed zero/pole regions cannot be robustly separated from `f′/f` alone.
//! The high-level APIs therefore make explicit assumptions about the target
//! singularity type.
//!
//! Multi-target cells are reported as centroids. Individual reconstruction from
//! higher contour moments may be added later.
mod argument;
mod cell;
mod config;
mod error;
mod function;
mod localisation;
mod oracle;
mod output;
mod subdivision;

pub use argument::ArgumentError;
pub use config::ArgumentConfig;
pub use error::FindRootsError;
pub use function::ComplexFunction;
pub use localisation::{SingularPointEstimate, SingularPointEstimateKind};
pub use oracle::ArgumentOracle;
pub use output::{ArgumentLeaf, ArgumentResult};
pub use subdivision::ShiftSplitOnBoundary;

pub use quadtree_core::{MaxWeightedScorePolicy, QuadTreeConfig, Rect, run_with_policy};

pub use quad_rs::IntegratorConfig;

use nalgebra::ComplexField;
use num_traits::{Float, FromPrimitive};
use quad_rs::{ComplexScalar, IntegrableFloat, IntegrationOutput};

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub enum SearchTarget {
    Zeros,
    Poles,
}

impl SearchTarget {
    pub fn sign<T: num_traits::Float>(self) -> T {
        match self {
            SearchTarget::Zeros => T::one(),
            SearchTarget::Poles => -T::one(),
        }
    }

    pub fn sign_isize(self) -> isize {
        match self {
            SearchTarget::Zeros => 1,
            SearchTarget::Poles => -1,
        }
    }
}

/// Locate zeros of a holomorphic function.
///
/// This assumes that `function` has no poles in `domain`.
///
/// If poles are present, the argument-principle count may become negative and
/// the solver will report an error rather than trying to interpret the result
/// as a zero count.
pub fn find_zeros<F, C, T>(
    function: F,
    domain: Rect<T>,
    config: ArgumentConfig<T>,
) -> Result<ArgumentResult<C>, FindRootsError<C>>
where
    F: ComplexFunction<Complex = C> + Clone,
    C: ComplexField<RealField = T> + Copy + IntegrationOutput<C, Float = T>,
    T: ComplexScalar<Complex = C> + Float + FromPrimitive + IntegrableFloat,
{
    find_singularities(function, domain, SearchTarget::Zeros, config)
}

/// Locate poles of a meromorphic function.
///
/// This uses the Argument Principle with the sign convention reversed, so that
/// poles contribute positive counts.
///
/// If zeros are also present in `domain`, they subtract from the pole count.
/// In mixed zero/pole regions, use separate numerator/denominator functions
/// when possible.
pub fn find_poles<F, C, T>(
    function: F,
    domain: Rect<T>,
    config: ArgumentConfig<T>,
) -> Result<ArgumentResult<C>, FindRootsError<C>>
where
    F: ComplexFunction<Complex = C> + Clone,
    C: ComplexField<RealField = T> + Copy + IntegrationOutput<C, Float = T>,
    T: ComplexScalar<Complex = C> + Float + FromPrimitive + IntegrableFloat,
{
    find_singularities(function, domain, SearchTarget::Poles, config)
}

pub fn find_singularities<F, C, T>(
    function: F,
    domain: Rect<T>,
    search_target: SearchTarget,
    config: ArgumentConfig<T>,
) -> Result<ArgumentResult<C>, FindRootsError<C>>
where
    F: ComplexFunction<Complex = C> + Clone,
    C: ComplexField<RealField = T> + Copy + IntegrationOutput<C, Float = T>,
    T: ComplexScalar<Complex = C> + Float + FromPrimitive + IntegrableFloat,
{
    let oracle = ArgumentOracle::new(
        function.clone(),
        config.integrator.clone(),
        config.zero_tol,
        config.residual_tolerance,
        search_target,
    );

    let result = run_with_policy(
        domain,
        oracle,
        MaxWeightedScorePolicy,
        ShiftSplitOnBoundary {
            shift_fraction: config.boundary_shift_fraction,
        },
        config.quad_tree,
    )?;

    let leaves: Vec<ArgumentLeaf<T>> = result.iter().map(ArgumentLeaf::from_raw_leaf).collect();
    let points = crate::localisation::localise_from_cells(
        &function,
        &leaves[..],
        config.integrator,
        config.zero_tol,
        search_target,
    )?;

    Ok(ArgumentResult {
        points,
        leaves,
        summary: result.summary,
        termination: result.termination,
    })
}
