//! # Argument Principle Root Finding
//!
//! `argument_principle` locates zeros of complex-valued holomorphic functions
//! using contour integration and adaptive quadtree refinement.
//!
//! The crate applies the Argument Principle to rectangular regions of the
//! complex plane. Each rectangle is surrounded by a contour, the logarithmic
//! derivative is integrated around that contour, and the resulting winding
//! number determines how many roots lie inside the region.
//!
//! Adaptive subdivision is provided by `quadtree_core`; contour integration is
//! provided by `quad_rs`.
//!
//! # Mathematical basis
//!
//! Let `f(z)` be holomorphic on and inside a closed contour `Γ`, with no zeros
//! on the contour. The Argument Principle states:
//!
//! ```text
//!              1
//! N = ---------------- ∮Γ f′(z) / f(z) dz
//!          2πi
//! ```
//!
//! where `N` is the number of enclosed zeros, counted with multiplicity.
//!
//! This crate evaluates that integral over rectangular cell boundaries. Cells
//! with non-zero root count or uncertain winding are refined adaptively.
//!
//! # Localisation
//!
//! Once refinement has completed, root locations are estimated using the first
//! logarithmic-derivative moment:
//!
//! ```text
//! S₁ = (1 / 2πi) ∮Γ z f′(z) / f(z) dz
//! ```
//!
//! If the contour encloses roots `z₁, …, zₙ`, counted with multiplicity, then:
//!
//! ```text
//! S₁ = z₁ + ... + zₙ
//! ```
//!
//! Therefore:
//!
//! ```text
//! S₁ / N
//! ```
//!
//! gives the multiplicity-weighted centroid of the enclosed roots.
//!
//! For a single-root cell this is a root estimate. For a multi-root cell this
//! is a cluster centroid, not an individual root location.
//!
//! # Boundary singularities
//!
//! A root on a cell boundary makes the logarithmic derivative singular on the
//! integration contour. This is common during adaptive subdivision because a
//! root may lie exactly on a split line.
//!
//! The crate treats this as a recoverable refinement event. The integrand
//! reports a near-singular contour point, the oracle converts that raw complex
//! coordinate into the quadtree coordinate system, and the subdivision policy
//! retries using a shifted split.
//!
//! This preserves the geometric partition while avoiding contour indentation,
//! which can otherwise make root-count bookkeeping ambiguous.
//!
//! # Basic usage
//!
//! ```no_run
//! use argloc::{find_singularities, SearchTarget, ArgumentConfig, Rect, HolomorphicFunction};
//! use num_complex::Complex;
//!
//! #[derive(Debug, Clone, Copy)]
//! struct Quadratic;
//!
//! impl HolomorphicFunction for Quadratic {
//!     type Complex = Complex<f64>;
//!
//!     fn value(&self, z: Self::Complex) -> Self::Complex {
//!         z * z - Complex::new(1.0, 0.0)
//!     }
//!
//!     fn derivative(&self, z: Self::Complex) -> Self::Complex {
//!         Complex::new(2.0, 0.0) * z
//!     }
//! }
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let domain = Rect::new(-2.0, 2.0, -1.5, 1.5)?;
//! let config = ArgumentConfig::new(1e-3);
//!
//! let result = find_singularities(Quadratic, domain, SearchTarget::Zeros, config)?;
//!
//! for root in result.roots {
//!     println!(
//!         "{:?}: z = {}, multiplicity = {}",
//!         root.kind,
//!         root.location,
//!         root.multiplicity
//!     );
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Main components
//!
//! - [`HolomorphicFunction`] defines the user-supplied function and derivative.
//! - [`ArgumentConfig`] controls quadtree refinement, integration tolerances,
//!   singularity thresholds, and boundary recovery.
//! - [`find_roots`] is the main public entrypoint.
//! - [`RootEstimate`] describes a root or root-cluster estimate.
//! - [`FindRootsError`] is the public error type for the high-level API.
//!
//! # Design philosophy
//!
//! The crate separates the algorithm into independent layers:
//!
//! - contour integration computes winding numbers and moments;
//! - the oracle turns rectangular cells into argument-principle data;
//! - the quadtree engine handles adaptive refinement;
//! - localisation interprets final cells as root estimates.
//!
//! This separation keeps the numerical complex-analysis code independent of
//! the adaptive refinement engine, while still allowing domain-specific error
//! recovery such as shifted subdivision near boundary singularities.
//!
//! # Limitations
//!
//! The current implementation focuses on holomorphic functions and zero
//! finding. Meromorphic functions, pole counting, and contour-indentation
//! semantics are intentionally not part of the first API.
//!
//! Multi-root cells are reported as cluster centroids. Individual reconstruction
//! from higher contour moments may be added later.
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
pub use function::HolomorphicFunction;
pub use localisation::{SingularPointEstimate, SingularPointEstimateKind};
pub use oracle::ArgumentOracle;
pub use output::{ArgumentLeaf, ArgumentResult};
pub use subdivision::ShiftSplitOnBoundary;

pub use quadtree_core::{MaxWeightedScorePolicy, QuadTreeConfig, Rect, run_with_policy};

pub use quad_rs::IntegratorConfig;

use nalgebra::ComplexField;
use num_traits::{Float, FromPrimitive};
use quad_rs::{ComplexScalar, IntegrableFloat, IntegrationOutput};
use quadtree_core::{QuadTree, QuadTreeError};

use crate::{cell::ArgumentCell, subdivision::SubdivisionError};

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

pub fn find_zeros<F, C, T>(
    function: F,
    domain: Rect<T>,
    config: ArgumentConfig<T>,
) -> Result<ArgumentResult<C>, FindRootsError<C>>
where
    F: HolomorphicFunction<Complex = C> + Clone,
    C: ComplexField<RealField = T> + Copy + IntegrationOutput<C, Float = T>,
    T: ComplexScalar<Complex = C> + Float + FromPrimitive + IntegrableFloat,
{
    find_singularities(function, domain, SearchTarget::Zeros, config)
}

pub fn find_poles<F, C, T>(
    function: F,
    domain: Rect<T>,
    config: ArgumentConfig<T>,
) -> Result<ArgumentResult<C>, FindRootsError<C>>
where
    F: HolomorphicFunction<Complex = C> + Clone,
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
    F: HolomorphicFunction<Complex = C> + Clone,
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
    let roots = crate::localisation::localise_from_cells(
        &function,
        &leaves[..],
        config.integrator,
        config.zero_tol,
        search_target,
    )?;

    Ok(ArgumentResult {
        roots,
        leaves,
        summary: result.summary,
        termination: result.termination,
    })
}
