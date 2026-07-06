//! Quadtree cell payloads for argument-principle refinement.
//!
//! This module defines the data stored on each quadtree leaf during the
//! root-counting phase.
//!
//! The quadtree crate is generic over cell data. For this crate, each cell
//! stores the result of applying the Argument Principle to the rectangular
//! contour surrounding that cell.
//!
//! A cell records:
//!
//! - the estimated number of enclosed roots,
//! - the residual from the nearest integer winding number,
//! - the numerical integration error,
//! - a scalar refinement score.
//!
//! The refinement score is what the generic quadtree engine uses to decide
//! convergence and prioritise subdivision. The precise scoring rule can evolve,
//! but the intended meaning is:
//!
//! ```text
//! score = 0      => cell is resolved
//! score > 0      => cell should remain eligible for refinement
//! ```

use num_traits::Float;
use quadtree_core::CellScore;

/// Data stored on a quadtree leaf during argument-principle refinement.
///
/// `ArgumentCell` is deliberately lightweight. It stores only the quantities
/// needed during the first pass of the algorithm: root counting and refinement
/// control.
///
/// More expensive information, such as contour moments, is not stored here by
/// default. Moments are better computed in a later localisation pass, and only
/// for cells that actually contain roots.
pub struct ArgumentCell<F> {
    /// Estimated number of roots enclosed by the cell contour.
    ///
    /// For holomorphic functions this is the number of zeros counted with
    /// multiplicity. Negative values are reserved for future meromorphic
    /// support, where poles may contribute to the argument-principle count.
    pub root_count: isize,

    /// Distance between the computed winding number and the nearest integer.
    ///
    /// A small residual indicates that the contour integral is numerically
    /// consistent with an integer root count.
    pub residual: F,

    /// Error estimate reported by the contour integration backend.
    pub integration_error: F,

    /// Scalar refinement score consumed by the quadtree policy.
    ///
    /// A score of zero means the cell is considered resolved.
    pub score: F,
}

impl<F> ArgumentCell<F>
where
    F: Float,
{
    /// Creates a new argument-principle cell payload.
    pub fn new(root_count: isize, residual: F, integration_error: F, score: F) -> Self {
        Self {
            root_count,
            residual,
            integration_error,
            score,
        }
    }

    /// Returns true if the cell contains at least one root.
    pub fn contains_roots(&self) -> bool {
        self.root_count > 0
    }

    /// Returns true if the cell is considered resolved by the current score.
    pub fn is_resolved(&self) -> bool {
        self.score <= F::zero()
    }
}

impl<F: Copy> CellScore<F> for ArgumentCell<F> {
    fn score(&self) -> F {
        self.score
    }
}
