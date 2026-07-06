//! Holomorphic function abstractions.
//!
//! This module defines the mathematical interface required by the argument
//! principle.
//!
//! The crate assumes the user supplies a complex-valued function together with
//! its first derivative. These are used to evaluate the logarithmic derivative
//!
//! ```text
//! f'(z) / f(z)
//! ```
//!
//! appearing in the Argument Principle.
//!
//! The derivative is kept explicit rather than approximated numerically,
//! improving both robustness and numerical accuracy of the contour integrals.
//!

/// A holomorphic function over the complex plane.
///
/// Implementations provide the complex function together with its first
/// derivative. These are used to evaluate contour integrals of the logarithmic
/// derivative
///
/// ```text
/// f'(z)
/// -----
/// f(z)
/// ```
///
/// required by the Argument Principle.
///
/// Functions are assumed to be analytic everywhere inside the search domain,
/// except at isolated poles if these are intentionally being analysed.
pub trait HolomorphicFunction {
    type Complex;
    /// Evaluate the function at `z`.
    fn value(&self, z: Self::Complex) -> Self::Complex;

    /// Evaluate the first derivative at `z`.
    fn derivative(&self, z: Self::Complex) -> Self::Complex;
}
