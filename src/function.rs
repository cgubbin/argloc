//! Complex function abstractions.
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

use std::{error::Error, fmt::Debug};

/// Value and first derivative of a complex function at one point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FunctionBundle<C> {
    value: C,
    derivative: C,
}

impl<C> FunctionBundle<C> {
    pub fn new(value: C, derivative: C) -> Self {
        Self { value, derivative }
    }

    pub fn value(&self) -> &C {
        &self.value
    }

    pub fn derivative(&self) -> &C {
        &self.derivative
    }

    pub fn into_parts(self) -> (C, C) {
        (self.value, self.derivative)
    }
}

/// A function over the complex plane.
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
pub trait ComplexFunction {
    type Complex: Copy;
    type Error: Debug + Error + Send + Sync + 'static + Clone;

    /// Evaluate the function and its first derivative at `z`.
    ///
    /// Implementations for which value and derivative are naturally computed
    /// together should override this operation directly.
    fn bundle(&self, z: Self::Complex) -> Result<FunctionBundle<Self::Complex>, Self::Error>;

    /// Evaluate the function at `z`.
    fn value(&self, z: Self::Complex) -> Result<Self::Complex, Self::Error> {
        Ok(*self.bundle(z)?.value())
    }

    /// Evaluate the first derivative at `z`.
    fn derivative(&self, z: Self::Complex) -> Result<Self::Complex, Self::Error> {
        Ok(*self.bundle(z)?.derivative())
    }
}
