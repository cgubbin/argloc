//! Argument Principle evaluation.
//!
//! This module implements the mathematical core of the crate.
//!
//! Given a contour and a holomorphic function, it evaluates contour integrals
//! derived from the Argument Principle.
//!
//! These integrals determine
//!
//! - the number of enclosed roots,
//! - contour moments,
//! - numerical residuals,
//! - quantities used during root localisation.
//!
//! The module contains no adaptive refinement logic; it simply evaluates the
//! required contour integrals. Higher-level components decide how these results
//! should be used.

use nalgebra::ComplexField;
use num_traits::{FromPrimitive, ToPrimitive};
use quad_rs::{
    ComplexScalar, Contour, FallibleIntegrable, IntegrableFloat, IntegrationOutput,
    IntegratorConfig, IntegratorError, integrate_complex_fallible,
};
use quadtree_core::{Rect, ScalerError};

use crate::{SearchTarget, function::ComplexFunction};

#[derive(thiserror::Error, Debug, Clone)]
pub enum ArgumentError<C: ComplexField> {
    #[error("error in integrand evaluation: {0}")]
    IntegrandEvaluation(#[from] LogDerivativeError<C>),

    #[error("contour integration failed: {0}")]
    Integration(MinimalIntegratorError<C>),

    #[error("computed winding {winding:?} is not close to an integer; residual={residual:?}")]
    NonIntegerWinding { winding: C, residual: C::RealField },

    #[error("failure in internal scaling: {0}")]
    Scaling(#[from] ScalerError<C::RealField>),

    #[error("boundary singularity at raw {raw_z:?}, scaled {scaled_z:?}")]
    BoundarySingularity {
        raw_z: C,
        scaled_z: C,
        raw_bounds: Rect<C::RealField>,
        scaled_bounds: Rect<C::RealField>,
        source: LogDerivativeError<C>,
    },
}

impl<C: ComplexField> From<IntegratorError<C, LogDerivativeError<C>>> for ArgumentError<C> {
    fn from(value: IntegratorError<C, LogDerivativeError<C>>) -> Self {
        match value {
            IntegratorError::User(error) => Self::IntegrandEvaluation(error),
            IntegratorError::ExceededMaxFunctionEvaluations => {
                Self::Integration(MinimalIntegratorError::ExceededMaxFunctionEvaluations)
            }
            IntegratorError::NonFiniteErrorEstimate => {
                Self::Integration(MinimalIntegratorError::NonFiniteErrorEstimate)
            }
            IntegratorError::NoSegments => Self::Integration(MinimalIntegratorError::NoSegments),
            IntegratorError::EmptySegment => {
                Self::Integration(MinimalIntegratorError::EmptySegment)
            }
            IntegratorError::NonFiniteIntegrand { point } => {
                Self::Integration(MinimalIntegratorError::NonFiniteIntegrand { point })
            }
            IntegratorError::PossibleSingularity { singularity } => {
                Self::Integration(MinimalIntegratorError::PossibleSingularity { singularity })
            }
            IntegratorError::PieceTooSmall => {
                Self::Integration(MinimalIntegratorError::PieceTooSmall)
            }
        }
    }
}

#[derive(thiserror::Error, Debug, PartialEq, Clone)]
/// Error type for integrator
pub enum MinimalIntegratorError<T> {
    #[error("exceeded the maximum number of function evaluations before convergence")]
    ExceededMaxFunctionEvaluations,
    #[error("non-finite error estimate")]
    NonFiniteErrorEstimate,
    #[error("no segments found")]
    NoSegments,
    #[error("empty integration segment")]
    EmptySegment,
    #[error("non-finite integrand at {point:?}")]
    NonFiniteIntegrand { point: T },
    #[error("possible singularity at {singularity:?}")]
    PossibleSingularity { singularity: T },
    #[error("refined contour piece smaller than minimum")]
    PieceTooSmall,
}

#[derive(thiserror::Error, Debug, Clone)]
pub enum LogDerivativeError<C: ComplexField> {
    #[error("function value was non-finite at {z:?}")]
    NonFiniteFunctionValue { z: C },

    #[error("function derivative was non-finite at {z:?}")]
    NonFiniteDerivativeValue { z: C },

    #[error("function value was too close to zero on the contour at {z:?}: |f(z)|={norm:?}")]
    NearSingularContour { z: C, norm: C::RealField },

    #[error("logarithmic derivative was non-finite at {z:?}")]
    NonFiniteLogDerivative { z: C },
}

/// Quantities computed from the Argument Principle.
///
/// The contained values are obtained from contour integrals of the logarithmic
/// derivative.
///
/// Besides the root count, the contour moments may be used to estimate the
/// centroid of enclosed roots or initialise subsequent localisation
/// algorithms.
#[allow(dead_code)]
pub struct ArgumentData<C: ComplexField> {
    pub winding: WindingData<C>,
    pub first_moment: MomentData<C>,
}

#[allow(dead_code)]
pub struct WindingData<C: ComplexField> {
    pub winding: C,
    pub root_count: isize,
    pub residual: C::RealField,
    pub integration_error: C::RealField,
}

#[allow(dead_code)]
pub struct MomentData<C: ComplexField> {
    pub power: usize,
    pub moment: C,
    pub integration_error: C::RealField,
}

/// Integrand corresponding to the logarithmic derivative.
///
/// The integrand is
///
/// ```text
/// f'(z)
/// -----.
/// f(z)
/// ```
///
/// Integrating this quantity around a closed contour yields the Argument
/// Principle integral from which the enclosed root count is obtained.
pub struct LogDerivative<'a, F, T> {
    pub function: &'a F,
    pub zero_tol: T,
}

impl<'a, F> FallibleIntegrable for LogDerivative<'a, F, <F::Complex as ComplexField>::RealField>
where
    F: ComplexFunction,
    F::Complex: ComplexField + Copy,
    <F::Complex as ComplexField>::RealField: IntegrableFloat,
{
    type Float = <F::Complex as ComplexField>::RealField;
    type Input = F::Complex;
    type Output = F::Complex;
    type Error = LogDerivativeError<F::Complex>;

    fn fallible_integrand(&self, z: &Self::Input) -> Result<Self::Output, Self::Error> {
        checked_log_derivative(self.function, *z, self.zero_tol)
    }
}

/// Integrand for contour moments of the logarithmic derivative.
///
/// The integrand is
///
/// ```text
///      k  f'(z)
///     z -------.
///        f(z)
/// ```
///
/// The zeroth moment counts enclosed roots.
///
/// Higher moments contain geometric information about the enclosed roots and
/// can be used for centroid estimation or subsequent localisation methods.
pub struct LogDerivativeMoment<'a, F, T> {
    pub function: &'a F,
    pub power: usize,
    pub zero_tol: T,
}

use num_traits::{One, Zero};

impl<'a, F> FallibleIntegrable
    for LogDerivativeMoment<'a, F, <F::Complex as ComplexField>::RealField>
where
    F: ComplexFunction,
    F::Complex: ComplexField + Copy,
    <F::Complex as ComplexField>::RealField:
        IntegrableFloat + ComplexScalar<Complex = F::Complex> + One + Zero,
{
    type Float = <F::Complex as ComplexField>::RealField;
    type Input = F::Complex;
    type Output = F::Complex;
    type Error = LogDerivativeError<F::Complex>;

    fn fallible_integrand(&self, z: &Self::Input) -> Result<Self::Output, Self::Error> {
        let log_derivative = checked_log_derivative(self.function, *z, self.zero_tol)?;

        let mut z_power = <<F::Complex as ComplexField>::RealField as ComplexScalar>::complex(
            <<F::Complex as ComplexField>::RealField as One>::one(),
            <<F::Complex as ComplexField>::RealField as Zero>::zero(),
        );

        for _ in 0..self.power {
            z_power *= *z;

            if !z_power.is_finite() {
                return Err(LogDerivativeError::NonFiniteLogDerivative { z: *z });
            }
        }

        let value = z_power * log_derivative;

        if !value.is_finite() {
            return Err(LogDerivativeError::NonFiniteLogDerivative { z: *z });
        }

        Ok(value)
    }
}

fn checked_log_derivative<F>(
    function: &F,
    z: F::Complex,
    zero_tol: <F::Complex as ComplexField>::RealField,
) -> Result<F::Complex, LogDerivativeError<F::Complex>>
where
    F: ComplexFunction,
    F::Complex: ComplexField + Copy,
{
    let fz = function.value(z);

    if !fz.is_finite() {
        return Err(LogDerivativeError::NonFiniteFunctionValue { z });
    }

    let dfz = function.derivative(z);

    if !dfz.is_finite() {
        return Err(LogDerivativeError::NonFiniteDerivativeValue { z });
    }

    let norm = fz.modulus();

    if norm <= zero_tol {
        return Err(LogDerivativeError::NearSingularContour { z, norm });
    }

    let value = dfz / fz;

    if !value.is_finite() {
        return Err(LogDerivativeError::NonFiniteLogDerivative { z });
    }

    Ok(value)
}

#[allow(dead_code)]
pub fn compute_argument_data<C, F>(
    function: &F,
    contour: Contour<C::RealField>,
    config: IntegratorConfig<C::RealField>,
    zero_tol: C::RealField,
    search_target: SearchTarget,
) -> Result<ArgumentData<C>, ArgumentError<C>>
where
    C: ComplexField + IntegrationOutput<C, Float = C::RealField> + Copy,
    <C as ComplexField>::RealField:
        ComplexScalar<Complex = C> + IntegrableFloat + FromPrimitive + ToPrimitive,
    F: ComplexFunction<Complex = C>,
{
    let winding = compute_winding(
        function,
        contour.clone(),
        config.clone(),
        zero_tol,
        search_target,
    )?;
    let first_moment = compute_moment(function, contour, config, 1, zero_tol, search_target)?;

    Ok(ArgumentData {
        winding,
        first_moment,
    })
}

pub fn compute_winding<C, F>(
    function: &F,
    contour: Contour<C::RealField>,
    config: IntegratorConfig<C::RealField>,
    zero_tol: C::RealField,
    search_target: SearchTarget,
) -> Result<WindingData<C>, ArgumentError<C>>
where
    C: ComplexField + IntegrationOutput<C, Float = C::RealField> + Copy,
    <C as ComplexField>::RealField:
        ComplexScalar<Complex = C> + IntegrableFloat + FromPrimitive + ToPrimitive,
    F: ComplexFunction<Complex = C>,
{
    let result = integrate_complex_fallible(LogDerivative { function, zero_tol }, contour, config)?;

    let winding = normalise_argument_integral(result.integral) * C::from_real(search_target.sign());

    let nearest: C::RealField = winding.real().round();
    let root_count = nearest.to_isize().expect("nearest should be representable");

    let residual = (winding
        - <<C as ComplexField>::RealField as ComplexScalar>::complex(nearest, C::zero().real()))
    .modulus();

    Ok(WindingData {
        winding,
        root_count,
        residual,
        integration_error: result.error,
    })
}

pub fn compute_moment<C, F>(
    function: &F,
    contour: Contour<C::RealField>,
    config: IntegratorConfig<C::RealField>,
    power: usize,
    zero_tol: C::RealField,
    search_target: SearchTarget,
) -> Result<MomentData<C>, ArgumentError<C>>
where
    C: ComplexField + IntegrationOutput<C, Float = C::RealField> + Copy,
    <C as ComplexField>::RealField:
        ComplexScalar<Complex = C> + IntegrableFloat + FromPrimitive + ToPrimitive,
    F: ComplexFunction<Complex = C>,
{
    let result = integrate_complex_fallible(
        LogDerivativeMoment {
            function,
            power,
            zero_tol,
        },
        contour,
        config,
    )?;

    Ok(MomentData {
        power,
        moment: normalise_argument_integral(result.integral) * C::from_real(search_target.sign()),
        integration_error: result.error,
    })
}

fn normalise_argument_integral<C>(integral: C) -> C
where
    C: ComplexField,
    <C as ComplexField>::RealField: ComplexScalar<Complex = C> + FromPrimitive,
{
    let two = C::one() + C::one();
    let pi =
        <<C as ComplexField>::RealField as FromPrimitive>::from_f64(std::f64::consts::PI).unwrap();
    let pi_i = <<C as ComplexField>::RealField as ComplexScalar>::complex(C::zero().real(), pi);

    integral / two / pi_i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::function::ComplexFunction;

    use approx::assert_relative_eq;
    use num_complex::Complex;
    use quad_rs::{CircularArc, Contour, ContourSegment, IntegratorConfig};

    const TOL: f64 = 1e-8;
    const ZERO_TOL: f64 = 1e-10;

    fn circle(radius: f64) -> Contour<f64> {
        Contour::from_pieces(vec![ContourSegment::CircularArc(CircularArc::new(
            Complex::new(0.0, 0.0),
            radius,
            0.0,
            2.0 * std::f64::consts::PI,
        ))])
    }

    fn config() -> IntegratorConfig<f64> {
        IntegratorConfig::default()
            .with_absolute_tolerance(1e-10)
            .with_relative_tolerance(1e-10)
    }

    #[derive(Debug, Clone, Copy)]
    struct Linear {
        root: Complex<f64>,
    }

    impl ComplexFunction for Linear {
        type Complex = Complex<f64>;

        fn value(&self, z: Complex<f64>) -> Complex<f64> {
            z - self.root
        }

        fn derivative(&self, _z: Complex<f64>) -> Complex<f64> {
            Complex::new(1.0, 0.0)
        }
    }

    #[test]
    fn compute_winding_counts_single_root_inside_contour() {
        let f = Linear {
            root: Complex::new(0.2, -0.1),
        };

        let data =
            compute_winding(&f, circle(1.0), config(), ZERO_TOL, SearchTarget::Zeros).unwrap();

        assert_eq!(data.root_count, 1);
        assert_relative_eq!(data.winding.re, 1.0, epsilon = TOL);
        assert_relative_eq!(data.winding.im, 0.0, epsilon = TOL);
        assert!(data.residual < TOL);
    }

    #[test]
    fn compute_winding_ignores_single_root_outside_contour() {
        let f = Linear {
            root: Complex::new(2.0, 0.0),
        };

        let data =
            compute_winding(&f, circle(1.0), config(), ZERO_TOL, SearchTarget::Zeros).unwrap();

        assert_eq!(data.root_count, 0);
        assert_relative_eq!(data.winding.re, 0.0, epsilon = TOL);
        assert_relative_eq!(data.winding.im, 0.0, epsilon = TOL);
        assert!(data.residual < TOL);
    }

    #[test]
    fn compute_first_moment_of_single_root_returns_root_location() {
        let root = Complex::new(0.25, -0.3);

        let f = Linear { root };

        let moment =
            compute_moment(&f, circle(1.0), config(), 1, ZERO_TOL, SearchTarget::Zeros).unwrap();

        assert_eq!(moment.power, 1);
        assert_relative_eq!(moment.moment.re, root.re, epsilon = TOL);
        assert_relative_eq!(moment.moment.im, root.im, epsilon = TOL);
    }

    #[test]
    fn compute_zeroth_moment_matches_winding() {
        let root = Complex::new(-0.2, 0.1);

        let f = Linear { root };

        let winding =
            compute_winding(&f, circle(1.0), config(), ZERO_TOL, SearchTarget::Zeros).unwrap();
        let moment0 =
            compute_moment(&f, circle(1.0), config(), 0, ZERO_TOL, SearchTarget::Zeros).unwrap();

        assert_relative_eq!(moment0.moment.re, winding.winding.re, epsilon = TOL);
        assert_relative_eq!(moment0.moment.im, winding.winding.im, epsilon = TOL);
    }

    #[derive(Debug, Clone, Copy)]
    struct Quadratic {
        a: Complex<f64>,
        b: Complex<f64>,
    }

    impl ComplexFunction for Quadratic {
        type Complex = Complex<f64>;
        fn value(&self, z: Complex<f64>) -> Complex<f64> {
            (z - self.a) * (z - self.b)
        }

        fn derivative(&self, z: Complex<f64>) -> Complex<f64> {
            (z - self.a) + (z - self.b)
        }
    }

    #[test]
    fn compute_winding_counts_two_roots_inside_contour() {
        let a = Complex::new(0.2, 0.1);
        let b = Complex::new(-0.3, 0.2);

        let f = Quadratic { a, b };

        let data =
            compute_winding(&f, circle(1.0), config(), ZERO_TOL, SearchTarget::Zeros).unwrap();

        assert_eq!(data.root_count, 2);
        assert_relative_eq!(data.winding.re, 2.0, epsilon = TOL);
        assert_relative_eq!(data.winding.im, 0.0, epsilon = TOL);
        assert!(data.residual < TOL);
    }

    #[test]
    fn compute_first_moment_of_two_roots_returns_root_sum() {
        let a = Complex::new(0.2, 0.1);
        let b = Complex::new(-0.3, 0.2);

        let f = Quadratic { a, b };

        let moment =
            compute_moment(&f, circle(1.0), config(), 1, ZERO_TOL, SearchTarget::Zeros).unwrap();

        let expected = a + b;

        assert_relative_eq!(moment.moment.re, expected.re, epsilon = TOL);
        assert_relative_eq!(moment.moment.im, expected.im, epsilon = TOL);
    }

    #[test]
    fn compute_second_moment_returns_sum_of_squares() {
        let a = Complex::new(0.2, 0.1);
        let b = Complex::new(-0.3, 0.2);

        let f = Quadratic { a, b };

        let moment =
            compute_moment(&f, circle(1.0), config(), 2, ZERO_TOL, SearchTarget::Zeros).unwrap();

        let expected = a * a + b * b;

        assert_relative_eq!(moment.moment.re, expected.re, epsilon = TOL);
        assert_relative_eq!(moment.moment.im, expected.im, epsilon = TOL);
    }

    #[derive(Debug, Clone, Copy)]
    struct DoubleRoot {
        root: Complex<f64>,
    }

    impl ComplexFunction for DoubleRoot {
        type Complex = Complex<f64>;
        fn value(&self, z: Complex<f64>) -> Complex<f64> {
            let dz = z - self.root;
            dz * dz
        }

        fn derivative(&self, z: Complex<f64>) -> Complex<f64> {
            Complex::new(2.0, 0.0) * (z - self.root)
        }
    }

    #[test]
    fn compute_winding_counts_multiplicity() {
        let root = Complex::new(0.2, -0.1);

        let f = DoubleRoot { root };

        let data =
            compute_winding(&f, circle(1.0), config(), ZERO_TOL, SearchTarget::Zeros).unwrap();

        assert_eq!(data.root_count, 2);
        assert_relative_eq!(data.winding.re, 2.0, epsilon = TOL);
        assert_relative_eq!(data.winding.im, 0.0, epsilon = TOL);
        assert!(data.residual < TOL);
    }

    #[test]
    fn compute_first_moment_counts_multiplicity() {
        let root = Complex::new(0.2, -0.1);

        let f = DoubleRoot { root };

        let moment =
            compute_moment(&f, circle(1.0), config(), 1, ZERO_TOL, SearchTarget::Zeros).unwrap();

        let expected = Complex::new(2.0, 0.0) * root;

        assert_relative_eq!(moment.moment.re, expected.re, epsilon = TOL);
        assert_relative_eq!(moment.moment.im, expected.im, epsilon = TOL);
    }

    #[derive(Debug, Clone, Copy)]
    struct OneInsideOneOutside {
        inside: Complex<f64>,
        outside: Complex<f64>,
    }

    impl ComplexFunction for OneInsideOneOutside {
        type Complex = Complex<f64>;
        fn value(&self, z: Complex<f64>) -> Complex<f64> {
            (z - self.inside) * (z - self.outside)
        }

        fn derivative(&self, z: Complex<f64>) -> Complex<f64> {
            (z - self.inside) + (z - self.outside)
        }
    }

    #[test]
    fn compute_winding_ignores_roots_outside_contour() {
        let inside = Complex::new(0.2, 0.0);
        let outside = Complex::new(2.0, 0.0);

        let f = OneInsideOneOutside { inside, outside };

        let data =
            compute_winding(&f, circle(1.0), config(), ZERO_TOL, SearchTarget::Zeros).unwrap();

        assert_eq!(data.root_count, 1);
        assert_relative_eq!(data.winding.re, 1.0, epsilon = TOL);
        assert_relative_eq!(data.winding.im, 0.0, epsilon = TOL);
    }

    #[test]
    fn compute_first_moment_ignores_roots_outside_contour() {
        let inside = Complex::new(0.2, 0.0);
        let outside = Complex::new(2.0, 0.0);

        let f = OneInsideOneOutside { inside, outside };

        let moment =
            compute_moment(&f, circle(1.0), config(), 1, ZERO_TOL, SearchTarget::Zeros).unwrap();

        assert_relative_eq!(moment.moment.re, inside.re, epsilon = TOL);
        assert_relative_eq!(moment.moment.im, inside.im, epsilon = TOL);
    }

    #[test]
    fn normalise_argument_integral_maps_two_pi_i_to_one() {
        let i = Complex::new(0.0, 1.0);
        let integral = 2.0 * std::f64::consts::PI * i;

        let normalised = normalise_argument_integral(integral);

        assert_relative_eq!(normalised.re, 1.0, epsilon = TOL);
        assert_relative_eq!(normalised.im, 0.0, epsilon = TOL);
    }
}
