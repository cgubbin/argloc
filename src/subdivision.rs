use nalgebra::ComplexField;
use quadtree_core::{
    Cell, EvaluationContext, Rect, ScalerError, SubdivisionFailure, SubdivisionPolicy,
    split_rect_at,
};

use crate::{argument::ArgumentError, cell::ArgumentCell};

pub struct ShiftSplitOnBoundary<T> {
    pub shift_fraction: T,
}

#[derive(thiserror::Error, Debug)]
pub enum SubdivisionError<T> {
    #[error(transparent)]
    Geometry(#[from] ScalerError<T>),
}

use num_traits::Float;

impl<C> SubdivisionPolicy<C::RealField, ArgumentCell<C::RealField>, ArgumentError<C>>
    for ShiftSplitOnBoundary<C::RealField>
where
    C: ComplexField + Copy,
    C::RealField: Float,
{
    type Error = SubdivisionError<C::RealField>;

    fn initial(
        &self,
        parent: &Cell<C::RealField, ArgumentCell<C::RealField>>,
        _ctx: EvaluationContext<'_, C::RealField>,
    ) -> Result<Vec<Rect<C::RealField>>, Self::Error> {
        Ok(parent.bounds().quadrants().map(|array| array.to_vec())?)
    }

    fn retry_after_failure(
        &self,
        parent: &Cell<C::RealField, ArgumentCell<C::RealField>>,
        failure: &SubdivisionFailure<C::RealField, ArgumentError<C>>,
        _ctx: EvaluationContext<'_, C::RealField>,
    ) -> Result<Option<Vec<Rect<C::RealField>>>, Self::Error> {
        let z = match &failure.error {
            ArgumentError::BoundarySingularity { scaled_z, .. } => *scaled_z,
            _ => return Ok(None),
        };

        let bounds = parent.bounds();
        let centre = bounds.centre();

        let zx = z.real();
        let zy = z.imaginary();

        let dx = bounds.width() * self.shift_fraction;
        let dy = bounds.height() * self.shift_fraction;

        let x_split = if zx >= centre.x {
            centre.x - dx
        } else {
            centre.x + dx
        };

        let y_split = if zy >= centre.y {
            centre.y - dy
        } else {
            centre.y + dy
        };

        Ok(Some(split_rect_at(bounds, x_split, y_split)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::argument::LogDerivativeError;

    use num_complex::Complex;
    use quadtree_core::Scaler2D;

    #[test]
    fn shift_split_uses_scaled_boundary_singularity_location() {
        let raw_domain = Rect::new(10.0, 20.0, -5.0, 5.0).unwrap();
        let scaler = Scaler2D::unit_square(raw_domain).unwrap();
        let ctx = EvaluationContext { scaler: &scaler };

        let parent = Cell::new(
            Rect::new(0.0, 1.0, 0.0, 1.0).unwrap(),
            0,
            ArgumentCell::new(1, 0.0, 0.0, 1.0),
        );

        let raw_z = Complex::new(15.0, -5.0);
        let scaled_z = Complex::new(0.5, 0.0);

        let failure = SubdivisionFailure {
            failed_child: Rect::new(0.0, 1.0, 0.0, 0.5).unwrap(),
            error: ArgumentError::BoundarySingularity {
                raw_z,
                scaled_z,
                raw_bounds: raw_domain,
                scaled_bounds: parent.bounds(),
                source: LogDerivativeError::NearSingularContour {
                    z: raw_z,
                    norm: 0.0,
                },
            },
        };

        let policy = ShiftSplitOnBoundary {
            shift_fraction: 0.1,
        };

        let retry = policy
            .retry_after_failure(&parent, &failure, ctx)
            .unwrap()
            .unwrap();

        // scaled_z.x == centre.x, so this convention takes the >= branch.
        // scaled_z.y < centre.y, so this takes the opposite y branch.
        assert_eq!(retry, split_rect_at(parent.bounds(), 0.4, 0.6).unwrap());
    }

    #[test]
    fn shift_split_does_not_retry_non_boundary_errors() {
        let raw_domain = Rect::new(10.0, 20.0, -5.0, 5.0).unwrap();
        let scaler = Scaler2D::unit_square(raw_domain).unwrap();
        let ctx = EvaluationContext { scaler: &scaler };

        let parent = Cell::new(
            Rect::new(0.0, 1.0, 0.0, 1.0).unwrap(),
            0,
            ArgumentCell::new(1, 0.0, 0.0, 1.0),
        );

        let failure = SubdivisionFailure {
            failed_child: parent.bounds(),
            error: ArgumentError::NonIntegerWinding {
                winding: Complex::new(0.5, 0.0),
                residual: 0.5,
            },
        };

        let policy = ShiftSplitOnBoundary {
            shift_fraction: 0.1,
        };

        let retry = policy.retry_after_failure(&parent, &failure, ctx).unwrap();

        assert!(retry.is_none());
    }
}
