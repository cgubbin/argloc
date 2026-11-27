// src/output.rs

use crate::{cell::ArgumentCell, localisation::SingularPointEstimate};

use nalgebra::ComplexField;
use num_traits::Float;
use quadtree_core::{QuadTreeResult, RawLeaf, RunSummary, Termination};

#[derive(Debug)]
pub struct ArgumentResult<C: ComplexField> {
    pub points: Vec<SingularPointEstimate<C>>,
    pub leaves: Vec<ArgumentLeaf<C::RealField>>,
    pub summary: RunSummary<C::RealField>,
    pub termination: Termination,
}

#[derive(Debug, Clone)]
pub struct ArgumentLeaf<T> {
    pub bounds: quadtree_core::Rect<T>,
    pub depth: usize,
    pub data: ArgumentCell<T>,
}

impl<C: ComplexField> ArgumentResult<C> {
    pub fn points(&self) -> &[SingularPointEstimate<C>] {
        &self.points
    }

    pub fn summary(&self) -> &RunSummary<C::RealField> {
        &self.summary
    }

    pub fn termination(&self) -> Termination {
        self.termination
    }
}

impl<T> ArgumentLeaf<T>
where
    T: Copy + Float,
{
    /// The leaf with its bounds clamped into `domain`: an edge that rounded
    /// outside the domain is moved onto the domain's edge.
    pub(crate) fn clamped_to(mut self, domain: quadtree_core::Rect<T>) -> Self {
        let b = &mut self.bounds;
        b.x_min = b.x_min.max(domain.x_min);
        b.x_max = b.x_max.min(domain.x_max);
        b.y_min = b.y_min.max(domain.y_min);
        b.y_max = b.y_max.min(domain.y_max);
        self
    }

    pub fn from_raw_leaf(leaf: RawLeaf<'_, T, ArgumentCell<T>>) -> Self {
        Self {
            bounds: leaf.bounds(),
            depth: leaf.depth(),
            data: leaf.data().clone(),
        }
    }
}

impl<C> From<QuadTreeResult<C::RealField, ArgumentCell<C::RealField>>> for ArgumentResult<C>
where
    C: ComplexField,
    C::RealField: Copy + Float + std::fmt::Debug,
{
    fn from(result: QuadTreeResult<C::RealField, ArgumentCell<C::RealField>>) -> Self {
        let leaves: Vec<_> = result.iter().map(ArgumentLeaf::from_raw_leaf).collect();

        Self {
            points: Vec::new(),
            leaves,
            summary: result.summary,
            termination: result.termination,
        }
    }
}
