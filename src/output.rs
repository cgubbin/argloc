// src/output.rs

use crate::{cell::ArgumentCell, localisation::SingularPointEstimate};

use nalgebra::ComplexField;
use num_traits::Float;
use quadtree_core::{QuadTreeResult, RawLeaf, RunSummary, Termination};

#[derive(Debug)]
pub struct ArgumentResult<C: ComplexField> {
    pub roots: Vec<SingularPointEstimate<C>>,
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

impl<T> ArgumentLeaf<T>
where
    T: Copy + Float,
{
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
            roots: Vec::new(),
            leaves,
            summary: result.summary,
            termination: result.termination,
        }
    }
}
