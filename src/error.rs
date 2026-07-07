use crate::{argument::ArgumentError, cell::ArgumentCell, subdivision::SubdivisionError};
use num_traits::Float;
use quadtree_core::{QuadTree, QuadTreeError, TrellisFloat};

#[allow(clippy::type_complexity)]
#[derive(thiserror::Error, Debug)]
pub enum FindSingularitiesError<C>
where
    C: nalgebra::ComplexField,
    C::RealField: Float + TrellisFloat,
{
    #[error("adaptive quadtree refinement failed: {0}")]
    Refinement(
        #[from]
        QuadTreeError<
            QuadTree<C::RealField, ArgumentCell<C::RealField>>,
            C::RealField,
            SubdivisionError<C::RealField>,
            ArgumentError<C>,
        >,
    ),

    #[error("root localisation failed: {0}")]
    Localisation(#[from] ArgumentError<C>),
}
