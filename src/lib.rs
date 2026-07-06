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
pub use localisation::{RootEstimate, RootEstimateKind};
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

pub fn find_roots<F, C, T>(
    function: F,
    domain: Rect<T>,
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
    )?;

    Ok(ArgumentResult {
        roots,
        leaves,
        summary: result.summary,
        termination: result.termination,
    })
}
