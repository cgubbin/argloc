# argloc

## argloc

`argloc` locates zeros and poles of complex-valued functions using the
Argument Principle, contour integration, and adaptive quadtree refinement.

The crate searches rectangular regions of the complex plane. Each region is
surrounded by a contour, the logarithmic derivative is integrated around
that contour, and the resulting winding number determines how many target
singularities lie inside the region.

Adaptive refinement is provided by `quadtree_core`; contour integration is
provided by `quad_rs`.

## Mathematical basis

For a meromorphic function `f(z)` and a closed contour `Γ` containing no
zeros or poles on the contour, the Argument Principle states:

```
             1
N - P = ------------ ∮Γ f′(z) / f(z) dz
         2πi
```

where:

- `N` is the number of zeros inside `Γ`;
- `P` is the number of poles inside `Γ`;
- both are counted with multiplicity.

`argloc` applies this identity to rectangular cell boundaries. Cells with
non-zero target count, uncertain winding, or unresolved numerical error are
refined adaptively.

## Zeros, poles, and API assumptions

The Argument Principle computes a **signed** count, `N - P`. It does not
separately report `N` and `P`.

This is a fundamental limitation. A region containing

```
3 zeros and 2 poles
```

has the same signed argument count as a region containing

```
1 zero and 0 poles.
```

Therefore a single logarithmic-derivative contour integral cannot, in
general, robustly separate zeros from poles in a mixed meromorphic function.

### `find_zeros`

[`find_zeros`] locates zeros of a function in a search domain.

It assumes that the supplied function has no poles in the search domain.
This is the usual holomorphic root-finding case, for example:

```
f(z) = z^3 - 1
```

If poles are present in the search region, the signed argument count may
become negative. In that case the assumptions of [`find_zeros`] have been
violated and the solver returns an error rather than silently producing an
incorrect result.

### `find_poles`

[`find_poles`] locates poles of a function in a search domain.

Internally, this reverses the sign convention so that poles contribute
positive counts. It is suitable for functions such as:

```
f(z) = 1 / (z^2 - 5)
```

where the poles are the objects of interest.

If zeros are also present in the same search region, they subtract from the
pole count. The solver cannot generally distinguish this from a smaller
number of poles using the argument-principle integral alone.

### Mixed rational functions

For a rational function

```
f(z) = g(z) / h(z)
```

zeros and poles should be handled separately when possible:

- use [`find_zeros`] on `g` to locate zeros;
- use [`find_zeros`] on `h`, or [`find_poles`] on `f`, to locate poles.

This is the robust approach because it avoids asking the signed count
`N - P` to recover two unknown quantities.

A future lower-level API may expose raw signed argument-count regions for
users who explicitly want to analyse `N - P` directly.

## Localisation

After refinement, target locations are estimated using the first
logarithmic-derivative moment:

```
S₁ = (1 / 2πi) ∮Γ z f′(z) / f(z) dz
```

For zero searches this gives the sum of enclosed zeros. For pole searches
the sign convention is reversed, giving the sum of enclosed poles.

If a contour encloses target points `z₁, …, zₙ`, counted with multiplicity,
then:

```
S₁ = z₁ + ... + zₙ
```

and:

```
S₁ / n
```

is the multiplicity-weighted centroid.

For a single-target cell this is a point estimate. For a multi-target cell
this is a cluster centroid, not an individual root or pole location.

## Boundary singularities

A zero or pole on a cell boundary makes `f′(z) / f(z)` singular on the
integration contour. This commonly occurs during adaptive subdivision when a
target lies exactly on a split line.

`argloc` treats this as a recoverable refinement event. The integrand reports
a near-singular contour point, the oracle maps that point into the quadtree
coordinate system, and the subdivision policy retries with a shifted split.

This preserves the rectangular partition and avoids contour indentation,
whose inclusion/exclusion semantics can make root-count bookkeeping
ambiguous.

## Basic usage

```rust
use argloc::{find_zeros, ArgumentConfig, ComplexFunction};
use num_complex::Complex;
use quadtree_core::Rect;

#[derive(Debug, Clone, Copy)]
struct Cubic;

impl ComplexFunction for Cubic {
    type Complex = Complex<f64>;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        z * z * z - Complex::new(1.0, 0.0)
    }

    fn derivative(&self, z: Self::Complex) -> Self::Complex {
        Complex::new(3.0, 0.0) * z * z
    }
}

let domain = Rect::new(-1.5, 1.5, -1.5, 1.5)?;
let config = ArgumentConfig::new(1e-3);

let result = find_zeros(Cubic, domain, config)?;

for estimate in result.points {
    println!(
        "{:?}: z = {}, multiplicity = {}",
        estimate.kind,
        estimate.location,
        estimate.multiplicity
    );
}
```

## Finding poles

```rust
use argloc::{find_poles, ArgumentConfig, ComplexFunction};
use num_complex::Complex;
use quadtree_core::Rect;

#[derive(Debug, Clone, Copy)]
struct ReciprocalQuadratic;

impl ComplexFunction for ReciprocalQuadratic {
    type Complex = Complex<f64>;

    fn value(&self, z: Self::Complex) -> Self::Complex {
        Complex::new(1.0, 0.0) / (z * z - Complex::new(5.0, 0.0))
    }

    fn derivative(&self, z: Self::Complex) -> Self::Complex {
        let d = z * z - Complex::new(5.0, 0.0);
        -Complex::new(2.0, 0.0) * z / (d * d)
    }
}

let domain = Rect::new(-3.0, 3.0, -1.0, 1.0)?;
let config = ArgumentConfig::new(1e-3);

let result = find_poles(ReciprocalQuadratic, domain, config)?;

for pole in result.points {
    println!("pole at {}, multiplicity {}", pole.location, pole.multiplicity);
}
```

## Main components

- [`ComplexFunction`] defines the user-supplied complex function and
  derivative.
- [`ArgumentConfig`] controls quadtree refinement, integration tolerances,
  singularity thresholds, and boundary recovery.
- [`find_zeros`] locates zeros under the no-poles-in-domain assumption.
- [`find_poles`] locates poles under the corresponding pole-search
  convention.
- [`SingularPointEstimate`] describes a target estimate or cluster centroid.
- [`FindSingularitiesError`] is the public error type for the high-level API.

## Design philosophy

The crate separates the algorithm into independent layers:

- contour integration computes winding numbers and moments;
- the oracle turns rectangular cells into argument-principle data;
- the quadtree engine handles adaptive refinement;
- localisation interprets final cells as target estimates.

This keeps the complex-analysis code independent of the adaptive refinement
engine, while still allowing domain-specific recovery such as shifted
subdivision near boundary singularities.

## Limitations

Mixed zero/pole regions cannot be robustly separated from `f′/f` alone.
The high-level APIs therefore make explicit assumptions about the target
singularity type.

Multi-target cells are reported as centroids. Individual reconstruction from
higher contour moments may be added later.

License: MIT
