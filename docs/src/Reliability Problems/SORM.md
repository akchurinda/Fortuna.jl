# [Second-Order Reliability Methods](@id SORMPage)

The SORM is an improvement over the FORM by accounting for the curved nature of the failure boundary given by ``g(\vec{X}) = 0`` around the design point ``\vec{x}^{*}``; thus, providing a better approximation of the probability of failure ``P_{f}``.

## Curve-Fitting Method

The CF method fits a hyper-paraboloid surface with a vertex at the design point ``\vec{x}^{*}`` and the principal curvatures matching the principal curvatures of the failure boundary given by ``g(\vec{X}) = 0`` at that point. The probabilities ``P_{f}`` of failure are estimated using [Hohenbichler:1988](@citet) and [Breitung:1984](@citet) approximations of the exact solution provided by [Tvedt:1990](@citet). The calculated probabilities of failure ``P_{f}`` are then used to estimate the generalized reliability indices ``\beta``, which account for the curved nature of the failure boundary given by ``g(\vec{X}) = 0`` around the design point ``\vec{x}^{*}``.

## Point-Fitting Method

The PF method fits a series of hyper-semiparaboloid surfaces with a vertex at the design point ``\vec{x}^{*}``. The principal curvatures of each surface are estimated using fitting points found at the intersections of a hyper-cylinder with axis coinciding with the design point ``\vec{u}^{*}`` and the failure boundary given by ``g(\vec{U}) = 0`` in ``U``-space. The PF method provides a better estimate of the probability of failure ``P_{f}`` than the CF method since it provides a better approximation of highly non-linear failure boundaries given by ``g(\vec{X}) = 0`` that are unsymmetrical about the design point ``\vec{x}^{*}``.

!!! note
    
    A great description of both methods can be found in [DerKiureghian:2022](@citet).

## API

```@docs
solve(Problem::ReliabilityProblem, AnalysisMethod::SORM)
SORM
CF
CFCache
PF
PFCache
```
