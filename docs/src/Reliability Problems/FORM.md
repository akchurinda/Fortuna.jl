# [First-Order Reliability Methods](@id FORMPage)

## Mean-Centered First-Order Second-Moment Method

The MCFOSM method is the simplest and least computationally expensive type of reliability method. It utilizes the first-order Taylor expansion of the limit state function ``g(\vec{X})`` at the mean values and the first two moments of the marginal random variables involved in the reliability problem to evaluate the reliability index. However, despite the fact that it is simple and does not require the complete knowledge of the random variables involved in the reliability problem, the MCFOSM method faces an issue known as the *invariance problem*. This problem arises because the resulting reliability index ``\beta`` is dependent on the formulation of the limit state function ``g(\vec{X})``. In other words, two equivalent limit state functions with the same failure boundaries produce two different reliability indices; thus, the use of MCFOSM method is not recommended.

## Rackwitz-Fiessler Method

The RF method, also known as the Equivalent Normal method, is an efficient way to solve reliability problems involving strictly uncorrelated random variables. The RF method overcomes the invariance problem faced by the MCFOSM method by using the first-order Taylor expansion of the limit state function at a point known as the design point ``\vec{x}^{*}`` that lies on the failure boundary given by ``g(\vec{X}) = 0``. Since the design point ``\vec{x}^{*}`` is not known a priori, the RF method is inherently an iterative method. At each iteration, the RF method replaces the original random variables with equivalent normal random variables, which allows for a direct transformation from the original non-normal space into the standard normal space in which the reliability index ``\beta`` is calculated.

## Plain and Improved Hasofer-Lind-Rackwitz-Fiessler Method

The HLRF method is the most accepted and efficient way to solve reliability problems involving both uncorrelated and correlated random variables. The HLRF method also overcomes the invariance problem faced by the MCFOSM method by using the first-order Taylor expansion of the limit state function at the design point ``\vec{x}^{*}``. Again, since the design point ``\vec{x}^{*}`` is not known a priori, the HLRF method is inherently an iterative method. At each iteration, the HLRF method performs [Nataf Transformation](@ref NatafTransformationPage) to transformation from the original non-normal space into the standard normal space in which the reliability index ``\beta`` is calculated. `Fortuna.jl` package implements two versions of HLRF method: plain HLRF method where the step size ``\lamba`` in the negative gradient descent is always set to unity and improved HLRF (iHLRF) method where the step size ``\lamba`` is determined using a line search algorithm.

## API

```@docs
solve(Problem::ReliabilityProblem, AnalysisMethod::FORM)
FORM
MCFOSM
MCFOSMCache
HLRF
HLRFCache
iHLRF
iHLRFCache
```
