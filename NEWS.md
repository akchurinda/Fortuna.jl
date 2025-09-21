# News

## Release V0.11.0

- Refactored the entire codebase to use `snake_case` for consistency and clarity.
- Improved the stability of the Nataf Transformation.

## Release V0.10.1

- Fixed a bug which caused the LHS technique to fail when the number samples is large (>1E8).

## Release V0.10.0

- `Fortuna.jl` package now can solve reliability problems defined using Tcl-based `OpenSees` and `Abaqus` software. A simple example with a cantilever beam can be found in `examples/OpenSees` and `example/Abaqus`. A detailed description of the general procedure is described in the documentation.
- A lot of optional keyword arguments are switched to lowercase to conform to the community standards. Most notably, `Differentiation` keyword in `solve()` is changed to `diff` to conform to community standards. The possible arguments are `:automatic` and `:numeric`.
- `solve()` function invoked with `AnalysisMethod` of type `SORM` now accepts a new optional `FORMSolution` keyword argument, which can be used to pass an existing solution of a reliability problem obtained using FORM, significantly speeding up the SORM.
- Fixed a problem with SORM prematurely erroring out if Hohenbichler-Rackwitz and/or Breitung conditions are not satisfied.
- `solve()` function invoked with `AnalysisMethod` of type `MC` and `IS` now accepts a new optional `showprogressbar` keyword argument, which can be used to track the progress of the progress in real time. Is in it nice?!
- Small code optimizations here and there.

## Release V0.9.1

- Implementation of FORM's gradient-based descent optimization submethods is now more standardized and follows a better code-implementation practice.
- Before, if FORM failed converge, it would throw an **error** stating that the solution did not converge and give no other output. Now, if FORM fails to converge to a satisfactory solution, it will throw a **warning** stating that the solution did not converge and provide the solution cache history to help diagnose the problem. Additionally, a new `Convergance` field was added to cache histories of HLRF, iHLRF, and RF methods, which can be used to check whether the convergence was achieved.

```julia
Solution = solve(Problem, FORM())

if Solution.Convergance == true
  # Solution converged...
  # Proceed with the post-processing...
else
  # Solution did not converge...
  # Diagnose the problem and try to solve again...
end
```

- Relaxed default tolerances in FORM's submethods from $10^{-9}$ to $10^{-6}$.

## Release V0.9.0

- Added functionality to solve inverse reliability problems.
- Added an option to choose a desired FORM's submethod for SORM.

## Release V0.8.2

- Fixed a bug with the iHLRF method stalling in very specific cases.
- Added precompilation of the source code for performance purposes.
- Added more examples with FEM solvers and generally clean up the documentation.

## Release V0.8.1

- Added fallback onto numerical differentiation for non-linear problem solver. Fixes a problem with Rackwitz-Fiessler and Point-Fitting methods stalling if a non-differentiable limit state function is encountered.

## Release V0.8.0

- Limit state functions can now be given by OpenSeesPy models directly in Julia! This is made possible by using the `PyCall.jl` and `Conda.jl` packages that allow to bring Python functionality into Julia.
- If `ForwardDiff.jl` package fails to compute gradients, jacobians, etc., then `Fortuna.jl` package will attempt to use `FiniteDiff.jl` package before erroring out. This fixes a problem of differentiating some functions outside of `Fortuna.jl` package that are implemented in specific precisions which are not compatible with `Dual()` numbers.
- A new `Differentiation` keyword was added to `solve()` function to indicate the desired differentiation scheme:
  - `Differentiation` = `:Automatic`, then the function will use automatic differentiation (`ForwardDiff.jl`) to compute gradients, jacobians, etc.
  - `Differentiation` = `:Numeric`, then the function will use numeric differentiation (`FiniteDiff.jl`) to compute gradients, jacobians, etc.
  - Important to note that if your limit state function is given by an OpenSeesPy model, solving the reliability problem with `Differentiation` = `:Automatic` will not fail because of the fallback onto `FiniteDiff.jl` package; however, it is highly recommended to use `Differentiation` = `:Numeric` from the start to avoid even attempting to use automatic differentiation before falling back onto numeric differentiation.
- Sampling technique for `rand()` function now must be given by symbols, i.e. `:ITS` or `:LHS`.

## Release V0.7.1

- `Dual()` number can now be propagated as moments of random variables.
- Fixed a bug with `randomvariables()` function.
- Transformation direction for `transformsamples()` and `getjacobian()` functions now must be given by symbols, i.e. `:X2U` or `:U2X`, which seems to be a more acceptable way of defining keyword arguments in Julia.
- Added proper error handling and optimized some parts of the code.

## Release V0.7.0

- Added Frechet distribution.
- Added functionality to perform sensitivity analysis w.r.t. to parameters of distributions involved in the reliability problem. To avoid confusion with sensitivity analysis w.r.t. to parameters of limit state function, the following categorization of sensitivity problems was employed:
  - `SensitivityProblemTypeI`:  Sensitivities w.r.t. the parameters of the limit state function.
  - `SensitivityProblemTypeII`: Sensitivities w.r.t. the parameters of the random vector.
- Cleaned up the documentation.

## Release V0.6.2

- Added more examples to the documentation and cleaned up a few bugs.

## Release V0.6.1

- Added functionality to solve reliability problems using Rackwitz-Fiessler (RF) method.

## Release V0.6.0

- The internals of the module were completely reworked to use `import` instead of `using` for most of the dependencies to avoid pollution of the global variable scope of the module and improve its performance. The code was also completely reorganized to allow for easier contributions from other people in the future.
- The following function and types were completely reworked:
  - `generaterv()` is replaced with `randomvariable()`.
  - `samplerv()` is replaced by extending `rand()` from `Distributions.jl` package.
  - `jointpdf()` is replaced by extending `pdf()` from `Distributions.jl` package.
  - `analyze()` is replaced with `solve()`.
  - `MCS()` is replated with `MC()`. Consequently, `MCSCache()` is replaced with `MCCache()`.
- `rand()` method is now completely compatible with `Random.jl` and `Distributions.jl` packages, i.e. you can now generate reproducible sequence of numbers if RNG seed is provided.
- `pdf()` method is now completely compatible with `Distributions.jl` package.
- HLRF and iHLRF methods can now be started from any arbitrary point $x_{0}$.

```julia
solve(Problem, FORM( HLRF(x₀ = [150, 275])))
solve(Problem, FORM(iHLRF(x₀ = [150, 275])))
```

- Added functionality to solve reliability problems with extremely small probabilities of failure using the Importance Sampling method.

```julia
ISSolution = solve(Problem, IS())
```

- Updated the Subset Simulation Method. It is now slightly faster and more tested.

- Isoprobabilistic transformation objects (`NatafTransformation` and `RosenblattTransformation`) are now broadcastable.

```julia
pdf.(TransformationObject, x)
```

- Documentation is updated to include all possible functionality.

## Release V0.5.2

- Added functionality to perform sensitivity analysis.
- Sped up the calculations of the importance vector $\gamma$. 

## Release V0.5.1

- Added measure of importance of random variables $\gamma$ to FORM. Note that the importance vector $\gamma$ is only available if the reliability analysis was carried out using HLRF or iHLRF methods.

```julia
# Plain Hasofer-Lind-Rackwitz-Fiessler method:
FORMSolution = analyze(Problem, FORM(HLRF()))
FORMSolution.γ

# Improved Hasofer-Lind-Rackwitz-Fiessler method:
FORMSolution = analyze(Problem, FORM(iHLRF()))
FORMSolution.γ
```

## Release V0.5.0

- Added functionality to probabilities of failure by means of Monte Carlo Simulations.

```julia
MCSSolution = analyze(Problem, MCS())
```

- Added Point-Fitting method that falls within a broader class of the Second-Order Reliability Methods (SORM).

```julia
SORMSolution = analyze(Problem, SORM(PF()))
```

- Added an option to sample random variables with correlated marginal random variables $\vec{X}$ using any implemented sampling technique. Note that your choice of the sampling technique pertains to the sampling technique that is used to generate samples in the space of uncorrelated standard normal random variables $\vec{U}$. The generated samples are then transformed from $\vec{U}$- to $\vec{X}$-space according to transformation you use to define a random vector with correlated marginal random variables $\vec{X}$, i.e., either Nataf or Rosennblatt (not yet implemented) transformation.

```julia
XSamples, ZSamples, USamples = samplerv(TransformationObject, 1000, ITS())
XSamples, ZSamples, USamples = samplerv(TransformationObject, 1000, LHS())
```

- Updated the documentation.
- Added Dependabot to automatically take care of GitHub Actions' versions.

## Release V0.4.1

- Accelerate the Subset Simulation Method.

```julia
SSMSolution = analyze(Problem, SSM())
```

## Release V0.4.0

- Added the Subset Simulation Method to compute probabilities of rare failure events.

## Release V0.3.4

- General housekeeping and bug hunting.
- Added a simple documentation describing the basic functionality of the package.
- `Fortuna.jl` package now reexports `Distributions.jl` package. This allows to access basic functions, like `mean()`, `std()`, `params()`, etc., without loading `Distributions.jl` package separately.

## Release V0.3.3

- Removed `println()` statement from `samplerv()` function that was left there by accident.

## Release V0.3.2

- Fixed the generalized reliability index calculation in the SORM.

## Release V0.3.1

- Added error-catching for the SORM's probability of failure approximations.

## Release V0.3.0

- Added Curve-Fitting method that falls within a broader class of the Second-Order Reliability Methods (SORM).
- Added cache output system for analysis results for easier work in the future. You can now access a lot of iterative data related to the reliability analysis, such as the value of the limit state function, its gradient, merit function, and more at each iteration.

```julia
FORMSolution = analyze(Problem, FORM())
SORMSolution = analyze(Problem, SORM())
```

- Benchmarked FORM, SORM, and Nataf Transformation functionalities against various sources. The sources for the benchmarks are indicated. 
- Added support for Weibull distribution.

## Release V0.2.0

- Sampling function `samplerv()` now uses native `rand()` and `randn()` functions from the `Random` package to perform ITS and LHS sampling.
- `SamplingTechnique` argument's options in `samplerv()` function have been changed from strings (`"ITS"`, `"LHS"`) to their own types (`ITS()`, `LHS()`).
- Added `analyze()` function that replaces `MCFOSM()` and `FORM()` functions. The deprecated functions `MCFOSM()` and `FORM()` are now `AnalysisMethod` argument's options in `analyze()` function with their own types. For example, to perform reliability analysis using FORM use the following syntax:

```julia
β       = analyze(Problem, FORM(MCFOSM()))
β, x, u = analyze(Problem, FORM(HLRF()))
β, x, u = analyze(Problem, FORM(iHLRF()))
```

## Release V0.1.0

- Initial release of `Fortuna.jl`.