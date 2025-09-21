# [Nataf Transformation](@id NatafTransformationPage)

## Overview

The Nataf Transformation is a widely utilized isoprobabilistic transformation in structural reliability analysis. Its purpose is to transform random vectors with correlated non-normal marginals ``\vec{X}`` into random vectors with uncorrelated standard normal marginals ``\vec{U}``. This transformation was first introduced by André Nataf in 1962 [Nataf:1962](@citet).

The Nataf Transformation ``\vec{U} = T^{N}(\vec{X})`` is composed of two transformations:

```math
\vec{U} = T^{N}(\vec{X}) = (T_{2}^{N} \circ T_{1}^{N})(\vec{X})
```

  - The first transformation ``\vec{Z} = T_{1}^{N}(\vec{X})`` transforms random vector with correlated *non-normal marginals* ``\vec{X}`` (with correlation matrix ``\rho^{X}``) into random vector with correlated *standard normal marginals* ``\vec{Z}`` (with correlation matrix ``\rho^{Z}``). Here, ``\Phi^{-1}(\cdot)`` is the inverse of the cumulative density function of a standard normal random variable and ``F_{X_{i}}(\cdot)`` is the cumulative density function of a marginal``X_{i}``.

```math
\vec{Z} = T_{1}^{N}(\vec{X}) = 
\begin{bmatrix} 
    \Phi^{-1}(F_{X_{1}}(X_{1})) \\ 
    \Phi^{-1}(F_{X_{2}}(X_{2})) \\
    \vdots \\ 
    \Phi^{-1}(F_{X_{n}}(X_{n})) 
\end{bmatrix} 
```

  - The second transformation ``\vec{U} = T_{2}^{N}(\vec{Z})`` transforms random vector with *correlated* standard normal marginals ``\vec{Z}`` into random vector with *uncorrelated* standard normal marginals ``\vec{U}``. Here, the matrix ``\Gamma`` is used to decorrelate the standard normal marginals of random vector ``\vec{Z}`` and can be chosen as any square-root matrix of the correlation matrix ``\rho^{Z}``. `Fortuna.jl` uses the Cholesky factor of the inverse of the correlation matrix ``(\rho^{Z})^{-1}`` as the matrix ``\Gamma``.

```math
\vec{U} = T_{2}^{N}(\vec{Z}) = \Gamma \vec{Z}
```

The first transformation ``\vec{Z} = T_{1}^{N}(\vec{X})`` causes so-called *correlation distortion*. The correlation distortion causes the correlation coefficient between two standard normal marginals ``Z_{i}`` and ``Z_{j}``, denoted by ``\rho_{ij}^{Z}``, to distort and differ from the original correlation coefficient between the corresponding non-normal marginals ``X_{i}`` and ``X_{j}``, denoted by ``\rho_{ij}^{X}``, such that ``\rho_{ij}^{Z} \neq \rho_{ij}^{X}``. The relationship between the components of the correlation matrices ``\rho_{ij}^{Z}`` and ``\rho_{ij}^{X}`` is given by

```math
\rho_{ij}^{X} = \dfrac{1}{\sigma_{X_i} \sigma_{X_j}} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} (F_{X_i}^{-1}(\Phi(z_i)) - \mu_{X_i}) (F_{X_j}^{-1}(\Phi(z_j)) - \mu_{X_j}) \phi_2(z_i, z_j, \rho_{ij}^{Z}) dz_i dz_j
```

where ``\phi_2(\cdot)`` is the bivariate standard normal probability density function. Generally, this integral cannot be inverted analytically to solve for the coefficients of the distorted correlation matrix ``\rho_{ij}^{Z}``. In order to compute these coefficients, `Fortuna.jl` package (1) employs a two-dimensions Gauss-Legendre quadrature implemented in [`FastGaussQuadrature.jl`](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) package to expand the integral into a finite summation using [Gauss–Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature) and (2) utilizes [`NonlinearSolve.jl`](https://github.com/SciML/NonlinearSolve.jl) package to find values of the coefficients of the correlation matrix ``\rho_{ij}^{Z}`` that satisfy the resulting expression.

## API

```@docs
NatafTransformation
getdistortedcorrelation(X::AbstractVector{<:Distributions.ContinuousUnivariateDistribution}, ρ_X::AbstractMatrix{<:Real})
transformsamples(TransformationObject::NatafTransformation, Samples::AbstractVector{<:Real}, TransformationDirection::Symbol)
getjacobian(TransformationObject::NatafTransformation, Samples::AbstractVector{<:Real}, TransformationDirection::Symbol)
pdf(TransformationObject::NatafTransformation, x::AbstractVector{<:Real})
```
