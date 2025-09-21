!!! warning
    
    This feature is currently under development.

# [Rosenblatt Transformation](@id RosenblattTransformationPage)

## Overview

The Rosenblatt Transformation is another widely utilized isoprobabilistic transformation in structural reliability analysis. Similar to the Nataf Transformation, its purpose is to transform random vectors with correlated non-normal marginals ``\vec{X}`` into random vectors with uncorrelated standard normal marginals ``\vec{U}``. This transformation was first introduced by Murray Rosenblatt in 1952 [Rosenblatt:1952](@cite).

The Nataf Transformation ``\vec{U} = T^{R}(\vec{X})`` is composed of two transformations:

```math
\vec{U} = T^{R}(\vec{X}) = (T_{2}^{R} \circ T_{1}^{R})(\vec{X})
```

  - The first transformation ``\vec{Z} = T_{1}^{R}(\vec{X})`` transforms random vector with *correlated non-normal marginal random variables* ``\vec{X}`` (with correlation matrix ``\rho^{X}``) into random vector with *uncorrelated uniform marginal random variables* ``\vec{Z}``.

```math
\vec{Z} = T_{1}^{R}(\vec{X}) = 
    \begin{bmatrix} 
    F_{X_{1}}(X_{1}) \\ 
    F_{X_{2} | X_{1}}(X_{2} | X_{1}) \\ 
    \vdots \\ 
    F_{X_{n} | X_{n - 1}, \dots, X_{1}}(X_{n} | X_{n - 1}, \dots, X_{1}) 
\end{bmatrix}
```

  - The second transformation ``\vec{U} = T_{2}^{R}(\vec{Z})`` transforms random vector with *uncorrelated uniform marginal random variables* ``\vec{Z}`` into random vector with *uncorrelated standard normal marginal random variables* ``\vec{U}``.

```math
\vec{U} = T_{2}^{R}(\vec{Z}) = \begin{bmatrix} \Phi^{-1}(Z_{1}) \\ \vdots \\ \Phi^{-1}(Z_{n}) \end{bmatrix}
```

## API

```@docs
RosenblattTransformation
```
