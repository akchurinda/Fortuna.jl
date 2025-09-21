# Importance Sampling

## Overview

The Importance Sampling (IS) is a useful Monte Carlo method which allows to estimate the probabilities of rare failure events ``P_{f}`` for reliability problems with both simple and complex limit state functions ``g(\vec{X})``. The IS method is based on the following reformulation of the general analytical expression for the probability of failure ``P_{f}``:

```math
P_{f} = P(\Omega_{f}) = \int_{\Omega_{f}} f_{\vec{X}}(\vec{x}) d\vec{x} = \int_{\mathbb{R}^{n}} \mathbb{I}(\vec{x}) f_{\vec{X}}(\vec{x}) = \int_{\mathbb{R}^{n}} \dfrac{\mathbb{I}(\vec{x}) f_{\vec{X}}(\vec{x})}{q(\vec{x})} q(\vec{x}) d\vec{x} = \mathbb{E}_{q}\left[\dfrac{\mathbb{I}(\vec{x}) f_{\vec{X}}(\vec{x})}{q(\vec{x})}\right]
```

where ``f_{\vec{X}}(\vec{x})`` is the target joint probability density function of the input random vector ``\vec{X}``, ``q(\vec{x})`` is the proposal probability density function, ``\Omega_{f} = \{\vec{X}: g(\vec{X}) \leq 0\}`` is the failure domain defined by the limit state function ``g(\vec{X})``, and ``\mathbb{I}(\vec{x})`` is the indicator function given by:

```math
\mathbb{I}(\vec{x}) = 
\begin{cases}
    1 & \text{if } \vec{x} \in \Omega_{f} \\
    0 & \text{otherwise}
\end{cases}
```

Therefore, the probability of failure ``P_{f}`` is defined as the expectation of ``\mathbb{I}(\vec{x}) f_{\vec{X}}(\vec{x}) / q(\vec{x})`` evaluated with respect to the proposal probability density function ``q(\vec{x})``. If samples of the input random vector ``\vec{x}`` are generated numerically from the proposal probability density function ``q(\vec{x})``, then the estimator of the probability of failure ``\hat{P}_{f}`` is

```math
\hat{P}_{f} = \dfrac{1}{N} \sum_{i = 1}^{N} \dfrac{\mathbb{I}(\vec{x}_{i}) f_{\vec{X}}(\vec{x}_{i})}{q(\vec{x}_{i})}
```

where ``N`` is the number of generated sampled. The estimator ``\hat{P}_{f}`` is unbiased, i.e., it correctly predicts the true probability of failure, such that  ``\mathbb{E}(\hat{P}_{f}) = P_{f}``.

If the proposal probability density function ``q(\vec{x})`` is chosen to be such that has large values in the failure domain ``\Omega_{f}`` (important region), then it is possible to relatively accurately estimate small probability of failure ``P_{f}`` with a small number of samples ``N``. The hard part is, of course, finding a "good" proposal probability density function ``q(\vec{x})``. Typically, it is recommended to use a multivariate normal distribution with uncorrelated marginals centered at the design point ``\vec{x}^{*}``, such that,

```math
q \sim N(\vec{M} = \vec{x}^{*}, \Sigma = \sigma I)
```

## API

```@docs
solve(Problem::ReliabilityProblem, AnalysisMethod::IS)
IS
ISCache
```
