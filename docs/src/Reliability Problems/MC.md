# Monte Carlo Simulations

## Overview

Using direct Monte Carlo simulations (MCS) is the simplest way to estimate the probability of failure ``P_{f}`` for reliability problems with both simple and complex limit state functions ``g(\vec{X})``. The brute force MCS relies on the following reformulation of the general analytical expression for the probability of failure ``P_{f}``:

```math
P_{f} = P(\Omega_{f}) = \int_{\Omega_{f}} f_{\vec{X}}(\vec{x}) d\vec{x} = \int_{\mathbb{R}^{n}} \mathbb{I}(\vec{x}) f_{\vec{X}}(\vec{x}) d\vec{x} = \mathbb{E}_{f}[\mathbb{I}(\vec{x})]
```

where ``f_{\vec{X}}(\vec{x})`` is the joint probability function of the input random vector ``\vec{X}``, ``\Omega_{f} = \{\vec{X}: g(\vec{X}) \leq 0\}`` is the failure domain defined by the limit state function ``g(\vec{X})``, and ``\mathbb{I}(\vec{x})`` is the indicator function given by:

```math
\mathbb{I}(\vec{x}) = 
\begin{cases}
    1 & \text{if } \vec{x} \in \Omega_{f} \\
    0 & \text{otherwise}
\end{cases}
```

Therefore, the probability of failure ``P_{f}`` is defined as the expectation of the indicator function ``\mathbb{I}(\vec{x})``. If samples of the input random vector ``\vec{x}`` are generated numerically, then the estimator of the probability of failure ``\hat{P}_{f}`` is

```math
\hat{P}_{f} = \dfrac{1}{N} \sum_{i = 1}^{N} \mathbb{I}(\vec{x}_{i})
```

where ``N`` is the number of generated sampled. The estimator ``\hat{P}_{f}`` is unbiased, i.e., it correctly predicts the true probability of failure, such that  ``\mathbb{E}(\hat{P}_{f}) = P_{f}``. The main drawback of using the MCS is that is becomes prohibitively expensive to use if the true probability is too small, e.g., ``P_{f} \leq 10^{-6}``, given that the variance of the estimator is inversely proportional to the number of generated samples, such that,

```math
\text{Var}(\hat{P}_{f}) = \frac{1}{N} P_{f} (1 - P_{f})
```

!!! tip
    
    For typical structural reliability problems with true probabilities of failure ``P_{f}`` of ``\approx 10^{-3}``, it is recommended to use ``N = 10^{6}`` samples to get the coefficient of variation of the estimator ``V_{P_{f}}`` of ``\approx 0.10``.
    
    ```@raw html
    <img src="../../assets/Theory-MonteCarlo-1.png" class="center" style="max-height:350px; border-radius:2.5px;"/>
    ```

## API

```@docs
solve(Problem::ReliabilityProblem, AnalysisMethod::MC)
MC
MCCache
```
