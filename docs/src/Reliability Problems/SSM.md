# Subset Simulation Method

## Overview

The Subset Simulation method (SSM) is a robust simulation technique that transforms a rare event into a sequence of multiple intermediate failure events with larger probabilities and efficiently approximates the probability of the mentioned rare event. That is, the failure event ``\Omega_{f} = \{\vec{X}: g(\vec{X}) \leq 0\}`` is expressed as a union of ``M`` nested intermediate events ``\Omega_{f_{1}}``, ``\dots``, ``\Omega_{f_{M}}``, such that ``\Omega_{f_{M}} \subset \dots \subset \Omega_{f_{1}}`` and ``\Omega_{f} = \cap_{i = 1}^{M} \Omega_{f_{i}}``. The intermediate failure events are defined as ``\Omega_{f_{i}} = \{\vec{X}: g(\vec{X}) \leq b_{i}\}``, where ``b_{1} > \dots > b_{M} = 0`` are non-negative thresholds selected such that each conditional probability ``P(\Omega_{f_{i + 1}} | \Omega_{f_{i}})`` equals a target conditional probability ``P_{0}``.

```math
P_{f} = P(\Omega_{f}) = P(\cap_{i = 1}^{M} \Omega_{f_{i}}) = P(\Omega_{f_{1}}) \prod_{i = 1}^{M - 1} P(\Omega_{f_{i + 1}} | \Omega_{f_{i}})
```

The threshold for the first failure event ``b_{1}`` is computed using the Monte Carlo simulations. The thresholds for the following intermediate failure events ``b_{i}`` are computed using the Monte Carlo Markov Chain samples generated from the conditional probability density functions ``f_{\vec{X}}(\vec{x} | \Omega_{f_{i}})``.

Current implementation of the SSM follows [Au:2001](@citet).

## API

```@docs
solve(Problem::ReliabilityProblem, AnalysisMethod::SSM)
SSM
SSMCache
```
