---
title: 'Fortuna.jl: Structural and System Reliability Analysis in Julia'
tags:
    - Julia
    - Reliability Analysis
    - Uncertainty Quantification
authors:
    - name: Damir Akchurin
      orcid: 0000-0003-0849-4917
      affiliation: 1
      corresponding: true
affiliations:
    - name: Department of Civil and Systems Engineering, Johns Hopkins University, Baltimore, MD
      index: 1
date: 1 July 2024
bibliography: paper.bib
---

# Summary

In any field of engineering, addressing uncertainties is crucial to ensure the adequate reliability - the ability to function without failure – of structures and systems, such as residential and commercial buildings, bridges, nuclear power plants, energy distribution networks, transportation networks, and even electric circuits. In these systems, uncertainties are ubiquitous and present in their geometric and material properties, the demands acting on them, and the measurements used to assess their condition. Moreover, uncertainties can also be present in the probabilistic models used to quantify them. As a result of the irreducibility of some of these uncertainties, it is impossible to guarantee the absolute reliability and serviceability of a system at any given point in time with certainty. However, it is possible to define and quantify measures of reliability, such as the failure probability of a system under given demands, using probabilistic methods developed within the field of structural and system reliability.

`Fortuna.jl` is an open-source general-purpose package for structural and system reliability analysis purely written in the Julia programming language [@Bezanson:2017]. It implements a wide suite of commonly used reliability analysis methods to solve reliability, inverse reliability, and sensitivity problems. `Fortuna.jl` leverages Julia’s high-performance capabilities to efficiently solve even the most challenging reliability problems. At the same time, `Fortuna.jl` is designed to be user-friendly and flexible, making it suitable for both research and teaching settings. It is intended that `Fortuna.jl` can serve as a platform for the development and implementation of new rapidly emerging reliability analysis methods.

# Statement of Need

Due to the largely numerical nature of almost all reliability analysis methods, which often require computation of gradient vectors and Hessian matrices, the use of computational resources is necessary for all but the most trivial problems. As a result, a large number of both open-source and commercial software packages have been developed in different programming languages over the past 30 years, such as `UQpy` (Python) [@Olivier:2020], `Pystra` (Python), `FERUM` (MATLAB) [@Bourinet:2009], and `CalREL` (FORTRAN) [@DerKiureghian:2006].

The development of `Fortuna.jl` was mainly motivated by the absence of packages for structural and system reliability analysis written in Julia. Additionally, it aimed to achieve an improved balance between user experience, ease of implementing new reliability analysis methods, computational efficiency, and interoperability with external finite element (FE) modeling software not directly available through Julia.

A key distinguishing feature of `Fortuna.jl` is that it is capable of performing differentiation of the limit state function, which defines the failure criterion for a given reliability problem, required by most reliability analysis methods using automatic differentiation techniques provided by `ForwardDiff.jl` package [@Revels:2016], significantly speeding up the analysis process. This contrasts with other software packages for structural and system reliability analysis, which typically rely on variants of finite difference approximations. As it was mentioned before, `Fortuna.jl` also allows users to easily define limit state functions using external FE modeling software, such as `Abaqus`, `OpenSees` [@McKenna:2010], and `SAFIR` [@Franssen:2017]. `Fortuna.jl` can also easily work with surrogate models of limit state functions that are computationally expensive to evaluate, such as when a limit state function is defined in terms of a complex FE model, using the functionality provided by `Surrogates.jl` package [@Surrogates:2024]. Lastly, `Fortuna.jl`’s ability to work with random variables is based on a widely adopted `Distributions.jl` package [@Besancon:2021], enabling seamless integration with other software packages for probabilistic analysis written in Julia, such as `Copulas.jl` [@Laverny:2024], `RxInter.jl` [@Bagaev:2023], and `Turing.jl` [@Ge:2018], which provide all the functionality needed for the development and implementation of more modern reliability analysis methods based on Bayesian inference.

`Fortuna.jl` has already been successfully used to compare the levels of safety achieved by the Allowable Strength Design (ASD) and Load and Resistance Factor Design (LRFD) methods, two design frameworks used to design individual component in structures [@Sabelli:2024; @Akchurin:2024]. `Fortuna.jl` is also being actively used to develop and implement a next-generation design framework that can account for and optimize system behavior in the design of steel structures, which cannot be achieved using the current methods such as the ASD and LRFD.

# Example

Consider a simple reliability problem from @Echard:2013 with the limit state function given by
\begin{equation}
g(U_1,U_2) = \frac{1}{2} (U_1 - 2) ^ 2 - \frac{3}{2} (U_2 - 5) ^ 3 - 3,
\label{EQ:LimitStateFunction}
\end{equation}
where $U_1$ and $U_2$ are two independent standard normal random variables. The failure domain defined by this limit state function is shown in \autoref{FIG:FailureDomain}. The reference geometric reliability index $\beta$ and failure probability $P_f$ obtained using the first-order reliability method (FORM) are $3.93$ and $4.21 \times 10 ^ {-5}$, respectively. These results can be easily recreated using `Fortuna.jl`:

```julia
# Preamble:
using Fortuna

# Define the random vector and its correlation matrix:
U = [randomvariable("Normal", "M", [0, 1]), randomvariable("Normal", "M", [0, 1])]
ρ = [1 0; 0 1]

# Define the limit state function:
g(u::AbstractVector) = 0.5 * (u[1] - 2) ^ 2 - 1.5 * (u[2] - 5) ^ 3 - 3

# Define the reliability problem and solve it using the FORM:
Problem = ReliabilityProblem(U, ρ, g)
Solution = solve(Problem, FORM())
println("Geometric reliability index: ", Solution.β)
println("Failure probability: ", Solution.PoF)
# Geometric reliability index: 3.932419
# Failure probability: 4.204761E-5
```

As shown in the code above, the results obtained using `Fortuna.jl` are consistent with the reference values.

![Failure domain defined by the limit state function in \autoref{EQ:LimitStateFunction}[^*]. \label{FIG:FailureDomain}](Example.pdf){ width=62.5% }

[^*]: The figure is created using `Makie.jl` package [@Danisch:2021].

# Acknowledgements

The author would like to thank academic and industrial partners of the “*Reliability 2030: Design of Steel as a System*” initiative for the financial support. The author also like to thank the American Institute of Steel Construction for the financial support of the "*System Reliability for Structural Steel*" project.

# References
