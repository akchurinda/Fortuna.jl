# Defining Random Variables

`Fortuna.jl` package builds its capacity to define random variables using `randomvariable()` wrapper function by utilizing the widely-adopted [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) package, enabling seamless integration with other probabilistic programming Julia packages such as [`Copulas.jl`](https://github.com/lrnv/Copulas.jl), [`Turing.jl`](https://github.com/TuringLang/Turing.jl) and [`RxInfer.jl`](https://github.com/biaslab/RxInfer.jl). However, unlike [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) package, `Fortuna.jl` package allows to define random variables not only using their *parameters*, but also using their *moments*, which often useful in the field of Structural and System Reliability Analysis.

```@setup generate_rv
using Fortuna
```

## Defining Random Variables Using Moments

To define a random variable using its moments, pass `"M"` as the second argument of `randomvariable()` function followed by the moments themselves:

```@example generate_rv
# Define a lognormally distributed random variable R with 
# mean (μ) of 15 and standard deviation (σ) of 10:
R = randomvariable("LogNormal", "M", [15, 10])
println("μ: $(mean(R))")
println("σ: $(std(R))")
```

## Defining Random Variables Using Parameters

To define a random variable using its parameters pass, `"P"` as the second argument of `randomvariable()` function followed by the parameters themselves:

```@example generate_rv
# Define a gamma-distributed random variable Q with 
# shape parameter (α) of 16 and scale parameter (θ) of 0.625:
Q = randomvariable("Gamma", "P", [16, 0.625])
println("α: $(params(Q)[1])")
println("θ: $(params(Q)[2])")
```

## Supported Random Variables

!!! note
    If you want to define a random variable that is not supported by `Fortuna.jl` package, please raise an issue on the [Github Issues](https://github.com/akchurinda/Fortuna.jl/issues) page.

`Fortuna.jl` package currently supports the following distributions:
- [Exponential](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Exponential)
- [Frechet](https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Frechet)
- [Gamma](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Gamma)
- [Gumbel](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Gumbel)
- [LogNormal](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.LogNormal)
- [Normal](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Normal)
- [Poisson](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Poisson)
- [Uniform](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Uniform)
- [Weibull](https://juliastats.org/Distributions.jl/latest/univariate/#Distributions.Weibull)

## API

```@docs
randomvariable(Distribution::AbstractString, DefineBy::AbstractString, Values::Union{Real, AbstractVector{<:Real}})
```
