# Reliability Problems

## Defining Reliability Problems

In general, 3 main "items" are always need to fully define a reliability problem and successfully solve it to find the associated probability of failure ``P_{f}`` and reliability index ``\beta``:

| Item           | Description                                        |
|:-------------- |:-------------------------------------------------- |
| ``\vec{X}``    | Random vector with correlated non-normal marginals |
| ``\rho^{X}``   | Correlation matrix                                 |
| ``g(\vec{X})`` | Limit state function                               |

`Fortuna.jl` package uses these 3 "items" to fully define reliability problems using a custom `ReliabilityProblem()` type as shown in the example below.

```@setup reliability_problems
using Fortuna
```

```@example reliability_problems
# Define random vector:
X_1 = randomvariable("Normal", "M", [10, 2])
X_2 = randomvariable("Normal", "M", [20, 5])
X = [X_1, X_2]

# Define correlation matrix:
ρ_X = [1 0.5; 0.5 1]

# Define limit state function:
g(x::Vector) = x[1] ^ 2 - 2 * x[2]

# Define reliability problem:
problem = ReliabilityProblem(X, ρ_X, g)

nothing # hide
```

!!! note
    
    The definition of the limit state function ``g(\vec{X})`` in `Fortuna.jl` package only pertains to its form (e.g., whether it is linear, square, exponential, etc. in each variable). The information about the random variables involved in the reliability problem is carried in the random vector ``\vec{X}`` and its correlation matrix ``\rho^{X}``, that you use when defining a reliability problem using a custom `ReliabilityProblem()` type.

## Solving Reliability Problems

After defining a reliability problem, `Fortuna.jl` allows to easily solve it using a wide suite of first- and second-order reliability methods using a single `solve()` function as shown in the example below.

```@example reliability_problems
# Perform reliability analysis using improved Hasofer-Lind-Rackwitz-Fiessler (iHLRF) method:
solution = solve(problem, FORM(iHLRF()))
println("β: $(solution.β)")
println("PoF: $(solution.PoF)")
```

Descriptions of all first- and second-order reliability methods implemented in `Fortuna.jl` can be found on [First-Order Reliability Methods](@ref FORMPage) and [Second-Order Reliability Methods](@ref SORMPage) pages.

## API

```@docs
ReliabilityProblem
```
