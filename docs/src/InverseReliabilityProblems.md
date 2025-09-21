# Inverse Reliability Problems

## Defining Inverse Reliability Problems

In general, 4 main "items" are always need to fully define an inverse reliability problem and successfully solve it to find the parameter of interest ``\theta``:

| Item                   | Description                                        |
|:---------------------- |:-------------------------------------------------- |
| ``\vec{X}``            | Random vector with correlated non-normal marginals |
| ``\rho^{X}``           | Correlation matrix                                 |
| ``g(\vec{X}, \theta)`` | Limit state function                               |
| ``\beta``              | Target reliability index                           |

`Fortuna.jl` package uses these 4 "items" to fully define inverse reliability problems of type I using `SensitivityProblem()` type as shown in the example below.

## Solving Inverse Reliability Problems

```@setup inverse_reliability_problem
using Fortuna
```

```@example inverse_reliability_problem
# Define the random vector:
X_1 = randomvariable("Normal", "M", [0, 1])
X_2 = randomvariable("Normal", "M", [0, 1])
X_3 = randomvariable("Normal", "M", [0, 1])
X_4 = randomvariable("Normal", "M", [0, 1])
X = [X_1, X_2, X_3, X_4]

# Define the correlation matrix:
ρ_X = Matrix{Float64}(1.0 * I, 4, 4)

# Define the limit state function:
g(x::Vector, θ::Real) = exp(-θ * (x[1] + 2 * x[2] + 3 * x[3])) - x[4] + 1.5

# Define the target reliability index:
β = 2

# Define an inverse reliability problem:
problem = InverseReliabilityProblem(X, ρ_X, g, β)

nothing # hide
```

## Solving Inverse Reliability Problems

After defining an inverse reliability problem, `Fortuna.jl` allows to easily solve it using a single `solve()` function as shown in the example below.

```@example inverse_reliability_problem
# Perform the inverse reliability analysis:
solution = solve(problem, 0.1; x_0=[0.2, 0.2, 0.2, 0.2])
println("x = $(solution.x[:, end])")
println("θ = $(solution.θ[end])")
```

## API

```@docs
solve(Problem::InverseReliabilityProblem, θ_0::Real)
InverseReliabilityProblem
InverseReliabilityProblemCache
```
