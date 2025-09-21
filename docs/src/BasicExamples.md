# Examples

```@setup 1
using Fortuna
using Distributions
using Random
Random.seed!(123)
```

## Isoprobabilistic Transformations

### Nataf Transformation

```@example 1
# Define random vector:
X_1 = randomvariable("Gamma", "M", [10, 1.5])
X_2 = randomvariable("Gamma", "M", [15, 2.5])
X = [X_1, X_2]

# Define correlation matrix:
ρ_X = [1 -0.75; -0.75 1]

# Perform Nataf Transformation:
nataf_obj = NatafTransformation(X, ρ_X)

# Generate 10000 samples of random vector in X-, Z-, and U-spaces using Latin Hypercube Sampling technique:
X_samples, U_samples, Z_samples = rand(nataf_obj, 10000, :LHS)

nothing # hide
```

```@raw html
<img src="../assets/Examples-NatafTransformation-1.png" class="center" style="max-height:350px; border-radius:2.5px;"/>
```

```@example 1
# Compute joint PDF of random vector:
x_range_1 = range(0, 20, 500)
x_range_2 = range(5, 25, 500)
f_samples = [pdf(nataf_obj, [x_1, x_2]) for x_1 in x_range_1, x_2 in x_range_2]

nothing # hide
```

```@raw html
<img src="../assets/Examples-NatafTransformation-2.png" class="center" style="max-height:350px; border-radius:2.5px;"/>
```

## Monte Carlo Methods

### Direct Monte Carlo Simulations

```@example 1
# Define random vector:
X_1 = randomvariable("Normal", "M", [0, 1])
X_2 = randomvariable("Normal", "M", [0, 1])
X = [X_1, X_2]

# Define correlation matrix:
ρ_X = [1 0; 0 1]

# Define limit state function:
β = 3
g(x::Vector) = β * sqrt(2) - x[1] - x[2]

# Define reliability problem:
problem = ReliabilityProblem(X, ρ_X, g)

# Perform reliability analysis using Monte Carlo simulations:
solution = solve(problem, MC())
println("MC:")
println("PoF: $(solution.PoF)")
```

```@raw html
<img src="../assets/Examples-MonteCarlo-1.png" class="center" style="max-height:350px; border-radius:2.5px;"/>
```

### Importance Sampling Method

```@example 1
# Define proposal probability density function:
q = MvNormal([β / sqrt(2), β / sqrt(2)], [1 0; 0 1])

# Perform reliability analysis using Monte Carlo simulations:
solution = solve(problem, IS(; q=q))
println("IS:")
println("PoF: $(solution.PoF)")
```

```@raw html
<img src="../assets/Examples-ImportanceSampling-1.png" class="center" style="max-height:350px; border-radius:2.5px;"/>
```

```@raw html
<img src="../assets/Examples-ImportanceSampling-2.png" class="center" style="max-height:350px; border-radius:2.5px;"/>
```

## First-Order Reliability Methods

### Mean-Centered First-Order Second-Moment Method

```@example 1
# Define random vector:
X_1 = randomvariable("Normal", "M", [10, 2])
X_2 = randomvariable("Normal", "M", [20, 5])
X = [X_1, X_2]

# Define correlation matrix:
ρ_X = [1 0.5; 0.5 1]

# Define two equivalent limit state functions:
g_1(x::Vector) = x[1] ^ 2 - 2 * x[2]
g_2(x::Vector) = 1 - 2 * x[2] / x[1] ^ 2

# Define reliability problems:
problem_1 = ReliabilityProblem(X, ρ_X, g_1)
problem_2 = ReliabilityProblem(X, ρ_X, g_2)

# Perform reliability analysis using Mean-Centered First-Order Second-Moment (MCFOSM) method:
solution_1 = solve(problem_1, FORM(MCFOSM()))
solution_2 = solve(problem_2, FORM(MCFOSM()))
println("MCFOSM:")
println("β from g₁: $(solution_1.β)")
println("β from g₂: $(solution_2.β)")
```

### Hasofer-Lind-Rackwitz-Fiessler Method

```@example 1
# Perform reliability analysis using Hasofer-Lind-Rackwitz-Fiessler (HLRF) method:
solution_1 = solve(problem_1, FORM(HLRF()))
solution_2 = solve(problem_2, FORM(HLRF()))
println("FORM:")
println("β from g₁: $(solution_1.β)")
println("β from g₂: $(solution_2.β)")
```

### Improved Hasofer-Lind-Rackwitz-Fiessler Method

```@example 1
# Perform reliability analysis using improved Hasofer-Lind-Rackwitz-Fiessler (iHLRF) method:
solution_1 = solve(problem_1, FORM(iHLRF()))
solution_2 = solve(problem_2, FORM(iHLRF()))
println("FORM:")
println("β from g₁: $(solution_1.β)")
println("β from g₂: $(solution_2.β)")
```

## Second-Order Reliability Methods

### Curve-Fitting Method

```@example 1
# Define random vector:
M_1 = randomvariable("Normal", "M", [250, 250 * 0.3])
M_2 = randomvariable("Normal", "M", [125, 125 * 0.3])
P = randomvariable("Gumbel", "M", [2500, 2500 * 0.2])
Y = randomvariable("Weibull", "M", [40000, 40000 * 0.1])
X = [M_1, M_2, P, Y]

# Define correlation matrix:
ρ_X = [1 0.5 0.3 0; 0.5 1 0.3 0; 0.3 0.3 1 0; 0 0 0 1]

# Define limit state function:
a = 0.190
s_1 = 0.030
s_2 = 0.015
g(x::Vector) = 1 - x[1] / (s_1 * x[4]) - x[2] / (s_2 * x[4]) - (x[3] / (a * x[4])) ^ 2

# Define reliability problem:
problem = ReliabilityProblem(X, ρ_X, g)

# Perform reliability analysis using Curve-Fitting (CF) method:
solution = solve(problem, SORM(CF()))
println("SORM:")
println("β from FORM: $(solution.form_solution.β)")
println("β from SORM: $(solution.β_2[1]) (Hohenbichler and Rackwitz)")
println("β from SORM: $(solution.β_2[2]) (Breitung)")
println("PoF from FORM: $(solution.form_solution.PoF)")
println("PoF from SORM: $(solution.PoF_2[1]) (Hohenbichler and Rackwitz)")
println("PoF from SORM: $(solution.PoF_2[2]) (Breitung)")
```

### Point-Fitting Method

```@example 1
# Perform reliability analysis using point-fitting SORM:
solution = solve(problem, SORM(PF()))
println("SORM:")
println("β from FORM: $(solution.form_solution.β)")
println("β from SORM: $(solution.β_2[1]) (Hohenbichler and Rackwitz)")
println("β from SORM: $(solution.β_2[2]) (Breitung)")
println("PoF from FORM: $(solution.form_solution.PoF)")
println("PoF from SORM: $(solution.PoF_2[1]) (Hohenbichler and Rackwitz)")
println("PoF from SORM: $(solution.PoF_2[2]) (Breitung)")
```

## Subset Simulation Method

```@example 1
# Define random vector:
X_1 = randomvariable("Normal", "M", [0, 1])
X_2 = randomvariable("Normal", "M", [0, 1])
X = [X_1, X_2]

# Define correlation matrix:
ρ_X = [1 0; 0 1]

# Define limit state function:
β = 3
g(x::Vector) = β * sqrt(2) - x[1] - x[2]

# Define reliability problem:
problem = ReliabilityProblem(X, ρ_X, g)

# Perform reliability analysis using Monte Carlo simulations:
solution = solve(problem, SSM())
println("SSM:")
println("PoF: $(solution.PoF)")
```

```@raw html
<img src="../assets/Examples-SubsetSimulationMethod-1.png" class="center" style="max-height:350px; border-radius:2.5px;"/>
```
