# https://www.researchgate.net/publication/370230768_Structural_reliability_analysis_by_line_sampling_A_Bayesian_active_learning_treatment

using Fortuna

# Define random vector:
m = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.05])
k_1 = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.10])
k_2 = randomvariable("LogNormal", "M", [0.2, 0.2 * 0.10])
r = randomvariable("LogNormal", "M", [0.5, 0.5 * 0.10])
F_1 = randomvariable("LogNormal", "M", [0.4, 0.4 * 0.20])
t_1 = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.20])
X = [m, k_1, k_2, r, F_1, t_1]

# Define correlation matrix:
ρ_X = Matrix(1.0 * I, 6, 6)

# Define limit state function:
function g(x::Vector)
    3 * x[4] -
    abs(((2 * x[5]) / (x[2] + x[3])) * sin((x[6] / 2) * sqrt((x[2] + x[3]) / x[1])))
end

# Define reliability problem:
problem = ReliabilityProblem(X, ρ_X, g)

# Perform the reliability analysis using FORM:
solution = solve(problem, FORM())
println("β   = $(solution.β)  ")
println("PoF = $(solution.PoF)")
