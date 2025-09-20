# https://www.researchgate.net/publication/370230768_Structural_reliability_analysis_by_line_sampling_A_Bayesian_active_learning_treatment

using Fortuna
using CairoMakie, MathTeXEngine
CairoMakie.activate!(type = :svg)

# Define random vector:
X_1  = randomvariable("Normal", "M", [0, 1])
X_2  = randomvariable("Normal", "M", [0, 1])
X   = [X_1, X_2]

# Define correlation matrix:
ρ_X  = [1 0; 0 1]

# Define limit state function:
a = 5.50
b = 0.02
c = 5 / 6
d = π / 3
g(x::Vector) = a - x[2] + b * x[1] ^ 3 + c * sin(d * x[1])

# Define reliability problem:
problem = ReliabilityProblem(X, ρ_X, g)

# Solve reliability problem using Subset Simulation Method:
solution = solve(problem, SSM())

# Plot the solution:
x_1_range = range(-9, +9, 500)
x_2_range = range(-9, +9, 500)
g_samples = [g([x_1, x_2]) for x_1 in x_1_range, x_2 in x_2_range]
begin
    F = Figure(size = 72 .* (6, 6), fonts = (; regular = texfont()), fontsize = 14)

    A = Axis(F[1, 1],
        xlabel = L"$x_{1}$", ylabel = L"$x_{2}$",
        xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5),
        xminorticksvisible = true, yminorticksvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        limits = (minimum(x_1_range), maximum(x_1_range), minimum(x_2_range), maximum(x_2_range)),
        aspect = 1)

    contour!(x_1_range, x_2_range, g_samples,
        levels = [0],
        color = (:black, 0.25))

    contourf!(x_1_range, x_2_range, g_samples,
        levels = [0],
        extendhigh = (:green, 0.25), extendlow = (:red, 0.25))

    for i in eachindex(solution.CSubset)
        contour!(x_1_range, x_2_range, g_samples,
            levels = [solution.CSubset[i]],
            color = (:black, 0.25),
            linestyle = :dash)

        scatter!(solution.XSamplesSubset[i][1, 1:100:end], solution.XSamplesSubset[i][2, 1:100:end],
            alpha = 0.5, 
            strokecolor = (:black, 0.5), strokewidth = 0.25,
            markersize = 6)
    end

    text!(-6.0, 6.0, text = L"$g(\vec{\mathbf{x}}) \leq 0$",
        color = :black, 
        align = (:center, :bottom), fontsize = 12)

    text!(-6.0, 6.0, text = "(Failure domain)",
        color = :black, 
        align = (:center, :top), fontsize = 12)

    text!(6.0, -6.0, text = L"$g(\vec{\mathbf{x}}) > 0$",
        color = :black, 
        align = (:center, :bottom), fontsize = 12)

    text!(6.0, -6.0, text = "(Safe domain)",
        color = :black, 
        align = (:center, :top), fontsize = 12)
    
    display(F)
end