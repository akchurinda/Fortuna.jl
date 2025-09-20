using Fortuna
using CairoMakie, MathTeXEngine
CairoMakie.activate!(type = :png, px_per_unit = 10)

X_1  = randomvariable("Normal", "M", [0, 1])
X_2  = randomvariable("Normal", "M", [0, 1])
X   = [X_1, X_2]

ρ_X = [1 0; 0 1]

NatafObject = NatafTransformation(X, ρ_X)

β               = 3
g(x::Vector)    = β * sqrt(2) - x[1] - x[2]

problem = ReliabilityProblem(X, ρ_X, g)
solution = solve(problem, MC())

x_1_range = range(-3, +6, 500)
x_2_range = range(-3, +6, 500)
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

    contour!(x_1_range, x_2_range, g_samples, label = L"$g(\vec{\mathbf{x}}) = 0$",
        levels = [0],
        color = (:black, 0.25))

    contourf!(x_1_range, x_2_range, g_samples,
        levels = [0],
        extendhigh = (:green, 0.25), extendlow = (:red, 0.25))

    scatter!(solution.Samples[1, 1:100:end], solution.Samples[2, 1:100:end],
        color = (:steelblue, 0.5), 
        strokecolor = (:black, 0.5), strokewidth = 0.25,
        markersize = 6)
    
    text!(4.5, 5.5, text = L"$g(\vec{\mathbf{x}}) \leq 0$",
        color = :black, 
        align = (:center, :center), fontsize = 12)

    text!(4.5, 5.0, text = "(Failure domain)",
        color = :black, 
        align = (:center, :center), fontsize = 12)

    text!(4.5, -2.0, text = L"$g(\vec{\mathbf{x}}) > 0$",
        color = :black, 
        align = (:center, :center), fontsize = 12)

    text!(4.5, -2.5, text = "(Safe domain)",
        color = :black, 
        align = (:center, :center), fontsize = 12)
    
    display(F)
end

save("docs/src/assets/Plots (Examples)/MonteCarlo-1.png", F)