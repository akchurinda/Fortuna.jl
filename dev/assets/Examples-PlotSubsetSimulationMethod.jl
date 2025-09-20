using Fortuna
using CairoMakie, MathTeXEngine
CairoMakie.activate!(type = :png, px_per_unit = 10)

X_1  = randomvariable("Normal", "M", [0, 1])
X_2  = randomvariable("Normal", "M", [0, 1])
X   = [X_1, X_2]

ρ_X  = [1 0; 0 1]

NatafObject = NatafTransformation(X, ρ_X)

β               = 3
g(x::Vector)    = β * sqrt(2) - x[1] - x[2]

problem = ReliabilityProblem(X, ρ_X, g)
solution = solve(problem, SSM())

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
        limits = (-3, +6, -3, +6),
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
            color = (:black, 0.25))

        scatter!(solution.XSamplesSubset[i][1, 1:100:end], solution.XSamplesSubset[i][2, 1:100:end],
            alpha = 0.5, 
            strokecolor = (:black, 0.5), strokewidth = 0.25,
            markersize = 6)
    end

    text!(4.5, 5.5, text = L"$g(\vec{\mathbf{x}}) \leq 0$",
        color = :black, 
        align = (:center, :center), fontsize = 12)

    text!(4.5, 5.0, text = "(Failure domain)",
        color = :black, 
        align = (:center, :center), fontsize = 12)

    text!(-1.5, -2.0, text = L"$g(\vec{\mathbf{x}}) > 0$",
        color = :black, 
        align = (:center, :center), fontsize = 12)

    text!(-1.5, -2.5, text = "(Safe domain)",
        color = :black, 
        align = (:center, :center), fontsize = 12)

    tooltip!(1.0, 1.0, L"$\Omega_{f_{1}}$",
        offset = 0, outline_linewidth = 1,
        fontsize = 12)

    tooltip!(1.7, 1.7, L"$\Omega_{f_{2}}$",
        offset = 0, outline_linewidth = 1,
        fontsize = 12)

    tooltip!(2.4, 2.4, L"$\Omega_{f_{3}}$",
        offset = 0, outline_linewidth = 1,
        fontsize = 12)

    display(F)
end

save("docs/src/assets/Plots (Examples)/SubsetSimulationMethod-1.png", F)