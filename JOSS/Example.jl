# Preamble:
using Fortuna

# Define the random vector:
U_1 = randomvariable("Normal", "M", [0, 1])
U_2 = randomvariable("Normal", "M", [0, 1])
U = [U_1, U_2]

# Define the correlation matrix:
ρ_U = [1 0; 0 1]

# Define the limit state function:
g(u::Vector) = 0.5 * (u[1] - 2) ^ 2 - 1.5 * (u[2] - 5) ^ 3 - 3

# Define the reliability problem:
problem = ReliabilityProblem(U, ρ, g)

# Solve the reliability problem using the FORM:
solution = solve(problem, FORM())
println("Geometric reliability index: ", solution.β)
println("Failure probability: ", solution.PoF)
# Geometric reliability index: 3.932419
# Failure probability: 4.204761E-5

# Plot the failure domain:
using Colors
using CairoMakie
CairoMakie.activate!(type = :svg)
CairoMakie.set_theme!(theme_latexfonts())

u_1_range = -6:0.01:6
u_2_range = -6:0.01:6

nataf_obj = NatafTransformation(U, ρ)
f_vals = [pdf(nataf_obj, [u_1, u_2]) for u_1 in u_1_range, u_2 in u_2_range]
f_vals = f_vals ./ maximum(f_vals)
g_vals = [g([u_1, u_2]) for u_1 in u_1_range, u_2 in u_2_range]

colors = Colors.JULIA_LOGO_COLORS
begin
    F = Figure(size = 72 .* (6, 6))

    A = Axis(F[1, 1],
        xlabel = L"u_1",
        ylabel = L"u_2",
        xticks = -6:3:6,
        yticks = -6:3:6,
        xminorticks = IntervalsBetween(5),
        yminorticks = IntervalsBetween(5),
        xminorgridvisible = true,
        yminorgridvisible = true,
        limits = (-6, 6, -6, 6),
        aspect = 1)

    contourf!(A, u_1_range, u_2_range, g_vals,
        levels     = [0],
        extendlow  = (colorant"#FF1F5B", 0.30),
        extendhigh = (colorant"#00CD6C", 0.30))

    contour!(A, u_1_range, u_2_range, g_vals,
        levels    = [0],
        color     = :black,
        linewidth = 0.5)

    contourf!(A, u_1_range, u_2_range, f_vals, 
        levels   = 0:0.1:1, 
        colormap = cgrad([:transparent, colorant"#009ADE"]))

    contour!(A, u_1_range, u_2_range, f_vals, 
        levels    = 0:0.1:1, 
        colormap  = cgrad([:transparent, :black]),
        linewidth = 0.5)

    text!(A, (+2, +5), 
        text = L"Failure domain, \\ $g(u_1, u_2) \leq 0$",
        align = (:center, :center))

    text!(A, (-2, -5), 
        text = L"Safe domain, \\ $g(u_1, u_2) > 0$",
        align = (:center, :center))

    text!(A, (+3, -3), 
        text = L"Joint PDF, \\ $f_{\vec{U}}(u_1, u_2)$",
        align = (:center, :center))

    arc!(A, (0, -3), 3, π / 12, π / 4,
        color     = :black,
        linewidth = 0.5)

    display(F)
end

save("joss/example.pdf", F)