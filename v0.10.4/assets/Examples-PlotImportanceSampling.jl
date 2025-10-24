using Fortuna, Distributions
using CairoMakie, MathTeXEngine
CairoMakie.activate!(; type=:png, px_per_unit=10)

X_1 = randomvariable("Normal", "M", [0, 1])
X_2 = randomvariable("Normal", "M", [0, 1])
X = [X_1, X_2]

ρ_X = [1 0; 0 1]

NatafObject = NatafTransformation(X, ρ_X)

β = 3
g(x::Vector) = β * sqrt(2) - x[1] - x[2]

problem = ReliabilityProblem(X, ρ_X, g)

ProposalPDF = MvNormal([β / sqrt(2), β / sqrt(2)], [1 0; 0 1])
NumSamples = vcat(1000:1000:9000, 10000:10000:90000, 100000:100000:900000, 1000000)
PoFValues = Matrix{Float64}(undef, length(NumSamples), 2)
for i in eachindex(NumSamples)
    display(i)
    PoFValues[i, 1] = solve(problem, MC(NumSamples[i], ITS())).PoF
    PoFValues[i, 2] = solve(problem, IS(ProposalPDF, NumSamples[i])).PoF
end

x_1_range = range(-3, +6, 500)
x_2_range = range(-3, +6, 500)
g_samples = [g([x_1, x_2]) for x_1 in x_1_range, x_2 in x_2_range]

begin
    F = Figure(; size=72 .* (6, 6), fonts=(; regular=texfont()), fontsize=14)

    A = Axis(
        F[1, 1];
        xlabel=L"$x_{1}$",
        ylabel=L"$x_{2}$",
        xminorticks=IntervalsBetween(5),
        yminorticks=IntervalsBetween(5),
        xminorticksvisible=true,
        yminorticksvisible=true,
        xminorgridvisible=true,
        yminorgridvisible=true,
        limits=(
            minimum(x_1_range), maximum(x_1_range), minimum(x_2_range), maximum(x_2_range)
        ),
        aspect=1,
    )

    contour!(
        x_1_range,
        x_2_range,
        g_samples;
        label=L"$g(\vec{\mathbf{x}}) = 0$",
        levels=[0],
        color=(:black, 0.25),
    )

    contourf!(
        x_1_range,
        x_2_range,
        g_samples;
        levels=[0],
        extendhigh=(:green, 0.25),
        extendlow=(:red, 0.25),
    )

    Samples = solve(problem, MC(10000, ITS())).Samples
    scatter!(
        Samples[1, :],
        Samples[2, :];
        label=L"Samples from $f_{\vec{\mathbf{X}}}(\vec{\mathbf{x}})$",
        color=(:steelblue, 0.5),
        strokecolor=(:black, 0.5),
        strokewidth=0.25,
        markersize=6,
    )

    Samples = solve(problem, IS(ProposalPDF, 10000)).Samples
    scatter!(
        Samples[1, :],
        Samples[2, :];
        label=L"Samples from $q(\vec{\mathbf{x}})$",
        color=(:crimson, 0.5),
        strokecolor=(:black, 0.5),
        strokewidth=0.25,
        markersize=6,
    )

    text!(
        4.5,
        5.5;
        text=L"$g(\vec{\mathbf{x}}) \leq 0$",
        color=:black,
        align=(:center, :center),
        fontsize=12,
    )

    text!(
        4.5,
        5.0;
        text="(Failure domain)",
        color=:black,
        align=(:center, :center),
        fontsize=12,
    )

    text!(
        4.5,
        -2.0;
        text=L"$g(\vec{\mathbf{x}}) > 0$",
        color=:black,
        align=(:center, :center),
        fontsize=12,
    )

    text!(
        4.5, -2.5; text="(Safe domain)", color=:black, align=(:center, :center), fontsize=12
    )

    axislegend(; position=:lt, framevisible=false, rowgap=0, merge=true, fontsize=12)

    display(F)
end

save("docs/src/assets/Plots (Examples)/ImportanceSampling-1.png", F)

PoF = cdf(Normal(), -β)
begin
    F = Figure(; size=72 .* (8, 6), fonts=(; regular=texfont()), fontsize=14)

    A = Axis(
        F[1, 1];
        xlabel=L"$N$",
        ylabel=L"$P_{f}$",
        xminorticks=IntervalsBetween(9),
        yminorticks=IntervalsBetween(5),
        xminorticksvisible=true,
        yminorticksvisible=true,
        xminorgridvisible=true,
        yminorgridvisible=true,
        xscale=log10,
        limits=(NumSamples[1], NumSamples[end], 0, 2 * PoF),
        aspect=4 / 3,
    )

    hlines!(
        cdf(Normal(), -β); label="Exact solution", color=:grey, linestyle=:dash, linewidth=1
    )

    scatterlines!(
        NumSamples,
        PoFValues[:, 1];
        label="Direct Monte Carlo Simulations",
        color=:steelblue,
        linestyle=:solid,
        linewidth=1,
        markersize=6,
    )

    scatterlines!(
        NumSamples,
        PoFValues[:, 2];
        label="Importance Sampling Method",
        color=:crimson,
        linestyle=:solid,
        linewidth=1,
        markersize=6,
    )

    axislegend(; position=:rt, framevisible=false, rowgap=0, merge=true, fontsize=12)

    display(F)
end

save("docs/src/assets/Plots (Examples)/ImportanceSampling-2.png", F)
