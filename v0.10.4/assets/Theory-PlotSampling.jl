using Fortuna, Random
using CairoMakie, MathTeXEngine
CairoMakie.activate!(; type=:png, px_per_unit=10)

Random.seed!(123)

# --------------------------------------------------
# HISTOGRAM
# --------------------------------------------------
X = randomvariable("Gamma", "M", [10, 1.5])
XSamples = rand(X, 10000, :LHS)

begin
    F = Figure(; size=72 .* (8, 6), fonts=(; regular=texfont()), fontsize=14)

    A = Axis(
        F[1, 1];
        xminorticks=IntervalsBetween(5),
        yminorticks=IntervalsBetween(5),
        xminorticksvisible=true,
        yminorticksvisible=true,
        xminorgridvisible=true,
        yminorgridvisible=true,
        xlabel=L"$x$",
        ylabel="PDF",
        limits=(0, 20, 0, nothing),
        aspect=4 / 3,
    )

    hist!(
        XSamples;
        color=:deepskyblue,
        strokecolor=:black,
        strokewidth=0.25,
        bins=50,
        normalization=:pdf,
    )

    display(F)
end

save("docs/src/assets/Plots (Theory)/Sampling-1.png", F)

# --------------------------------------------------
# SCATTER (UNCORRELATED)
# --------------------------------------------------
X_1 = randomvariable("Gamma", "M", [10, 1.5])
X_2 = randomvariable("Gamma", "M", [15, 2.5])
X = [X_1, X_2]
XSamples = rand(X, 10000, :LHS)

begin
    F = Figure(; size=72 .* (6, 6), fonts=(; regular=texfont()), fontsize=14)

    A = Axis(
        F[1, 1];
        xminorticks=IntervalsBetween(5),
        yminorticks=IntervalsBetween(5),
        xminorticksvisible=true,
        yminorticksvisible=true,
        xgridvisible=true,
        ygridvisible=true,
        xminorgridvisible=true,
        yminorgridvisible=true,
        xlabel=L"$x_{1}$",
        ylabel=L"$x_{2}$",
        limits=(0, 20, 5, 25),
        aspect=1,
    )

    scatter!(
        XSamples[1, :],
        XSamples[2, :];
        color=(:deepskyblue, 0.75),
        strokecolor=(:black, 0.75),
        strokewidth=0.25,
        markersize=6,
    )

    display(F)
end

save("docs/src/assets/Plots (Theory)/Sampling-2.png", F)

# --------------------------------------------------
# SCATTER (CORRELATED)
# --------------------------------------------------
X_1 = randomvariable("Gamma", "M", [10, 1.5])
X_2 = randomvariable("Gamma", "M", [15, 2.5])
X = [X_1, X_2]
ρ_X = [1 -0.75; -0.75 1]
TransformationObject = NatafTransformation(X, ρ_X)
XSamples, ZSamples, USamples = rand(TransformationObject, 10000, :LHS)

begin
    F = Figure(; size=72 .* (6, 6), fonts=(; regular=texfont()), fontsize=14)

    A = Axis(
        F[1, 1];
        xminorticks=IntervalsBetween(5),
        yminorticks=IntervalsBetween(5),
        xminorticksvisible=true,
        yminorticksvisible=true,
        xgridvisible=true,
        ygridvisible=true,
        xminorgridvisible=true,
        yminorgridvisible=true,
        xlabel=L"$x_{1}$",
        ylabel=L"$x_{2}$",
        limits=(0, 20, 5, 25),
        aspect=1,
    )

    scatter!(
        XSamples[1, :],
        XSamples[2, :];
        color=(:deepskyblue, 0.75),
        strokecolor=(:black, 0.75),
        strokewidth=0.25,
        markersize=6,
    )

    display(F)
end

save("docs/src/assets/Plots (Theory)/Sampling-3.png", F)
