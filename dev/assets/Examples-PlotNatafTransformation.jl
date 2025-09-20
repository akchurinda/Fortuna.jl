using Fortuna
using CairoMakie, MathTeXEngine
CairoMakie.activate!(type = :png, px_per_unit = 10)

X_1  = randomvariable("Gamma", "M", [10, 1.5])
X_2  = randomvariable("Gamma", "M", [15, 2.5])
X   = [X_1, X_2]

ρ_X = [1 -0.75; -0.75 1]

NatafObject = NatafTransformation(X, ρ_X)

XSamples, ZSamples, USamples = rand(NatafObject, 10000, LHS())

begin
    F = Figure(size = 72 .* (18, 6), fonts = (; regular = texfont()), fontsize = 14)

    A = Axis(F[1, 1], 
        title = L"$X$-space",
        xlabel = L"$x_{1}$", ylabel = L"$x_{2}$",
        xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5),
        xminorticksvisible = true, yminorticksvisible = true,
        xgridvisible = true, ygridvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        limits = (0, 20, 5, 25),
        aspect = 1)
    
    scatter!(XSamples[1, :], XSamples[2, :],
        color = (:crimson, 0.5), 
        strokecolor = (:black, 0.5), strokewidth = 0.25,
        markersize = 6)
    
    A = Axis(F[1, 2], 
        title = L"$Z$-space",
        xlabel = L"$z_{1}$", ylabel = L"$z_{2}$",
        xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5),
        xminorticksvisible = true, yminorticksvisible = true,
        xgridvisible = true, ygridvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        limits = (-6, +6, -6, +6),
        aspect = 1)
    
    scatter!(ZSamples[1, :], ZSamples[2, :],
        color = (:steelblue, 0.5),
        strokecolor = (:black, 0.5), strokewidth = 0.25,
        markersize = 6)

    A = Axis(F[1, 3], 
        title = L"$U$-space",
        xlabel = L"$u_{1}$", ylabel = L"$u_{2}$",
        xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5),
        xminorticksvisible = true, yminorticksvisible = true,
        xgridvisible = true, ygridvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        limits = (-6, +6, -6, +6),
        aspect = 1)
    
    scatter!(USamples[1, :], USamples[2, :],
        color = (:forestgreen, 0.5),
        strokecolor = (:black, 0.5), strokewidth = 0.25,
        markersize = 6)

    display(F)
end

save("docs/src/assets/Plots (Examples)/NatafTransformation-1.png", F)

x_1_range = range(0, 20, 500)
x_2_range = range(5, 25, 500)
fSamples = [pdf(NatafObject, [x_1, x_2]) for x_1 in x_1_range, x_2 in x_2_range]

begin
    F = Figure(size = 72 .* (6, 6), fonts = (; regular = texfont()), fontsize = 14)

    A = Axis(F[1, 1],
        xlabel = L"$x_{1}$", ylabel = L"$x_{2}$",
        xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5),
        xminorticksvisible = true, yminorticksvisible = true,
        xgridvisible = true, ygridvisible = true,
        xminorgridvisible = true, yminorgridvisible = true,
        limits = (0, 20, 5, 25),
        aspect = 1)

    contourf!(x_1_range, x_2_range, fSamples,
        levels = minimum(fSamples) : (maximum(fSamples) - minimum(fSamples)) / 15 : maximum(fSamples),
        colormap = cgrad([:transparent, :deepskyblue]))
    
    contour!(x_1_range, x_2_range, fSamples,
        levels = minimum(fSamples) : (maximum(fSamples) - minimum(fSamples)) / 15 : maximum(fSamples),
        colormap = cgrad([:transparent, :black]),
        linewidth = 0.5)

    display(F)
end

save("docs/src/assets/Plots (Examples)/NatafTransformation-2.png", F)
