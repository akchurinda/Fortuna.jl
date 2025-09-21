using CairoMakie, MathTeXEngine
using FileIO
CairoMakie.activate!(; type=:png, px_per_unit=10)

begin
    F = Figure(; size=72 .* (8, 6), fonts=(; regular=texfont()), fontsize=16)

    A = Axis(F[1, 1]; limits=(-2, +8 + 2, -3, +6), aspect=4 / 3)

    lines!(A, [0, 0], [0, +3]; color=:black)

    lines!(A, [0, +8], [+3, +3]; color=:black)

    lines!(A, [+8, +8], [+3, +0]; color=:black)

    lines!(A, [-0.5, +0.5], [0, 0]; color=:black)

    lines!(A, [+8 - 0.5, +8 + 0.5], [0, 0]; color=:black)

    band!(A, [-0.5, +0.5], [-0.25, -0.25], [0, 0]; color=(:grey, 0.5))

    band!(A, [+8 - 0.5, +8 + 0.5], [-0.25, -0.25], [0, 0]; color=(:grey, 0.5))

    arrows!(
        A,
        [0],
        [+3],
        [+1],
        [0];
        color=:red,
        arrowhead=BezierPath([
            MoveTo(Point2f(-0.5, -1)), LineTo(0, 0), LineTo(0.5, -1), ClosePath()
        ]),
        align=:tailend,
    )

    for x in 0:8
        arrows!(
            A,
            [x],
            [+3],
            [0],
            [-1];
            color=:red,
            arrowhead=BezierPath([
                MoveTo(Point2f(-0.5, -1)), LineTo(0, 0), LineTo(0.5, -1), ClosePath()
            ]),
            align=:tailend,
        )
    end

    lines!(A, [0, +8], [+3 + 1, +3 + 1]; color=:red)

    text!(A, +0.1, +1.5; text=L"$L_{c} = 3 \text{ m}$", align=(:left, :center))

    text!(A, +8 + 0.1, +1.5; text=L"$L_{c} = 3 \text{ m}$", align=(:left, :center))

    text!(A, +4, +3 - 0.1; text=L"$L_{b} = 8 \text{ m}$", align=(:center, :top))

    text!(A, -1 - 0.1, +3; text=L"$F$", align=(:right, :center))

    text!(A, +4, +4 + 0.1; text=L"$w = 4000 \text{ kN/m}$", align=(:center, :bottom))

    text!(
        A,
        +2,
        -1;
        text=L"$E \sim \text{Normal}(\mu = 210 \text{ GPa}, V = 0.10)$",
        align=(:left, :bottom),
    )

    text!(
        A,
        +2,
        -1.5;
        text=L"$F \sim \text{Normal}(\mu = 2 \text{ kN}, V = 0.10)$",
        align=(:left, :bottom),
    )

    hidedecorations!(A)
    hidespines!(A)

    Inset = Axis(F[1, 1]; height=Relative(0.25), valign=0.40, aspect=DataAspect())

    Fire = load(
        assetpath(
            "/Users/damirakchurin/Desktop/Programming Projects/Fortuna.jl/docs/src/assets/Fire.png",
        ),
    )
    image!(Inset, rotr90(Fire))

    hidedecorations!(Inset)
    hidespines!(Inset)

    display(F)
end

save("docs/src/assets/Examples-AdvancedUsage-2.png", F)
