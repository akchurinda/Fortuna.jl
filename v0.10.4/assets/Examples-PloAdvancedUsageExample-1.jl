using CairoMakie, MathTeXEngine
CairoMakie.activate!(; type=:png, px_per_unit=10)

begin
    F = Figure(; size=72 .* (8, 6), fonts=(; regular=texfont()), fontsize=16)

    A = Axis(F[1, 1]; limits=(-50, +250, -300 / 4 * 3 / 2, +300 / 4 * 3 / 2), aspect=4 / 3)

    lines!(A, [0, +180], [0, 0]; color=:black)

    lines!(A, [0, 0], [-20, +20]; color=:black)

    band!(A, [-20, 0], [-20, -20], [+20, +20]; color=(:grey, 0.5))

    arrows!(
        A,
        [+180],
        [0],
        [0],
        [-20];
        color=:red,
        arrowhead=BezierPath([
            MoveTo(Point2f(-0.5, -1)), LineTo(0, 0), LineTo(0.5, -1), ClosePath()
        ]),
        align=:tailend,
    )

    arrows!(
        A,
        [+180],
        [0],
        [-20],
        [0];
        color=:red,
        arrowhead=BezierPath([
            MoveTo(Point2f(-0.5, -1)), LineTo(0, 0), LineTo(0.5, -1), ClosePath()
        ]),
        align=:tailend,
    )

    text!(A, +90, 0; text=L"$L = 180 \text{ in.}$", align=(:center, :bottom))

    text!(A, +180, +20; text=L"$H = 1 \text{ kip}$", align=(:center, :bottom))

    text!(A, +200, 0; text=L"$P = 50 \text{ kip}$", align=(:left, :center))

    text!(A, +50, -50; text=L"$A = 9.12 \text{ in.}^2$", align=(:left, :bottom))

    text!(
        A,
        +50,
        -65;
        text=L"$E \sim \text{Normal}(\mu = 29000 \text{ ksi}, V = 0.05)$",
        align=(:left, :bottom),
    )

    text!(
        A,
        +50,
        -80;
        text=L"$I \sim \text{Normal}(\mu = 110 \text{ in.}^{4}, V = 0.05)$",
        align=(:left, :bottom),
    )

    hidedecorations!(A)
    hidespines!(A)

    display(F)
end

save("docs/src/assets/Examples-AdvancedUsage-1.png", F)
