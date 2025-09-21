@testset "Inverse Reliability Problems" begin
    # Example 6.12 (p. 174) from "Structural and System Reliability" book by Armen Der Kiureghian

    # Define the random vector:
    X_1 = randomvariable("Normal", "M", [0, 1])
    X_2 = randomvariable("Normal", "M", [0, 1])
    X_3 = randomvariable("Normal", "M", [0, 1])
    X_4 = randomvariable("Normal", "M", [0, 1])
    X = [X_1, X_2, X_3, X_4]

    # Define the correlation matrix:
    ρ_X = Matrix{Float64}(1.0 * I, 4, 4)

    # Define the limit state function:
    g(x::Vector, θ::Real) = exp(-θ * (x[1] + 2 * x[2] + 3 * x[3])) - x[4] + 1.5

    # Define the target reliability index:
    β = 2

    # Define an inverse reliability problem:
    problem = InverseReliabilityProblem(X, ρ_X, g, β)

    # Perform the inverse reliability analysis:
    solution = solve(problem, 0.1, x_0=[0.2, 0.2, 0.2, 0.2])

    # Test the results:
    @test isapprox(solution.x[:, end], [+0.218, +0.436, +0.655, +1.826], atol=1E-3)
    @test isapprox(solution.θ[end], 0.367, atol=1E-3)
end
