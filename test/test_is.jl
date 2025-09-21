@testset "Importance Sampling #1" begin
    # Set an RNG seed:
    Random.seed!(123)

    # Define a list of reliability indices of interest:
    β_list = 1:6

    for i in eachindex(β_list)
        # Define a random vector of correlated marginal distributions:
        X_1 = randomvariable("Normal", "M", [0, 1])
        X_2 = randomvariable("Normal", "M", [0, 1])
        X = [X_1, X_2]
        ρ_X = [1 0; 0 1]

        # Define a limit state function:
        g(x::Vector) = β_list[i] * sqrt(2) - x[1] - x[2]

        # Define a reliability problem:
        problem = ReliabilityProblem(X, ρ_X, g)

        # Perform the reliability analysis using Importance Sampling method:
        q = MvNormal([β_list[i] / sqrt(2), β_list[i] / sqrt(2)], [1 0; 0 1])
        solution = solve(problem, IS(q, 10 ^ 6))

        # Test the results:
        @test isapprox(solution.PoF, cdf(Normal(), -β_list[i]), rtol=5E-2)
    end
end
