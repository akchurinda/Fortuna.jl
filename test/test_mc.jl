@testset "Monte Carlo #1" begin
    # Set an RNG seed:
    Random.seed!(123)

    # Define a list of reliability indices of interest:
    β_list = 1:3

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
        solution = solve(problem, MC())

        # Test the results:
        @test isapprox(solution.PoF, cdf(Normal(), -β_list[i]), rtol=5E-2)
    end
end

@testset "Monte Carlo #2" begin
    # Set an RNG seed:
    Random.seed!(123)

    # Define a random vector of correlated marginal distributions:
    M_1 = randomvariable("Normal", "M", [250, 250 * 0.3])
    M_2 = randomvariable("Normal", "M", [125, 125 * 0.3])
    P = randomvariable("Gumbel", "M", [2500, 2500 * 0.2])
    Y = randomvariable("Weibull", "M", [40000, 40000 * 0.1])
    X = [M_1, M_2, P, Y]
    ρ_X = [1 0.5 0.3 0; 0.5 1 0.3 0; 0.3 0.3 1 0; 0 0 0 1]

    # Define a limit state function:
    a = 0.190
    s_1 = 0.030
    s_2 = 0.015
    g(x::Vector) = 1 - x[1] / (s_1 * x[4]) - x[2] / (s_2 * x[4]) - (x[3] / (a * x[4]))^2

    # Define a reliability problem:
    problem = ReliabilityProblem(X, ρ_X, g)

    # Perform the reliability analysis using Monte Carlo simulations:
    solution = solve(problem, MC())

    # Test the results:
    @test isapprox(solution.PoF, 0.00931, rtol=5E-2)
end
