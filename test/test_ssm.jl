@testset "Reliability Analysis: SSM #1" begin
    # Set an RNG seed:
    Random.seed!(123)

    # Define a list of reliability indices of interest:
    β_list = 1:6

    for i in eachindex(β_list)
        # Define a random vector of uncorrelated marginal distributions:
        X_1 = randomvariable("Normal", "M", [0, 1])
        X_2 = randomvariable("Normal", "M", [0, 1])
        X  = [X_1, X_2]
        ρ_X = [1 0; 0 1]

        # Define a limit state function:
        g(x::Vector) = β_list[i] * sqrt(2) - x[1] - x[2]

        # Define reliability problems:
        problem = ReliabilityProblem(X, ρ_X, g)

        # Perform the reliability analysis using SSM:
        solution = solve(problem, SSM())

        # Test the results:
        @test isapprox(solution.PoF, cdf(Normal(), -β_list[i]), rtol = 0.10)
    end
end

@testset "Reliability Analysis: SSM #2" begin
    # https://www.researchgate.net/publication/370230768_Structural_reliability_analysis_by_line_sampling_A_Bayesian_active_learning_treatment

    # Set an RNG seed:
    Random.seed!(123)

    # Define random vector:
    X_1 = randomvariable("Normal", "M", [0, 1])
    X_2 = randomvariable("Normal", "M", [0, 1])
    X  = [X_1, X_2]
    ρ_X = [1 0; 0 1]

    # Define limit state function:
    a = 5.50
    b = 0.02
    c = 5 / 6
    d = π / 3
    g(x::Vector) = a - x[2] + b * x[1] ^ 3 + c * sin(d * x[1])

    # Define reliability problem:
    problem = ReliabilityProblem(X, ρ_X, g)

    # Solve reliability problem using Subset Simulation Method:
    solution = solve(problem, SSM())

    # Test the results:
    @test isapprox(solution.PoF, 3.53 * 10 ^ (-7), rtol = 0.10)
end