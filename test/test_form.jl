@testset "FORM #1 - MCFOSM" begin
    # Example 5.1 (p. 110) from "Structural and System Reliability" book by Armen Der Kiureghian

    # Define a random vector of correlated marginal distributions:
    X_1 = randomvariable("Normal", "M", [10, 2])
    X_2 = randomvariable("Normal", "M", [20, 5])
    X = [X_1, X_2]
    ρ_X = [1 0.5; 0.5 1]

    # Define two equivalent limit state functions to demonstrate the invariance problem of the MCFOSM method:
    g_1(x::Vector) = x[1]^2 - 2 * x[2]
    g_2(x::Vector) = 1 - 2 * x[2] / x[1]^2

    # Define reliability problems:
    problem_1 = ReliabilityProblem(X, ρ_X, g_1)
    problem_2 = ReliabilityProblem(X, ρ_X, g_2)

    # Perform the reliability analysis using MCFOSM:
    solution_1 = solve(problem_1, FORM(MCFOSM()))
    solution_2 = solve(problem_2, FORM(MCFOSM()))

    # Test the results:
    @test isapprox(solution_1.β, 1.66, rtol=1E-2)
    @test isapprox(solution_2.β, 4.29, rtol=1E-2)
end

@testset "FORM #2 - RF" begin
    # Test from UQPy package (https://github.com/SURGroup/UQpy/tree/master)

    # Define a random vector of correlated marginal distributions:
    X_1 = randomvariable("Normal", "M", [200, 20])
    X_2 = randomvariable("Normal", "M", [150, 10])
    X = [X_1, X_2]
    ρ_X = [1 0; 0 1]

    # Define two equivalent limit state functions to demonstrate the invariance problem of the MCFOSM method:
    g(x::Vector) = x[1] - x[2]

    # Define reliability problems:
    problem = ReliabilityProblem(X, ρ_X, g)

    # Perform the reliability analysis using iHLRF:
    solution = solve(problem, FORM(RF()))

    # Test the results:
    @test isapprox(solution.β, 2.236067977499917, rtol=1E-9)
    @test isapprox(solution.x[:, end], [160, 160], rtol=1E-9)
    @test isapprox(solution.u[:, end], [-2, 1], rtol=1E-9)
end

@testset "FORM #3 - HLRF" begin
    # Example 5.2 (p. 118) from "Structural and System Reliability" book by Armen Der Kiureghian

    # Define a random vector of correlated marginal distributions:
    X_1 = randomvariable("Normal", "M", [10, 2])
    X_2 = randomvariable("Normal", "M", [20, 5])
    X = [X_1, X_2]
    ρ_X = [1 0.5; 0.5 1]

    # Define two equivalent limit state functions to demonstrate the invariance problem of the MCFOSM method:
    g_1(x::Vector) = x[1]^2 - 2 * x[2]
    g_2(x::Vector) = 1 - 2 * x[2] / x[1]^2

    # Define reliability problems:
    problem_1 = ReliabilityProblem(X, ρ_X, g_1)
    problem_2 = ReliabilityProblem(X, ρ_X, g_2)

    # Perform the reliability analysis using HLRF:
    solution_1 = solve(problem_1, FORM(HLRF()))
    solution_2 = solve(problem_2, FORM(HLRF()))

    # Test the results:
    @test isapprox(solution_1.β, 2.11, rtol=1E-2)
    @test isapprox(solution_2.β, 2.11, rtol=1E-2)
    @test isapprox(solution_1.x[:, end], [6.14, 18.9], rtol=1E-2)
    @test isapprox(solution_2.x[:, end], [6.14, 18.9], rtol=1E-2)
    @test isapprox(solution_1.u[:, end], [-1.928, 0.852], rtol=1E-2)
    @test isapprox(solution_2.u[:, end], [-1.928, 0.852], rtol=1E-2)
end

@testset "FORM #4 - HLRF" begin
    # Test from UQPy package (https://github.com/SURGroup/UQpy/tree/master)

    # Define a random vector of correlated marginal distributions:
    X_1 = randomvariable("Normal", "M", [200, 20])
    X_2 = randomvariable("Normal", "M", [150, 10])
    X = [X_1, X_2]
    ρ_X = [1 0; 0 1]

    # Define two equivalent limit state functions to demonstrate the invariance problem of the MCFOSM method:
    g(x::Vector) = x[1] - x[2]

    # Define reliability problems:
    problem = ReliabilityProblem(X, ρ_X, g)

    # Perform the reliability analysis using HLRF:
    solution = solve(problem, FORM(HLRF()))

    # Test the results:
    @test isapprox(solution.β, 2.236067977499917, rtol=1E-9)
    @test isapprox(solution.PoF, 0.012673659338729965, rtol=1E-9)
    @test isapprox(solution.x[:, end], [160, 160], rtol=1E-9)
    @test isapprox(solution.u[:, end], [-2, 1], rtol=1E-9)
end

@testset "FORM #5 - HLRF" begin
    # Example 6.5 (p. 147) from "Structural and System Reliability" book by Armen Der Kiureghian

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

    # Perform the reliability analysis using HLRF:
    solution = solve(problem, FORM(HLRF()))

    # Test the results:
    @test isapprox(solution.β, 2.47, rtol=1E-2)
    @test isapprox(solution.PoF, 0.00682, rtol=1E-2)
    @test isapprox(solution.x[:, end], [341, 170, 3223, 31770], rtol=1E-2)
    @test isapprox(solution.u[:, end], [1.210, 0.699, 0.941, -1.80], rtol=1E-2)
    @test isapprox(solution.γ, [0.269, 0.269, 0.451, -0.808], rtol=1E-2)
    # Note: There is a typo in the book for this example. The last coordinate of the design point in U-space must be -1.80.
end

@testset "FORM #6 - HLRF" begin
    # https://www.researchgate.net/publication/370230768_Structural_reliability_analysis_by_line_sampling_A_Bayesian_active_learning_treatment

    # Define random vector:
    m = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.05])
    k_1 = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.10])
    k_2 = randomvariable("LogNormal", "M", [0.2, 0.2 * 0.10])
    r = randomvariable("LogNormal", "M", [0.5, 0.5 * 0.10])
    F_1 = randomvariable("LogNormal", "M", [0.4, 0.4 * 0.20])
    t_1 = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.20])
    X = [m, k_1, k_2, r, F_1, t_1]
    ρ_X = Matrix(1.0 * I, 6, 6)

    # Define limit state function:
    g(x::Vector) =
        3 * x[4] -
        abs(((2 * x[5]) / (x[2] + x[3])) * sin((x[6] / 2) * sqrt((x[2] + x[3]) / x[1])))

    # Define reliability problem:
    problem = ReliabilityProblem(X, ρ_X, g)

    # Perform the reliability analysis using HLRF:
    solution = solve(problem, FORM(HLRF()))

    # Test the results:
    @test isapprox(solution.PoF, 4.88 * 10 ^ (-8), rtol=1E-2)
end

@testset "FORM #7 - iHLRF" begin
    # Example 5.2 (p. 118) from "Structural and System Reliability" book by Armen Der Kiureghian

    # Define a random vector of correlated marginal distributions:
    X_1 = randomvariable("Normal", "M", [10, 2])
    X_2 = randomvariable("Normal", "M", [20, 5])
    X = [X_1, X_2]
    ρ_X = [1 0.5; 0.5 1]

    # Define two equivalent limit state functions to demonstrate the invariance problem of the MCFOSM method:
    g_1(x::Vector) = x[1]^2 - 2 * x[2]
    g_2(x::Vector) = 1 - 2 * x[2] / x[1]^2

    # Define reliability problems:
    problem_1 = ReliabilityProblem(X, ρ_X, g_1)
    problem_2 = ReliabilityProblem(X, ρ_X, g_2)

    # Perform the reliability analysis using iHLRF:
    solution_1 = solve(problem_1, FORM(iHLRF()))
    solution_2 = solve(problem_2, FORM(iHLRF()))

    # Test the results:
    @test isapprox(solution_1.β, 2.11, rtol=1E-2)
    @test isapprox(solution_2.β, 2.11, rtol=1E-2)
    @test isapprox(solution_1.x[:, end], [6.14, 18.9], rtol=1E-2)
    @test isapprox(solution_2.x[:, end], [6.14, 18.9], rtol=1E-2)
    @test isapprox(solution_1.u[:, end], [-1.928, 0.852], rtol=1E-2)
    @test isapprox(solution_2.u[:, end], [-1.928, 0.852], rtol=1E-2)
end

@testset "FORM #8 - iHLRF" begin
    # Test from UQPy package (https://github.com/SURGroup/UQpy/tree/master)

    # Define a random vector of correlated marginal distributions:
    X_1 = randomvariable("Normal", "M", [200, 20])
    X_2 = randomvariable("Normal", "M", [150, 10])
    X = [X_1, X_2]
    ρ_X = [1 0; 0 1]

    # Define two equivalent limit state functions to demonstrate the invariance problem of the MCFOSM method:
    g(x::Vector) = x[1] - x[2]

    # Define reliability problems:
    problem = ReliabilityProblem(X, ρ_X, g)

    # Perform the reliability analysis using iHLRF:
    solution = solve(problem, FORM(iHLRF()))

    # Test the results:
    @test isapprox(solution.β, 2.236067977499917, rtol=1E-9)
    @test isapprox(solution.PoF, 0.012673659338729965, rtol=1E-9)
    @test isapprox(solution.x[:, end], [160, 160], rtol=1E-9)
    @test isapprox(solution.u[:, end], [-2, 1], rtol=1E-9)
end

@testset "FORM #9 - iHLRF" begin
    # Example 6.5 (p. 147) from "Structural and System Reliability" book by Armen Der Kiureghian

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

    # Perform the reliability analysis using iHLRF:
    solution = solve(problem, FORM(iHLRF()))

    # Test the results:
    @test isapprox(solution.β, 2.47, rtol=1E-2)
    @test isapprox(solution.PoF, 0.00682, rtol=1E-2)
    @test isapprox(solution.x[:, end], [341, 170, 3223, 31770], rtol=1E-2)
    @test isapprox(solution.u[:, end], [1.210, 0.699, 0.941, -1.80], rtol=1E-2)
    @test isapprox(solution.γ, [0.269, 0.269, 0.451, -0.808], rtol=1E-2)
    # Note: There is a typo in the book for this example. The last coordinate of the design point in U-space must be -1.80.
end

@testset "FORM #10 - iHLRF" begin
    # https://www.researchgate.net/publication/370230768_Structural_reliability_analysis_by_line_sampling_A_Bayesian_active_learning_treatment

    # Define random vector:
    m = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.05])
    k_1 = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.10])
    k_2 = randomvariable("LogNormal", "M", [0.2, 0.2 * 0.10])
    r = randomvariable("LogNormal", "M", [0.5, 0.5 * 0.10])
    F_1 = randomvariable("LogNormal", "M", [0.4, 0.4 * 0.20])
    t_1 = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.20])
    X = [m, k_1, k_2, r, F_1, t_1]

    # Define correlation matrix:
    ρ_X = Matrix(1.0 * I, 6, 6)

    # Define limit state function:
    g(x::Vector) =
        3 * x[4] -
        abs(((2 * x[5]) / (x[2] + x[3])) * sin((x[6] / 2) * sqrt((x[2] + x[3]) / x[1])))

    # Define reliability problem:
    problem = ReliabilityProblem(X, ρ_X, g)

    # Perform the reliability analysis using iHLRF:
    solution = solve(problem, FORM(iHLRF()))

    # Test the results:
    @test isapprox(solution.PoF, 4.88 * 10 ^ (-8), rtol=1E-2)
end
