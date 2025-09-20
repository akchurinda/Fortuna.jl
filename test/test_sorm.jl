@testset "SORM #1 - CF" begin
    # Examples 6.5 (p. 147) and 7.2 (p. 188) from "Structural and System Reliability" book by Armen Der Kiureghian

    # Define a random vector of correlated marginal distributions:
    M_1 = randomvariable("Normal",  "M", [250,   250   * 0.3])
    M_2 = randomvariable("Normal",  "M", [125,   125   * 0.3])
    P  = randomvariable("Gumbel",  "M", [2500,  2500  * 0.2])
    Y  = randomvariable("Weibull", "M", [40000, 40000 * 0.1])
    X  = [M_1, M_2, P, Y]
    ρ_X = [1 0.5 0.3 0; 0.5 1 0.3 0; 0.3 0.3 1 0; 0 0 0 1]

    # Define a limit state function:
    a            = 0.190
    s_1           = 0.030
    s_2           = 0.015
    g(x::Vector) = 1 - x[1] / (s_1 * x[4]) - x[2] / (s_2 * x[4]) - (x[3] / (a * x[4]))^2

    # Define a reliability problem:
    problem = ReliabilityProblem(X, ρ_X, g)

    # Perform the reliability analysis using curve-fitting SORM:
    solution = solve(problem, SORM(CF()))

    # Test the results:
    @test isapprox(solution.β_2,   [2.35, 2.35],         rtol = 1E-2)
    @test isapprox(solution.PoF_2, [0.00960, 0.00914],   rtol = 1E-2)
    @test isapprox(solution.κ,    [-0.155, -0.0399, 0], rtol = 1E-2)
end

@testset "SORM #2 - CF" begin
    # https://www.researchgate.net/publication/370230768_Structural_reliability_analysis_by_line_sampling_A_Bayesian_active_learning_treatment

    # Define random vector:
    m   = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.05])
    k_1  = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.10])
    k_2  = randomvariable("LogNormal", "M", [0.2, 0.2 * 0.10])
    r   = randomvariable("LogNormal", "M", [0.5, 0.5 * 0.10])
    F_1  = randomvariable("LogNormal", "M", [0.4, 0.4 * 0.20])
    t_1  = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.20])
    X   = [m, k_1, k_2, r, F_1, t_1]

    # Define correlation matrix:
    ρ_X  = Matrix(1.0 * I, 6, 6)

    # Define limit state function:
    g(x::Vector) = 3 * x[4] - abs(((2 * x[5]) / (x[2] + x[3])) * sin((x[6] / 2) * sqrt((x[2] + x[3]) / x[1])))

    # Define reliability problem:
    problem = ReliabilityProblem(X, ρ_X, g)

    # Solve reliability problem using Curve-Fitting Method:
    solution = solve(problem, SORM(CF()))

    # Test the results:
    @test isapprox(solution.PoF_2[1], 4.08 * 10 ^ (-8), rtol = 5E-2)
    @test isapprox(solution.PoF_2[2], 4.08 * 10 ^ (-8), rtol = 5E-2)
end

@testset "SORM #3 - PF" begin
    # Examples 6.5 (p. 147) and 7.7 (p. 196) from "Structural and System Reliability" book by Armen Der Kiureghian

    # Define a random vector of correlated marginal distributions:
    M_1 = randomvariable("Normal", "M",  [250,   250   * 0.3])
    M_2 = randomvariable("Normal", "M",  [125,   125   * 0.3])
    P  = randomvariable("Gumbel", "M",  [2500,  2500  * 0.2])
    Y  = randomvariable("Weibull", "M", [40000, 40000 * 0.1])
    X  = [M_1, M_2, P, Y]
    ρ_X = [1 0.5 0.3 0; 0.5 1 0.3 0; 0.3 0.3 1 0; 0 0 0 1]

    # Define a limit state function:
    a            = 0.190
    s_1           = 0.030
    s_2           = 0.015
    g(x::Vector) = 1 - x[1] / (s_1 * x[4]) - x[2] / (s_2 * x[4]) - (x[3] / (a * x[4]))^2

    # Define a reliability problem:
    problem = ReliabilityProblem(X, ρ_X, g)

    # Perform the reliability analysis using point-fitting SORM:
    solution = solve(problem, SORM(PF()))

    # Test the results:
    @test isapprox(solution.β_2,                   [2.36, 2.36],       rtol = 5E-2)
    @test isapprox(solution.PoF_2,                 [0.00913, 0.00913], rtol = 5E-2)
    @test isapprox(solution.neg_fit_pts[1, :], [-2.47, +2.27],     rtol = 5E-2)
    @test isapprox(solution.neg_fit_pts[2, :], [-2.47, +2.43],     rtol = 5E-2)
    @test isapprox(solution.neg_fit_pts[3, :], [-2.47, +2.05],     rtol = 5E-2)
    @test isapprox(solution.pos_fit_pts[1, :], [+2.47, +2.34],     rtol = 5E-2)
    @test isapprox(solution.pos_fit_pts[2, :], [+2.47, +2.44],     rtol = 5E-2)
    @test isapprox(solution.pos_fit_pts[3, :], [+2.47, +2.13],     rtol = 5E-2)
    @test isapprox(solution.κ_1[1, :],             [-0.0630, -0.0405], rtol = 5E-2)
    @test isapprox(solution.κ_1[2, :],             [-0.0097, -0.0120], rtol = 5E-2)
    @test isapprox(solution.κ_1[3, :],             [-0.1380, -0.1110], rtol = 5E-2)
end

@testset "SORM #4 - PF" begin
    # https://www.researchgate.net/publication/370230768_Structural_reliability_analysis_by_line_sampling_A_Bayesian_active_learning_treatment

    # Define random vector:
    m  = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.05])
    k_1 = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.10])
    k_2 = randomvariable("LogNormal", "M", [0.2, 0.2 * 0.10])
    r  = randomvariable("LogNormal", "M", [0.5, 0.5 * 0.10])
    F_1 = randomvariable("LogNormal", "M", [0.4, 0.4 * 0.20])
    t_1 = randomvariable("LogNormal", "M", [1.0, 1.0 * 0.20])
    X  = [m, k_1, k_2, r, F_1, t_1]

    # Define correlation matrix:
    ρ_X  = Matrix(1.0 * I, 6, 6)

    # Define limit state function:
    g(x::Vector) = 3 * x[4] - abs(((2 * x[5]) / (x[2] + x[3])) * sin((x[6] / 2) * sqrt((x[2] + x[3]) / x[1])))

    # Define reliability problem:
    problem = ReliabilityProblem(X, ρ_X, g)

    # Solve reliability problem using Point-Fitting Method:
    solution = solve(problem, SORM(PF()))
    
    # Test the results:
    @test isapprox(solution.PoF_2[1], 4.08 * 10 ^ (-8), rtol = 5E-2)
    @test isapprox(solution.PoF_2[2], 4.08 * 10 ^ (-8), rtol = 5E-2)
end