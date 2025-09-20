@testset "Sensitivity Problems (Type I)" begin
    # Example 6.7 (p. 161) from "Structural and System Reliability" book by Armen Der Kiureghian

    # Define a random vector of correlated marginal distributions:
    M_1 = randomvariable("Normal",  "M", [250,   250   * 0.3])
    M_2 = randomvariable("Normal",  "M", [125,   125   * 0.3])
    P  = randomvariable("Gumbel",  "M", [2500,  2500  * 0.2])
    Y  = randomvariable("Weibull", "M", [40000, 40000 * 0.1])
    X  = [M_1, M_2, P, Y]
    ρ_X = [1 0.5 0.3 0; 0.5 1 0.3 0; 0.3 0.3 1 0; 0 0 0 1]

    # Define a limit state function:
    g(x::Vector, θ::Vector) = 1 - x[1] / (θ[1] * x[4]) - x[2] / (θ[2] * x[4]) - (x[3] / (θ[3] * x[4])) ^ 2

    # Define parameters of the limit state function:
    s_1 = 0.030
    s_2 = 0.015
    a  = 0.190
    Θ  = [s_1, s_2, a]

    # Define a sensitivity problem:
    problem = SensitivityProblemTypeI(X, ρ_X, g, Θ)

    # Perform the sensitivity analysis:
    solution = solve(problem)

    # Test the results:
    @test isapprox(solution.∇β,   [+36.80, +73.60, +9.260], rtol = 1E-2)
    @test isapprox(solution.∇PoF, [-0.700, -1.400, -0.176], rtol = 1E-2)
end

@testset "Sensitivity Problems (Type II) #1" begin
    # Example 6.7 (p. 161) from "Structural and System Reliability" book by Armen Der Kiureghian

    # Define the random vector as a function of its moments:
    function XFunction(Θ::Vector)
        M_1 = randomvariable("Normal",  "M", [Θ[1], Θ[2]])
        M_2 = randomvariable("Normal",  "M", [Θ[3], Θ[4]])
        P  = randomvariable("Gumbel",  "P", [Θ[5], Θ[6]])
        Y  = randomvariable("Weibull", "P", [Θ[7], Θ[8]])

        return [M_1, M_2, P, Y]
    end

    # Define the correlation matrix:
    ρ_X = [
        1.0 0.5 0.3 0.0
        0.5 1.0 0.3 0.0
        0.3 0.3 1.0 0.0
        0.0 0.0 0.0 1.0]

    # Define the parameters of the random vector:
    Θ = [
        250,  250   * 0.30,
        125,  125   * 0.30,
        2257, 1 / 0.00257,
        12.2, 41700]

    # Define the limit state function:
    a            = 0.190
    s_1           = 0.030
    s_2           = 0.015
    g(x::Vector) = 1 - x[1] / (s_1 * x[4]) - x[2] / (s_2 * x[4]) - (x[3] / (a * x[4])) ^ 2

    # Define a sensitivity problem:
    problem  = SensitivityProblemTypeII(XFunction, ρ_X, g, Θ)

    # Perform the sensitivity analysis:
    solution = solve(problem)

    # Test the results:
    @test isapprox(solution.∇β,   1E-3 * [-3.240, -3.920, -6.480, -7.840, -0.546, -1.334, +88.8, +0.0951], rtol = 5E-2)
    @test isapprox(solution.∇PoF, 1E-4 * [+0.617, +0.746, +1.230, +1.490, +0.104, -0.254, -16.9, -0.0181], rtol = 5E-2)
end

@testset "Sensitivity Problems (Type II) #2" begin
    # Example 6.7 (p. 161) from "Structural and System Reliability" book by Armen Der Kiureghian

    # Define the random vector as a function of its moments:
    function XFunction(Θ::Vector)
        M_1 = randomvariable("Normal",  "M", [Θ[1], Θ[2]])
        M_2 = randomvariable("Normal",  "M", [Θ[3], Θ[4]])
        P  = randomvariable("Gumbel",  "M", [Θ[5], Θ[6]])
        Y  = randomvariable("Weibull", "M", [Θ[7], Θ[8]])

        return [M_1, M_2, P, Y]
    end

    # Define the correlation matrix:
    ρ_X = [
        1.0 0.5 0.3 0.0
        0.5 1.0 0.3 0.0
        0.3 0.3 1.0 0.0
        0.0 0.0 0.0 1.0]

    # Define the parameters of the random vector:
    Θ = [
        250,   250   * 0.30,
        125,   125   * 0.30,
        2500,  2500  * 0.20,
        40000, 40000 * 0.10]

    # Define the limit state function:
    a            = 0.190
    s_1           = 0.030
    s_2           = 0.015
    g(x::Vector) = 1 - x[1] / (s_1 * x[4]) - x[2] / (s_2 * x[4]) - (x[3] / (a * x[4])) ^ 2

    # Define a sensitivity problem:
    problem  = SensitivityProblemTypeII(XFunction, ρ_X, g, Θ)

    # Perform the sensitivity analysis:
    solution = solve(problem)

    # Test the results:
    @test isapprox(solution.∇β, 1E-3 * [-3.240, -3.920, -6.480, -7.840, -0.546, -0.790, 0.124, -0.245], rtol = 5E-2)
end

@testset "Sensitivity Problems (Type II) #3" begin
    # "FORM Sensitivities to Distribution Parameters with the Nataf Transformation"
    # by J.-M. Bourinet
    # 10.1007/978-3-319-52425-2_12

    # Define the random vector as a function of its moments:
    function XFunction(Θ::Vector)
        X_1 = randomvariable("LogNormal", "M", [Θ[1], Θ[2]])
        X_2 = randomvariable("LogNormal", "M", [Θ[3], Θ[4]])
        X_3 = randomvariable("Uniform",   "M", [Θ[5], Θ[6]])

        return [X_1, X_2, X_3]
    end

    # Define the correlation matrix:
    ρ_X = [
        1.0 0.3 0.2
        0.3 1.0 0.2
        0.2 0.2 1.0]

    # Define the moments of the random vector:
    Θ = [
        500,  100,
        2000, 400,
        5,    0.5]

    # Define the limit state function:
    g(x::Vector) = 1 - x[2] / (1000 * x[3]) - (x[1] / (200 * x[3])) ^ 2

    # Define a sensitivity problem:
    problem  = SensitivityProblemTypeII(XFunction, ρ_X, g, Θ)
    
    # Perform the sensitivity analysis:
    solution = solve(problem)

    # Test the results:
    @test isapprox(solution.∇β, [-0.0059, -0.0079, -0.0009, -0.0006, 1.2602, -1.1942], rtol = 1E-2)
end

@testset "Sensitivity Problems (Type II) #4" begin
    # "FORM Sensitivities to Distribution Parameters with the Nataf Transformation"
    # by J.-M. Bourinet
    # 10.1007/978-3-319-52425-2_12

    # Define the random vector as a function of its moments:
    function XFunction(Θ::Vector)
        X_1 = randomvariable("LogNormal", "M", [Θ[1], Θ[2]])
        X_2 = randomvariable("LogNormal", "M", [Θ[3], Θ[4]])
    
        return [X_1, X_2]
    end

    # Define the correlation matrix:
    ρ_X = [
        1.0 0.5
        0.5 1.0]

    # Define the moments of the random vector:
    Θ = [
        5.0, 5.0,
        1.0, 1.0]

    # Define the limit state function:
    g(x::Vector) = x[1] - x[2]

    # Define a sensitivity problem:
    problem  = SensitivityProblemTypeII(XFunction, ρ_X, g, Θ)
    
    # Perform the sensitivity analysis:
    solution = solve(problem)

    # Test the results:
    @test isapprox(solution.∇β, [0.5184, -0.2548, -1.3629, 0.0446], rtol = 1E-2)
end