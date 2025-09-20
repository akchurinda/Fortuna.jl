"""
    SensitivityProblemTypeI <: AbstractReliabilityProblem

Type used to define sensitivity problems of type I (sensitivities w.r.t. the parameters of the limit state function).

$(TYPEDFIELDS)
"""
mutable struct SensitivityProblemTypeI <: AbstractReliabilityProblem
    "Random vector ``\\vec{X}``"
    X::AbstractVector{<:Distributions.UnivariateDistribution}
    "Correlation matrix ``\\rho^{X}``"
    ρ_X::AbstractMatrix{<:Real}
    "Limit state function ``g(\\vec{X}, \\vec{\\Theta})``"
    g::Function
    "Parameters of limit state function ``\\vec{\\Theta}``"
    Θ::AbstractVector{<:Real}
end

"""
    SensitivityProblemTypeII <: AbstractReliabilityProblem

Type used to define sensitivity problems of type II (sensitivities w.r.t. the parameters of the random vector).

$(TYPEDFIELDS)
"""
mutable struct SensitivityProblemTypeII <: AbstractReliabilityProblem
    "Random vector ``\\vec{X}(\\vec{\\Theta})``"
    X::Function
    "Correlation matrix ``\\rho^{X}``"
    ρ_X::AbstractMatrix{<:Real}
    "Limit state function ``g(\\vec{X})``"
    g::Function
    "Parameters of limit state function ``\\vec{\\Theta}``"
    Θ::AbstractVector{<:Real}
end


"""
    SensitivityProblemCache

Type used to store results of sensitivity analysis for problems of type I (sensitivities w.r.t. the parameters of the limit state function).

$(TYPEDFIELDS)
"""
struct SensitivityProblemCache
    "Results of reliability analysis performed using First-Order Reliability Method (FORM)"
    form_solution::iHLRFCache
    "Sensivity vector of reliability index ``\\vec{\\nabla}_{\\vec{\\Theta}} \\beta``"
    ∇β::Vector{Float64}
    "Sensivity vector of probability of failure ``\\vec{\\nabla}_{\\vec{\\Theta}} P_{f}``"
    ∇PoF::Vector{Float64}
end

"""
    solve(problem::SensitivityProblemTypeI; backend = AutoForwardDiff())

Function used to solve sensitivity problems of type I (sensitivities w.r.t. the parameters of the limit state function).
"""
function solve(problem::SensitivityProblemTypeI; backend = AutoForwardDiff())
    # Extract the problem data:
    X  = problem.X
    ρ_X = problem.ρ_X
    g  = problem.g
    Θ  = problem.Θ

    # Define a reliability problem for the FORM analysis:
    g₁(x) = g(x, Θ)
    form_problem = ReliabilityProblem(X, ρ_X, g₁)

    # Solve the reliability problem using the FORM:
    form_solution = solve(form_problem, FORM(), backend = backend)
    x            = form_solution.x[:, end]
    u            = form_solution.u[:, end]
    β            = form_solution.β

    # Perform Nataf transformation:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Define gradient functions of the limit state function in X- and U-spaces, and compute the sensitivities vector for the reliability index:
    ∇β = try
        local ∇g(x, θ) = gradient(unknown -> g(x, unknown), backend, θ)
        local ∇G(u, θ) = gradient(unknown -> G(g, θ, nataf_obj, unknown), backend, u)
        ∇g(x, Θ) / LinearAlgebra.norm(∇G(u, Θ))
    catch
        local ∇g(x, θ) = gradient(unknown -> g(x, unknown), AutoFiniteDiff(), θ)
        local ∇G(u, θ) = gradient(unknown -> G(g, θ, nataf_obj, unknown), AutoFiniteDiff(), u)
        ∇g(x, Θ) / LinearAlgebra.norm(∇G(u, Θ))
    end

    # Compute the sensitivities vector for the probability of failure:
    ∇PoF = -Distributions.pdf(Distributions.Normal(), β) * ∇β

    return SensitivityProblemCache(form_solution, ∇β, ∇PoF)
end

"""
    solve(problem::SensitivityProblemTypeII; backend = AutoForwardDiff())

Function used to solve sensitivity problems of type II (sensitivities w.r.t. the parameters of the random vector).
"""
function solve(problem::SensitivityProblemTypeII; backend = AutoForwardDiff())
    # Extract the problem data:
    X  = problem.X
    ρ_X = problem.ρ_X
    g  = problem.g
    Θ  = problem.Θ

    # Define a reliability problem for the FORM analysis:
    form_problem = ReliabilityProblem(X(Θ), ρ_X, g)

    # Solve the reliability problem using the FORM:
    form_solution = solve(form_problem, FORM(), backend = backend)
    x            = form_solution.x[:, end]
    α            = form_solution.α[:, end]
    β            = form_solution.β

    # Define the Jacobian of the transformation function w.r.t. the parameters of the random vector and compute the sensitivity vector for the reliability index:
    ∇β = try
        local ∇T(θ) = jacobian(unknown -> transformsamples(NatafTransformation(X(unknown), ρ_X), x, :X2U), backend, θ)
        vec(LinearAlgebra.transpose(α) * ∇T(Θ))
    catch
        local ∇T(θ) = jacobian(unknown -> transformsamples(NatafTransformation(X(unknown), ρ_X), x, :X2U), AutoFiniteDiff(), θ)
        vec(LinearAlgebra.transpose(α) * ∇T(Θ))
    end

    # Compute the sensitivity vector for the probability of failure:
    ∇PoF = -Distributions.pdf(Distributions.Normal(), β) * ∇β

    return SensitivityProblemCache(form_solution, ∇β, ∇PoF)
end

function G(g::Function, Θ::AbstractVector{<:Real}, nataf_obj::NatafTransformation, U_sample::AbstractVector{<:Real})
    # Transform samples:
    X_sample = transformsamples(nataf_obj, U_sample, :U2X)

    # Evaluate the limit state function at the transform samples:
    G_sample = g(X_sample, Θ)

    # Return the result:
    return G_sample
end