"""
    InverseReliabilityProblem <: AbstractReliabilityProblem

Type used to define inverse reliability problems.

$(TYPEDFIELDS)
"""
mutable struct InverseReliabilityProblem <: AbstractReliabilityProblem
    "Random vector ``\\vec{X}``"
    X::AbstractVector{<:Distributions.UnivariateDistribution}
    "Correlation matrix ``\\rho^{X}``"
    ρ_X ::AbstractMatrix{<:Real}
    "Limit state function ``g(\\vec{X}, \\theta)``"
    g  ::Function
    "Target reliability index ``\\beta_t``"
    β  ::Real
end

"""
    InverseReliabilityProblemCache

Type used to store results of inverse reliability analysis.

$(TYPEDFIELDS)
"""
struct InverseReliabilityProblemCache
    "Design points in X-space at each iteration ``\\vec{x}_{i}^{*}``"
    x::Matrix{Float64}
    "Design points in U-space at each iteration ``\\vec{u}_{i}^{*}``"
    u::Matrix{Float64}
    "Parameter of interest at each iteration ``\\theta_{i}``"
    θ::Vector{Float64}
    "Limit state function at each iteration ``G(\\vec{u}_{i}^{*}, \\theta_{i})``"
    G::Vector{Float64}
    "Gradient of the limit state function at each iteration ``\\nabla_{\\vec{u}} G(\\vec{u}_{i}^{*}, \\theta_{i})``"
    ∇G_u::Matrix{Float64}
    "Gradient of the limit state function at each iteration ``\\nabla_{\\theta} G(\\vec{u}_{i}^{*}, \\theta_{i})``"
    ∇G_θ::Vector{Float64}
    "Normalized negative gradient of the limit state function at each iteration ``\\vec{\\alpha}_{i}``"
    α::Matrix{Float64}
    "Search direction for the design point in U-space at each iteration ``\\vec{d}_{u_{i}}``"
    du::Matrix{Float64}
    "Search direction for the parameter of interest at each iteration ``\\vec{d}_{u_{i}}``"
    dθ::Vector{Float64}
    "``c_{1}``-coefficients at each iteration ``c_{1_{i}}``"
    c_1::Vector{Float64}
    "``c_{2}``-coefficients at each iteration ``c_{2_{i}}``"
    c_2::Vector{Float64}
    "First merit function at each iteration ``m_{1_{i}}``"
    m_1::Vector{Float64}
    "Second merit function at each iteration ``m_{2_{i}}``"
    m_2::Vector{Float64}
    "Merit function at each iteration ``m_{i}``"
    m::Vector{Float64}
    "Step size at each iteration ``\\lambda_{i}``"
    λ::Vector{Float64}
end

"""
    solve(problem::InverseReliabilityProblem, θ_0::Real; 
        x_0::Union{Nothing, Vector{<:Real}} = nothing, 
        max_num_iters = 250, ϵ₁ = 10E-6, ϵ₂ = 10E-6, ϵ₃ = 10E-3,
        backend = AutoForwardDiff())

Function used to solve inverse reliability problems.
"""
function solve(problem::InverseReliabilityProblem, θ_0::Real; 
    max_num_iters = 250, ϵ₁ = 1E-6, ϵ₂ = 1E-6, ϵ₃ = 1E-6,
    x_0::Union{Nothing, Vector{<:Real}} = nothing, 
    c_0::Union{Nothing, Real} = nothing,
    backend = AutoForwardDiff())
    # Extract the problem data:
    X  = problem.X
    ρ_X = problem.ρ_X
    g  = problem.g
    β  = problem.β

    # Compute number of dimensions: 
    num_dims = length(X)

    # Preallocate:
    x   = Matrix{Float64}(undef, num_dims, max_num_iters)
    u   = Matrix{Float64}(undef, num_dims, max_num_iters)
    θ   = Vector{Float64}(undef, max_num_iters)
    G   = Vector{Float64}(undef, max_num_iters)
    ∇G_u = Matrix{Float64}(undef, num_dims, max_num_iters)
    ∇G_θ = Vector{Float64}(undef, max_num_iters)
    α   = Matrix{Float64}(undef, num_dims, max_num_iters)
    du  = Matrix{Float64}(undef, num_dims, max_num_iters)
    dθ  = Vector{Float64}(undef, max_num_iters)
    c_1  = Vector{Float64}(undef, max_num_iters)
    c_2  = Vector{Float64}(undef, max_num_iters)
    m_1  = Vector{Float64}(undef, max_num_iters)
    m_2  = Vector{Float64}(undef, max_num_iters)
    m   = Vector{Float64}(undef, max_num_iters)
    λ   = Vector{Float64}(undef, max_num_iters)

    # Perform the Nataf Transformation:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Initialize the design point in X-space:
    x[:, 1] = isnothing(x_0) ? mean.(X) : x_0

    # Initialize the unknown parameter:
    θ[1] = θ_0

    # Compute the initial design point in U-space:
    u[:, 1] = transformsamples(nataf_obj, x[:, 1], :X2U)

    # Evaluate the limit state function at the initial design point:
    G₀ = g(x[:, 1], θ[1])

    # Start iterating:
    for i in 1:(max_num_iters - 1)
        # Compute the design point in X-space:
        if i != 1
            x[:, i] = transformsamples(nataf_obj, u[:, i], :U2X)
        end

        # Compute the Jacobian of the transformation of the design point from X- to U-space:
        J_x_to_u = getjacobian(nataf_obj, x[:, i], :X2U)

        # Evaluate the limit state function at the design point in X-space:
        G[i] = g(x[:, i], θ[i])

        # Evaluate gradients of the limit state function at the design point in X-space:
        ∇g_x = try
            local ∇g_x(x, θ) = LinearAlgebra.transpose(gradient(Unknown -> g(Unknown, θ), backend, x))
            ∇g_x(x[:, i], θ[i])
        catch
            local ∇g_x(x, θ) = LinearAlgebra.transpose(gradient(Unknown -> g(Unknown, θ), AutoFiniteDiff(), x))
            ∇g_x(x[:, i], θ[i])
        end

        ∇g_θ = try
            local ∇g_θ(x, θ) = derivative(Unknown -> g(x, Unknown), backend, θ)
            ∇g_θ(x[:, i], θ[i])
        catch
            local ∇g_θ(x, θ) = derivative(Unknown -> g(x, Unknown), AutoFiniteDiff(), θ)
            ∇g_θ(x[:, i], θ[i])
        end

        # Convert the evaluated gradients of the limit state function from X- to U-space:
        ∇G_u[:, i] = vec(∇g_x * J_x_to_u)
        ∇G_θ[i]    = ∇g_θ

        # Compute the normalized negative gradient vector at the design point in U-space:
        α[:, i] = -∇G_u[:, i] / LinearAlgebra.norm(∇G_u[:, i])

        # Compute the c-coefficients:
        c_1[i] = isnothing(c_0) ? 2 * LinearAlgebra.norm(u[:, i]) / LinearAlgebra.norm(∇G_u[:, i]) + 10 : c_0
        c_2[i] = 1

        # Compute the merit functions at the current design point:
        m_1[i] = 0.5 * LinearAlgebra.norm(u[:, i]) ^ 2 + c_1[i] * abs(G[i])
        m_2[i] = 0.5 * c_2[i] * (LinearAlgebra.norm(u[:, i]) - β) ^ 2
        m[i]  = m_1[i] + m_2[i]

        # Compute the search directions:
        du[:, i] = β * α[:, i] - u[:, i]
        dθ[i]    = (LinearAlgebra.norm(∇G_u[:, i]) / ∇G_θ[i]) * (β - LinearAlgebra.dot(α[:, i], u[:, i]) - G[i] / LinearAlgebra.norm(∇G_u[:, i]))

        # Find a step size that satisfies m(uᵢ + λᵢdᵢ) < m(uᵢ):
        λ_t = 1
        u_t = u[:, i] + λ_t * du[:, i]
        θ_t = θ[i] + λ_t * dθ[i]
        x_t = transformsamples(nataf_obj, u_t, :U2X)
        G_t = g(x_t, θ_t)
        m_1_t = 0.5 * LinearAlgebra.norm(u_t) ^ 2 + c_1[i] * abs(G_t)
        m_2_t = 0.5 * c_2[i] * (LinearAlgebra.norm(u[:, i]) - β) ^ 2
        m_t  = m_1_t + m_2_t
        while m_t > m[i]
            # Update the step size:
            λ_t = λ_t / 2

            # Recalculate the merit function:
            u_t = u[:, i] + λ_t * du[:, i]
            θ_t = θ[i] + λ_t * dθ[i]
            x_t = transformsamples(nataf_obj, u_t, :U2X)
            G_t = g(x_t, θ_t)
            m_1_t = 0.5 * LinearAlgebra.norm(u_t) ^ 2 + c_1[i] * abs(G_t)
            m_2_t = 0.5 * c_2[i] * (LinearAlgebra.norm(u_t) - β) ^ 2
            m_t = m_1_t + m_2_t
        end

        # Update the step size:
        λ[i] = λ_t

        # Compute the new design point in U-space:
        u[:, i + 1] = u[:, i] + λ[i] * du[:, i]
        θ[i + 1] = θ[i] + λ[i] * dθ[i]

        # Compute the new design point in X-space:
        x[:, i + 1] = transformsamples(nataf_obj, u[:, i + 1], :U2X)

        # Check for convergance:
        criterion_1 = abs(g(x[:, i], θ[i]) / G₀) # Check if the limit state function is close to zero.
        criterion_2 = LinearAlgebra.norm(u[:, i] - LinearAlgebra.dot(α[:, i], u[:, i]) * α[:, i]) # Check if the design point is on the failure boundary.
        criterion_3 = LinearAlgebra.norm(u[:, i + 1] - u[:, i]) / LinearAlgebra.norm(u[:, i]) 
                   + abs(θ[i + 1] - θ[i]) / abs(θ[i]) 
                   + abs(LinearAlgebra.dot(α[:, i], u[:, i]) - β) / β # Check if the solution has converged.
        if criterion_1 < ϵ₁ && criterion_2 < ϵ₂ && criterion_3 < ϵ₃ && i != max_num_iters
            # Clean up the results:
            x   = x[:, 1:i]
            u   = u[:, 1:i]
            θ   = θ[1:i]
            G   = G[1:i]
            ∇G_u = ∇G_u[:, 1:i]
            ∇G_θ = ∇G_θ[1:i]
            α   = α[:, 1:i]
            du  = du[:, 1:i]
            dθ  = dθ[1:i]
            c_1  = c_1[1:i]
            c_2  = c_2[1:i]
            m_1  = m_1[1:i]
            m_2  = m_2[1:i]
            m   = m[1:i]
            λ   = λ[1:i]
            
            # Return results:
            return InverseReliabilityProblemCache(x, u, θ, G, ∇G_u, ∇G_θ, α, du, dθ, c_1, c_2, m_1, m_2, m, λ)

            # Break out:
            continue
        else
            # Check for convergance:
            i == max_num_iters && error("The solution did not converge. Try increasing the maximum number of iterations (max_num_iters) or relaxing the convergance criterions (ϵ₁, ϵ₂, and ϵ₃).")
        end
    end
end