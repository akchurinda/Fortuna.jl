"""
    FORM <: AbstractReliabililyAnalysisMethod

Type used to perform reliability analysis using First-Order Reliability Method (FORM).

$(TYPEDFIELDS)
"""
Base.@kwdef struct FORM <: AbstractReliabililyAnalysisMethod
    submethod::FORMSubmethod = iHLRF()
end

"""
    MCFOSM <: FORMSubmethod

Type used to perform reliability analysis using Mean-Centered First-Order Second-Moment (MCFOSM) method.

$(TYPEDFIELDS)
"""
struct MCFOSM <: FORMSubmethod # Mean-Centered First-Order Second-Moment method
end

"""
    RF <: FORMSubmethod

Type used to perform reliability analysis using Rackwitz-Fiessler (RF) method.

$(TYPEDFIELDS)
"""
Base.@kwdef struct RF <: FORMSubmethod # Rackwitz-Fiessler method
    "Maximum number of iterations"
    max_num_iter::Integer = 250
    "convergence criterion ``\\epsilon``"
    tol::Real = 1E-6
end

"""
    HLRF <: FORMSubmethod

Type used to perform reliability analysis using Hasofer-Lind Rackwitz-Fiessler (HLRF) method.

$(TYPEDFIELDS)
"""
Base.@kwdef struct HLRF <: FORMSubmethod # Hasofer-Lind Rackwitz-Fiessler method
    "Maximum number of iterations"
    max_num_iter::Integer = 250
    "convergence criterion #1 ``\\epsilon_{1}``"
    tol_1::Real = 1E-6
    "convergence criterion #1 ``\\epsilon_{2}``"
    tol_2::Real = 1E-6
    "Starting point ``x_{0}``"
    x_0::Union{Nothing,Vector{<:Real}} = nothing
end

"""
    iHLRF <: FORMSubmethod

Type used to perform reliability analysis using improved Hasofer-Lind Rackwitz-Fiessler (iHLRF) method.

$(TYPEDFIELDS)
"""
Base.@kwdef struct iHLRF <: FORMSubmethod # Improved Hasofer-Lind Rackwitz-Fiessler method
    "Maximum number of iterations"
    max_num_iter::Integer = 250
    "convergence criterion #1 ``\\epsilon_{1}``"
    tol_1::Real = 1E-6
    "convergence criterion #1 ``\\epsilon_{2}``"
    tol_2::Real = 1E-6
    "Starting point ``x_{0}``"
    x_0::Union{Nothing,Vector{<:Real}} = nothing
    "c-coefficient applied at each iteration ``c_{0}``"
    c_0::Union{Nothing,Real} = nothing
end

"""
    MCFOSMCache

Type used to store results of reliability analysis performed using Mean-Centered First-Order Second-Moment (MCFOSM) method.

$(TYPEDFIELDS)
"""
struct MCFOSMCache
    "Reliability index ``\\beta``"
    β::Float64
end

"""
    RFCache

Type used to store results of reliability analysis performed using Rackwitz-Fiessler (RF) method.

$(TYPEDFIELDS)
"""
struct RFCache
    "Reliability index ``\\beta``"
    β::Float64
    "Design points in X-space at each iteration ``\\vec{x}_{i}^{*}``"
    x::Matrix{Float64}
    "Design points in U-space at each iteration ``\\vec{u}_{i}^{*}``"
    u::Matrix{Float64}
    "Means of equivalent normal marginals at each iteration ``\\vec{\\mu}_{i}``"
    μ::Matrix{Float64}
    "Standard deviations of equivalent normal marginals at each iteration ``\\vec{\\sigma}_{i}``"
    σ::Matrix{Float64}
    "Gradient of the limit state function at each iteration ``\\nabla G(\\vec{u}_{i}^{*})``"
    ∇G::Matrix{Float64}
    "Normalized negative gradient of the limit state function at each iteration ``\\vec{\\alpha}_{i}``"
    α::Matrix{Float64}
    "convergence status"
    convergence::Bool
end

"""
    HLRFCache

Type used to store results of reliability analysis performed using Hasofer-Lind Rackwitz-Fiessler (HLRF) method.

$(TYPEDFIELDS)
"""
struct HLRFCache
    "Reliability index ``\\beta``"
    β::Float64
    "Probability of failure ``P_{f}``"
    PoF::Float64
    "Design points in X-space at each iteration ``\\vec{x}_{i}^{*}``"
    x::Matrix{Float64}
    "Design points in U-space at each iteration ``\\vec{u}_{i}^{*}``"
    u::Matrix{Float64}
    "Limit state function at each iteration ``G(\\vec{u}_{i}^{*})``"
    G::Vector{Float64}
    "Gradient of the limit state function at each iteration ``\\nabla G(\\vec{u}_{i}^{*})``"
    ∇G::Matrix{Float64}
    "Normalized negative gradient of the limit state function at each iteration ``\\vec{\\alpha}_{i}``"
    α::Matrix{Float64}
    "Search direction at each iteration ``\\vec{d}_{i}``"
    d::Matrix{Float64}
    "Importance vector ``\\vec{\\gamma}``"
    γ::Vector{Float64}
    "convergence status"
    convergence::Bool
end

"""
    iHLRFCache

Type used to store results of reliability analysis performed using improved Hasofer-Lind Rackwitz-Fiessler (iHLRF) method.

$(TYPEDFIELDS)
"""
struct iHLRFCache
    "Reliability index ``\\beta``"
    β::Float64
    "Probability of failure ``P_{f}``"
    PoF::Float64
    "Design points in X-space at each iteration ``\\vec{x}_{i}^{*}``"
    x::Matrix{Float64}
    "Design points in U-space at each iteration ``\\vec{u}_{i}^{*}``"
    u::Matrix{Float64}
    "Limit state function at each iteration ``G(\\vec{u}_{i}^{*})``"
    G::Vector{Float64}
    "Gradient of the limit state function at each iteration ``\\nabla G(\\vec{u}_{i}^{*})``"
    ∇G::Matrix{Float64}
    "Normalized negative gradient of the limit state function at each iteration ``\\vec{\\alpha}_{i}``"
    α::Matrix{Float64}
    "Search direction at each iteration ``\\vec{d}_{i}``"
    d::Matrix{Float64}
    "c-coefficient at each iteration ``c_{i}``"
    c::Vector{Float64}
    "Merit function at each iteration ``m_{i}``"
    m::Vector{Float64}
    "Step size at each iteration ``\\lambda_{i}``"
    λ::Vector{Float64}
    "Importance vector ``\\vec{\\gamma}``"
    γ::Vector{Float64}
    "convergence status"
    convergence::Bool
end

"""
    solve(problem::ReliabilityProblem, AnalysisMethod::FORM; backend = AutoForwardDiff())

Function used to solve reliability problems using First-Order Reliability Method (FORM).
"""
function solve(problem::ReliabilityProblem, AnalysisMethod::FORM; backend=AutoForwardDiff())
    # Extract the analysis method:
    submethod = AnalysisMethod.submethod

    # Extract the problem data:
    X = problem.X
    ρ_X = problem.ρ_X
    g = problem.g

    if !isa(submethod, MCFOSM) &&
        !isa(submethod, RF) &&
        !isa(submethod, HLRF) &&
        !isa(submethod, iHLRF)
        throw(ArgumentError("Invalid FORM submethod!"))
    elseif isa(submethod, MCFOSM)
        # Compute the means of marginal distrbutions:
        M_X = Distributions.mean.(X)

        # Convert the correlation matrix into covariance matrix:
        σ_X = Distributions.std.(X)
        D_X = LinearAlgebra.diagm(σ_X)
        Σ_X = D_X * ρ_X * D_X

        # Compute gradient of the limit state function and evaluate it at the means of the marginal distributions:
        ∇g = try
            DifferentiationInterface.gradient(g, backend, M_X)
        catch
            DifferentiationInterface.gradient(g, AutoFiniteDiff(), M_X)
        end

        # Compute the reliability index:
        β = g(M_X) / sqrt(LinearAlgebra.transpose(∇g) * Σ_X * ∇g)

        # Return results:
        return MCFOSMCache(β)
    elseif isa(submethod, RF)
        # Extract the analysis details:
        max_num_iter = submethod.max_num_iter
        tol = submethod.tol

        # Error-catching:
        ρ_X == LinearAlgebra.I || throw(
            ArgumentError(
                "RF method is only applicable to random vectors with uncorrelated marginals!",
            ),
        )

        # Compute number of dimensions: 
        num_dims = length(X)

        # Preallocate:
        β = Vector{Float64}(undef, max_num_iter)
        x = Matrix{Float64}(undef, num_dims, max_num_iter)
        u = Matrix{Float64}(undef, num_dims, max_num_iter)
        μ = Matrix{Float64}(undef, num_dims, max_num_iter)
        σ = Matrix{Float64}(undef, num_dims, max_num_iter)
        ∇G = Matrix{Float64}(undef, num_dims, max_num_iter)
        α = Matrix{Float64}(undef, num_dims, max_num_iter)
        convergence = true

        # Initialize the design point in X-space:
        x[:, 1] = mean.(X)

        # Force the design point to lay on the failure boundary:
        function F(u, p)
            x′ = zeros(eltype(u), num_dims)
            x′[1:(end - 1)] = p[1:(end - 1)]
            x′[end] = u

            return g(x′)
        end

        problem = NonlinearSolve.NonlinearProblem(F, mean(X[end]), x[:, 1])
        x[end, 1] = try
            solution = NonlinearSolve.solve(problem, nothing; abstol=1E-9, reltol=1E-9)
            solution.u
        catch
            solution = NonlinearSolve.solve(
                problem,
                NonlinearSolve.FastShortcutNonlinearPolyalg(; autodiff=AutoFiniteDiff());
                abstol=1E-9,
                reltol=1E-9,
            )
            solution.u
        end

        # Prepare gradient of the limit state function at the initial design point:
        ∇g = similar(x[:, 1])

        # Start iterating:
        for i in 1:max_num_iter
            # Compute the mean and standard deviation values of the equivalient normal marginals:
            for j in 1:num_dims
                σ[j, i] =
                    Distributions.pdf(
                        Distributions.Normal(),
                        Distributions.quantile(
                            Distributions.Normal(), Distributions.cdf(X[j], x[j, i])
                        ),
                    ) / Distributions.pdf(X[j], x[j, i])
                μ[j, i] =
                    x[j, i] -
                    σ[j, i] * Distributions.quantile(
                        Distributions.Normal(), Distributions.cdf(X[j], x[j, i])
                    )
            end

            # Compute the design point in U-space:
            u[:, i] = (x[:, i] - μ[:, i]) ./ σ[:, i]

            # Evaluate gradient of the limit state function at the design point in U-space:
            try
                gradient!(g, ∇g, backend, x[:, i])
            catch
                gradient!(g, ∇g, AutoFiniteDiff(), x[:, i])
            end
            ∇G[:, i] = -σ[:, i] .* ∇g

            # Compute the reliability index:
            β[i] = LinearAlgebra.dot(∇G[:, i], u[:, i]) / LinearAlgebra.norm(∇G[:, i])

            # Compute the normalized negative gradient vector at the design point in U-space:
            α[:, i] = ∇G[:, i] / LinearAlgebra.norm(∇G[:, i])

            # Check for convergance:
            if i != 1
                criterion = abs(β[i] - β[i - 1])
                if criterion < tol || i == max_num_iter
                    if i == max_num_iter
                        @warn """
                        RF method did not converge in the given maximum number of iterations (max_num_iter = $max_num_iter)!
                        Try increasing the maximum number of iterations (max_num_iter) or relaxing the convergance criterion (tol)!
                        """

                        convergence = false
                    end

                    # Clean up the results:
                    β = β[i]
                    x = x[:, 1:i]
                    u = u[:, 1:i]
                    μ = μ[:, 1:i]
                    σ = σ[:, 1:i]
                    ∇G = ∇G[:, 1:i]
                    α = α[:, 1:i]

                    # Return results:
                    return RFCache(β, x, u, μ, σ, ∇G, α, convergence)

                    # Break out:
                    continue
                end
            end

            # Compute the new design point in U-space:
            u[:, i + 1] = β[i] * α[:, i]

            # Compute the new design point in X-space:
            x[:, i + 1] = μ[:, i] + σ[:, i] .* u[:, i + 1]

            # Force the design point to lay on the failure boundary:
            problem = NonlinearSolve.NonlinearProblem(F, x[end, i + 1], x[:, i + 1])
            x[end, i + 1] = try
                solution = NonlinearSolve.solve(problem, nothing; abstol=1E-9, reltol=1E-9)
                solution.u
            catch
                solution = NonlinearSolve.solve(
                    problem,
                    NonlinearSolve.FastShortcutNonlinearPolyalg(;
                        autodiff=AutoFiniteDiff()
                    );
                    abstol=1E-9,
                    reltol=1E-9,
                )
                solution.u
            end
        end
    elseif isa(submethod, HLRF)
        # Extract the analysis details:
        max_num_iter = submethod.max_num_iter
        tol_1 = submethod.tol_1
        tol_2 = submethod.tol_2
        x_0 = submethod.x_0

        # Compute number of dimensions: 
        num_dims = length(X)

        # Preallocate:
        x = Matrix{Float64}(undef, num_dims, max_num_iter)
        u = Matrix{Float64}(undef, num_dims, max_num_iter)
        G = Vector{Float64}(undef, max_num_iter)
        ∇G = Matrix{Float64}(undef, num_dims, max_num_iter)
        α = Matrix{Float64}(undef, num_dims, max_num_iter)
        d = Matrix{Float64}(undef, num_dims, max_num_iter)
        convergence = true

        # Perform the Nataf Transformation:
        nataf_obj = NatafTransformation(X, ρ_X)

        # Initialize the design point in X-space:
        x[:, 1] = isnothing(x_0) ? mean.(X) : x_0

        # Compute the initial design point in U-space:
        u[:, 1] = transformsamples(nataf_obj, x[:, 1], :X2U)

        # Evaluate the limit state function at the initial design point:
        G₀ = g(x[:, 1])

        # Set the step size to unity:
        λ = 1

        # Prepare gradient of the limit state function at the initial design point:
        ∇g = similar(x[:, 1])

        # Start iterating:
        for i in 1:max_num_iter
            # Compute the Jacobian of the transformation of the design point from X- to U-space:
            J_x_to_u = getjacobian(nataf_obj, x[:, i], :X2U)

            # Evaluate the limit state function at the design point in X-space:
            G[i] = g(x[:, i])

            # Evaluate gradient of the limit state function at the design point in X-space:
            try
                gradient!(g, ∇g, backend, x[:, i])
            catch
                gradient!(g, ∇g, AutoFiniteDiff(), x[:, i])
            end

            # Convert the evaluated gradient of the limit state function from X- to U-space:
            ∇G[:, i] = transpose(J_x_to_u) * ∇g # vec(∇g * J_x_to_u)

            # Compute the normalized negative gradient vector at the design point in U-space:
            α[:, i] = -∇G[:, i] / LinearAlgebra.norm(∇G[:, i])

            # Compute the search direction:
            d[:, i] =
                (
                    G[i] / LinearAlgebra.norm(∇G[:, i]) +
                    LinearAlgebra.dot(α[:, i], u[:, i])
                ) * α[:, i] - u[:, i]

            # Check for convergance:
            criterion_1 = abs(g(x[:, i]) / G₀) # Check if the limit state function is close to zero.
            criterion_2 = LinearAlgebra.norm(
                u[:, i] - LinearAlgebra.dot(α[:, i], u[:, i]) * α[:, i]
            ) # Check if the design point is on the failure boundary.
            if (criterion_1 < tol_1 && criterion_2 < tol_2) || i == max_num_iter
                # Check for convergance:
                if i == max_num_iter
                    @warn """
                    HL-RF method did not converge in the given maximum number of iterations (max_num_iter = $max_num_iter)!
                    Try increasing the maximum number of iterations (max_num_iter) or relaxing the convergance criteria (tol_1, tol_2)!
                    """

                    convergence = false
                end

                # Compute the reliability index:
                β = LinearAlgebra.dot(α[:, i], u[:, i])

                # Compute the probability of failure:
                PoF = Distributions.cdf(Distributions.Normal(), -β)

                # Compute the importance vector:
                L_inv = nataf_obj.L_inv
                γ = vec(
                    (LinearAlgebra.transpose(α[:, i]) * L_inv) /
                    LinearAlgebra.norm(LinearAlgebra.transpose(α[:, i]) * L_inv),
                )

                # Clean up the results:
                x = x[:, 1:i]
                u = u[:, 1:i]
                G = G[1:i]
                ∇G = ∇G[:, 1:i]
                α = α[:, 1:i]
                d = d[:, 1:i]

                # Return results:
                return HLRFCache(β, PoF, x, u, G, ∇G, α, d, γ, convergence)

                # Break out:
                continue
            end

            # Compute the new design point in U-space:
            u[:, i + 1] = u[:, i] + λ * d[:, i]

            # Compute the new design point in X-space:
            x[:, i + 1] = transformsamples(nataf_obj, u[:, i + 1], :U2X)
        end
    elseif isa(submethod, iHLRF)
        # Extract the analysis details:
        max_num_iter = submethod.max_num_iter
        tol_1 = submethod.tol_1
        tol_2 = submethod.tol_2
        x_0 = submethod.x_0
        c_0 = submethod.c_0

        # Compute number of dimensions: 
        num_dims = length(X)

        # Preallocate:
        x = Matrix{Float64}(undef, num_dims, max_num_iter)
        u = Matrix{Float64}(undef, num_dims, max_num_iter)
        G = Vector{Float64}(undef, max_num_iter)
        ∇G = Matrix{Float64}(undef, num_dims, max_num_iter)
        α = Matrix{Float64}(undef, num_dims, max_num_iter)
        d = Matrix{Float64}(undef, num_dims, max_num_iter)
        c = Vector{Float64}(undef, max_num_iter - 1)
        m = Vector{Float64}(undef, max_num_iter - 1)
        λ = Vector{Float64}(undef, max_num_iter - 1)
        convergence = true

        # Perform the Nataf Transformation:
        nataf_obj = NatafTransformation(X, ρ_X)

        # Initialize the design point in X-space:
        x[:, 1] = isnothing(x_0) ? mean.(X) : x_0

        # Compute the initial design point in U-space:
        u[:, 1] = transformsamples(nataf_obj, x[:, 1], :X2U)

        # Evaluate the limit state function at the initial design point:
        G₀ = g(x[:, 1])

        # Prepare gradient of the limit state function at the initial design point:
        ∇g = similar(x[:, 1])

        # Start iterating:
        for i in 1:max_num_iter
            # Compute the Jacobian of the transformation of the design point from X- to U-space:
            J_x_to_u = getjacobian(nataf_obj, x[:, i], :X2U)

            # Evaluate the limit state function at the design point in X-space:
            G[i] = g(x[:, i])

            # Evaluate gradient of the limit state function at the design point in X-space:
            try
                gradient!(g, ∇g, backend, x[:, i])
            catch
                gradient!(g, ∇g, AutoFiniteDiff(), x[:, i])
            end

            # Convert the evaluated gradient of the limit state function from X- to U-space:
            ∇G[:, i] = transpose(J_x_to_u) * ∇g # vec(∇g * J_x_to_u)

            # Compute the normalized negative gradient vector at the design point in U-space:
            α[:, i] = -∇G[:, i] / LinearAlgebra.norm(∇G[:, i])

            # Compute the search direction:
            d[:, i] =
                (
                    G[i] / LinearAlgebra.norm(∇G[:, i]) +
                    LinearAlgebra.dot(α[:, i], u[:, i])
                ) * α[:, i] - u[:, i]

            # Check for convergance:
            criterion_1 = abs(g(x[:, i]) / G₀) # Check if the limit state function is close to zero.
            criterion_2 = LinearAlgebra.norm(
                u[:, i] - LinearAlgebra.dot(α[:, i], u[:, i]) * α[:, i]
            ) # Check if the design point is on the failure boundary.
            if (criterion_1 < tol_1 && criterion_2 < tol_2) || i == max_num_iter
                # Check for convergance:
                if i == max_num_iter
                    @warn """
                    iHL-RF method did not converge in the given maximum number of iterations (max_num_iter = $max_num_iter)!
                    Try increasing the maximum number of iterations (max_num_iter) or relaxing the convergance criteria (tol_1, tol_2)!
                    """

                    convergence = false
                end

                # Compute the reliability index:
                β = LinearAlgebra.dot(α[:, i], u[:, i])

                # Compute the probability of failure:
                PoF = Distributions.cdf(Distributions.Normal(), -β)

                # Compute the importance vector:
                L_inv = nataf_obj.L_inv
                γ = vec(
                    (LinearAlgebra.transpose(α[:, i]) * L_inv) /
                    LinearAlgebra.norm(LinearAlgebra.transpose(α[:, i]) * L_inv),
                )

                # Clean up the results:
                x = x[:, 1:i]
                u = u[:, 1:i]
                G = G[1:i]
                ∇G = ∇G[:, 1:i]
                α = α[:, 1:i]
                d = d[:, 1:i]
                c = c[1:(i - 1)]
                m = m[1:(i - 1)]
                λ = λ[1:(i - 1)]

                # Return results:
                return iHLRFCache(β, PoF, x, u, G, ∇G, α, d, c, m, λ, γ, convergence)

                # Break out:
                continue
            end

            # Compute the c-coefficient:
            c[i] = if isnothing(c_0)
                2 * LinearAlgebra.norm(u[:, i]) / LinearAlgebra.norm(∇G[:, i]) + 10
            else
                c_0
            end

            # Compute the merit function at the current design point:
            m[i] = 0.5 * LinearAlgebra.norm(u[:, i]) ^ 2 + c[i] * abs(G[i])

            # Find a step size that satisfies m(uᵢ + λᵢdᵢ) < m(uᵢ):
            λ_t = 1
            u_t = u[:, i] + λ_t * d[:, i]
            x_t = transformsamples(nataf_obj, u_t, :U2X)
            G_t = g(x_t)
            m_t = 0.5 * LinearAlgebra.norm(u_t) ^ 2 + c[i] * abs(G_t)
            counter = 1
            while !(m_t ≤ m[i])
                if counter == 30
                    break
                end

                # Update the step size:
                λ_t = λ_t / 2

                # Recalculate the merit function:
                u_t = u[:, i] + λ_t * d[:, i]
                x_t = transformsamples(nataf_obj, u_t, :U2X)
                G_t = g(x_t)
                m_t = 0.5 * LinearAlgebra.norm(u_t) ^ 2 + c[i] * abs(G_t)
                counter = counter + 1
            end

            # Update the step size:
            λ[i] = λ_t

            # Compute the new design point in U-space:
            u[:, i + 1] = u[:, i] + λ[i] * d[:, i]

            # Compute the new design point in X-space:
            x[:, i + 1] = transformsamples(nataf_obj, u[:, i + 1], :U2X)
        end
    end
end
