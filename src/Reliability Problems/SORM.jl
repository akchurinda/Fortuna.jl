"""
    SORM <: AbstractReliabililyAnalysisMethod

Type used to perform reliability analysis using Second-Order Reliability Method (SORM).

$(TYPEDFIELDS)
"""
Base.@kwdef struct SORM <: AbstractReliabililyAnalysisMethod
    submethod::SORMSubmethod = CF()
end

"""
    CF <: SORMSubmethod

Type used to perform reliability analysis using Curve-Fitting (CF) method.

$(TYPEDFIELDS)
"""
Base.@kwdef struct CF <: SORMSubmethod # Curve-Fitting method
    "Step size used to compute the Hessian at the design point in ``U``-space"
    Δ::Real = 1 / 1000
end

"""
    PF <: SORMSubmethod

Type used to perform reliability analysis using Point-Fitting (PF) method.

$(TYPEDFIELDS)
"""
Base.@kwdef struct PF <: SORMSubmethod # Point-Fitting method

end

"""
    CFCache

Type used to perform reliability analysis using Point-Fitting (PF) method.

$(TYPEDFIELDS)
"""
struct CFCache # Curve-Fitting method
    "Results of reliability analysis performed using First-Order Reliability Method (FORM)"
    form_solution::Union{RFCache, HLRFCache, iHLRFCache}
    "Generalized reliability indices ``\\beta``"
    β_2::Vector{Union{Missing, Float64}}
    "Probabilities of failure ``P_{f}``"
    PoF_2::Vector{Union{Missing, Float64}}
    "Principal curvatures ``\\kappa``"
    κ::Vector{Float64}
end

"""
    PFCache

Type used to perform reliability analysis using Point-Fitting (PF) method.

$(TYPEDFIELDS)
"""
struct PFCache # Point-Fitting method
    "Results of reliability analysis performed using First-Order Reliability Method (FORM)"
    form_solution::iHLRFCache
    "Generalized reliability index ``\\beta``"
    β_2::Vector{Union{Missing, Float64}}
    "Probabilities of failure ``P_{f}``"
    PoF_2::Vector{Union{Missing, Float64}}
    "Fitting points on the negative side of the hyper-cylinder"
    neg_fit_pts::Matrix{Float64}
    "Fitting points on the positive side of the hyper-cylinder"
    pos_fit_pts::Matrix{Float64}
    "Principal curvatures on the negative and positive sides"
    κ_1::Matrix{Float64}
    "Principal curvatures of each hyper-semiparabola"
    κ_2::Matrix{Float64}
end

"""
solve(Problem::ReliabilityProblem, AnalysisMethod::SORM; 
    form_solution::Union{Nothing, HLRFCache, iHLRFCache} = nothing,
    FORMConfig::FORM = FORM(), 
    backend = AutoForwardDiff())

Function used to solve reliability problems using Second-Order Reliability Method (SORM).
"""
function solve(Problem::ReliabilityProblem, AnalysisMethod::SORM; 
        form_solution::Union{Nothing, RFCache, HLRFCache, iHLRFCache} = nothing,
        form_config::FORM = FORM(), 
        backend = AutoForwardDiff())
    # Extract the analysis method:
    submethod  = AnalysisMethod.submethod

    # Error-catching:
    isa(form_config.submethod, MCFOSM) && throw(ArgumentError("MCFOSM cannot be used with SORM as it does not provide any information about the design point!"))

    # Determine the design point using FORM:
    form_solution = isnothing(form_solution) ? solve(Problem, form_config, backend = backend) : form_solution
    u            = form_solution.u[:, end]
    ∇G           = form_solution.∇G[:, end]
    α            = form_solution.α[:, end]
    β_1           = form_solution.β

    # Extract the problem data:
    X  = Problem.X
    ρ_X = Problem.ρ_X
    g  = Problem.g

    if !isa(submethod, CF) && !isa(submethod, PF)
        error("Invalid SORM submethod!")
    elseif isa(submethod, CF)
        # Extract the analysis details:
        Δ = submethod.Δ

        # Compute number of dimensions: 
        num_dims = length(X)

        # Perform Nataf transformation:
        nataf_obj = NatafTransformation(X, ρ_X)

        # Compute the Hessian at the design point in U-space:
        H = gethessian(g, nataf_obj, num_dims, u, Δ)

        # Compute the orthonomal matrix:
        R = getorthonormal(α, num_dims)

        # Evaluate the principal curvatures:
        A = R * H * LinearAlgebra.transpose(R) / LinearAlgebra.norm(∇G)
        κ = LinearAlgebra.eigen(A[1:(end - 1), 1:(end - 1)]).values

        # Compute the probabilities of failure:
        PoF_2 = Vector{Union{Missing, Float64}}(undef, 2)

        begin # Hohenbichler-Rackwitz (1988)
            ψ = Distributions.pdf(Distributions.Normal(), β_1) / Distributions.cdf(Distributions.Normal(), -β_1)

            if all(κᵢ -> ψ * κᵢ > -1, κ)
                PoF_2[1] = Distributions.cdf(Distributions.Normal(), -β_1) * prod(κᵢ -> 1 / sqrt(1 + ψ * κᵢ), κ)
            else
                PoF_2[1] = missing
                @warn "Condition of Hohenbichler-Rackwitz's approximation of the probability of failure was not satisfied!"
            end
        end

        begin # Breitung (1984)
            if all(κᵢ -> β_1 * κᵢ > -1, κ)
                PoF_2[2] = Distributions.cdf(Distributions.Normal(), -β_1) * prod(κᵢ -> 1 / sqrt(1 + β_1 * κᵢ), κ)
            else
                PoF_2[2] = missing
                @warn "Condition of Breitung's approximation of the probability of failure was not satisfied!"
            end
        end

        # Compute the generalized reliability index:
        β_2 = [ismissing(PoF_2[i]) ? missing : -Distributions.quantile(Distributions.Normal(), PoF_2[i]) for i in eachindex(PoF_2)]

        # Return results:
        return CFCache(form_solution, β_2, PoF_2, κ)
    elseif isa(submethod, PF)
        # Compute number of dimensions: 
        num_dims = length(X)

        # Perform Nataf transformation:
        nataf_obj = NatafTransformation(X, ρ_X)

        # Compute the orthonomal matrix:
        R = getorthonormal(α, num_dims)
        u_rot = R * u
        if u_rot[end] < 0
            R = -R
        end

        # Compute radius of a hypercylinder:
        if β_1 < 1
            H = 1
        elseif β_1 ≥ 1 && β_1 ≤ 3
            H = β_1
        elseif β_1 > 3
            H = 3
        end

        # Compute fitting points:
        pos_fit_pts = Matrix{Float64}(undef, num_dims - 1, 2)
        neg_fit_pts = Matrix{Float64}(undef, num_dims - 1, 2)
        κ_1             = Matrix{Float64}(undef, num_dims - 1, 2)
        for i in 1:(num_dims - 1)
            function F(u, p)
                u_prime      = zeros(eltype(u), num_dims)
                u_prime[i]   = p
                u_prime[end] = u
            
                return G_prime(g, nataf_obj, R, u_prime)
            end

            # Negative side:
            neg_problem  = NonlinearSolve.NonlinearProblem(F, β_1, -H)
            neg_solution = try
                NonlinearSolve.solve(neg_problem, nothing, abstol = 1E-9, reltol = 1E-9)
            catch
                NonlinearSolve.solve(neg_problem, NonlinearSolve.FastShortcutNonlinearPolyalg(autodiff = AutoFiniteDiff()), abstol = 1E-9, reltol = 1E-9)
            end
            neg_fit_pts[i, 1] = -H
            neg_fit_pts[i, 2] = neg_solution.u

            # Positive side:
            pos_problem  = NonlinearSolve.NonlinearProblem(F, β_1, +H)
            pos_solution = try
                NonlinearSolve.solve(pos_problem, nothing, abstol = 1E-9, reltol = 1E-9)
            catch
                NonlinearSolve.solve(neg_problem, NonlinearSolve.FastShortcutNonlinearPolyalg(autodiff = AutoFiniteDiff()), abstol = 1E-9, reltol = 1E-9)
            end
            pos_fit_pts[i, 1] = +H
            pos_fit_pts[i, 2] = pos_solution.u

            # Curvatures:
            κ_1[i, 1] = 2 * (neg_fit_pts[i, 2] - β_1) / (neg_fit_pts[i, 1] ^ 2) # Negative side
            κ_1[i, 2] = 2 * (pos_fit_pts[i, 2] - β_1) / (pos_fit_pts[i, 1] ^ 2) # Positive side
        end

        # Compute number of hyperquadrants used to fit semiparabolas:
        num_hyperquadrants = 2 ^ (num_dims - 1)

        # Get all possible permutations of curvatures:
        idx = Base.Iterators.repeated(1:2, num_dims - 1)
        idx = Base.Iterators.product(idx...)
        idx = collect(idx)
        idx = vec(idx)

        κ_2 = Matrix{Float64}(undef, num_hyperquadrants, num_dims - 1)
        for i in 1:num_hyperquadrants
            for j in 1:(num_dims - 1)
                κ_2[i, j] = κ_1[j, idx[i][j]]
            end
        end

        # Compute the probabilities of failure for each semiparabola:
        PoF_2 = Matrix{Union{Missing, Float64}}(undef, num_hyperquadrants, 2)
        for i in 1:num_hyperquadrants
            κ = κ_2[i, :]

            begin # Hohenbichler-Rackwitz (1988)
                ψ = Distributions.pdf(Distributions.Normal(), β_1) / Distributions.cdf(Distributions.Normal(), -β_1)

                if all(κᵢ -> ψ * κᵢ > -1, κ)
                    PoF_2[i, 1] = Distributions.cdf(Distributions.Normal(), -β_1) * prod(κᵢ -> 1 / sqrt(1 + ψ * κᵢ), κ)
                else
                    PoF_2[i, 1] = missing
                    @warn "Condition of Hohenbichler-Rackwitz's approximation of the probability of failure was not satisfied!"
                end
            end

            begin # Breitung (1984)
                if all(κᵢ -> β_1 * κᵢ > -1, κ)
                    PoF_2[i, 2] = Distributions.cdf(Distributions.Normal(), -β_1) * prod(κᵢ -> 1 / sqrt(1 + β_1 * κᵢ), κ)
                else
                    PoF_2[i, 2] = missing
                    @warn "Condition of Breitung's approximation of the probability of failure was not satisfied!"
                end
            end
        end

        PoF_2 = (1/ num_hyperquadrants) * PoF_2
        PoF_2 = sum(PoF_2, dims = 1)
        PoF_2 = vec(PoF_2)

        # Compute the generalized reliability index:
        β_2 = [ismissing(PoF_2[i]) ? missing : -Distributions.quantile(Distributions.Normal(), PoF_2[i]) for i in eachindex(PoF_2)]

        # Return results:
        return PFCache(form_solution, β_2, PoF_2, neg_fit_pts, pos_fit_pts, κ_1, κ_2)
    end
end

function gethessian(g::Function, nataf_obj::NatafTransformation, num_dims::Integer, u::Vector{Float64}, Δ::Real)
    # Preallocate:
    H = Matrix{Float64}(undef, num_dims, num_dims)

    for i in 1:num_dims
        for j in 1:num_dims
            # Define the pertubation directions:
            e_i = zeros(num_dims,)
            e_j = zeros(num_dims,)
            e_i[i] = 1
            e_j[j] = 1

            # Perturb the design point in U-space:
            u_1 = u + Δ * e_i + Δ * e_j
            u_2 = u + Δ * e_i - Δ * e_j
            u_3 = u - Δ * e_i + Δ * e_j
            u_4 = u - Δ * e_i - Δ * e_j

            # Transform the perturbed design points from X- to U-space:
            x_1 = transformsamples(nataf_obj, u_1, :U2X)
            x_2 = transformsamples(nataf_obj, u_2, :U2X)
            x_3 = transformsamples(nataf_obj, u_3, :U2X)
            x_4 = transformsamples(nataf_obj, u_4, :U2X)

            # Evaluate the limit state function at the perturbed points:
            G_1 = g(x_1)
            G_2 = g(x_2)
            G_3 = g(x_3)
            G_4 = g(x_4)

            # Evaluate the entries of the Hessian using finite difference method:
            H[i, j] = (G_1 - G_2 - G_3 + G_4) / (4 * Δ ^ 2)
        end
    end

    return H
end

function getorthonormal(α::Vector{Float64}, num_dims::Integer)
    # Initilize the matrix:
    A       = Matrix(1.0 * I, num_dims, num_dims)
    A       = reverse(A, dims = 2)
    A[:, 1] = LinearAlgebra.transpose(α)

    # Perform QR factorization:
    Q, _ = LinearAlgebra.qr(A)
    Q = Matrix(Q)

    # Clean up the result:
    R = LinearAlgebra.transpose(reverse(Q, dims = 2))
    R = Matrix(R)

    return R
end

function G_prime(g::Function, nataf_obj::NatafTransformation, R::Matrix{Float64}, U_prime_samples::AbstractVector)
    # Transform samples from U'- to X-space:
    U_samples = LinearAlgebra.transpose(R) * U_prime_samples
    X_samples = transformsamples(nataf_obj, U_samples, :U2X)

    # Evaluate the limit state function at the transform samples:
    G_prime_samples = g(X_samples)

    # Return the result:
    return G_prime_samples
end