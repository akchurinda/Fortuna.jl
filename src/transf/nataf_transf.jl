"""
    NatafTransformation <: AbstractTransformation

Type used to perform Nataf Transformation.

$(TYPEDFIELDS)
"""
mutable struct NatafTransformation <: AbstractIsoprobabilisticTransformation
    "Random vector ``\\vec{X}``"
    X::AbstractVector{<:Distributions.ContinuousUnivariateDistribution}
    "Correlation matrix ``\\rho^{X}``"
    ρ_X::AbstractMatrix{<:Real}
    "Distorted correlation matrix ``\\rho^{Z}``"
    ρ_Z::AbstractMatrix{<:Real}
    "Lower triangular matrix of the Cholesky decomposition of the distorted correlation matrix ``L``"
    L::AbstractMatrix{<:Real}
    "Inverse of the lower triangular matrix of the Cholesky decomposition of the distorted correlation matrix ``L^{-1}``"
    L_inv::AbstractMatrix{<:Real}

    function NatafTransformation(
        X::AbstractVector{<:Distributions.ContinuousUnivariateDistribution},
        ρ_X::AbstractMatrix{<:Real},
    )
        # Compute the number of dimensions:
        N_d = length(X)

        # Error-catching:
        size(ρ_X) == (N_d, N_d) || throw(
            DimensionMismatch(
                "Size of the correlation matrix is not compatible with the dimensionality of the random vector!",
            ),
        )
        maximum(abs.(ρ_X - I)) ≤ 1 || throw(
            ArgumentError(
                "Off-diagonal entries of the correlation matrix must be between -1 and +1!",
            ),
        )
        isposdef(ρ_X) ||
            throw(ArgumentError("Correlation matrix must be a positive-definite matrix!"))

        # Compute the distorted correlation matrix:
        ρ_Z, L, L_inv = getdistortedcorrelation(X, ρ_X)

        # Return the Nataf Transformation object with the computed distorted correlation matrix:
        return new(X, ρ_X, ρ_Z, L, L_inv)
    end
end
Base.broadcastable(x::NatafTransformation) = Ref(x)

"""
    getdistortedcorrelation(X::AbstractVector{<:Distributions.UnivariateDistribution}, ρ_X::AbstractMatrix{<:Real})

Function used to compute the distorted correlation matrix ``\\rho^{Z}``.
"""
function getdistortedcorrelation(
    X::AbstractVector{<:Distributions.ContinuousUnivariateDistribution},
    ρ_X::AbstractMatrix{<:Real},
)
    # Compute the number of dimensions:
    N_d = length(X)

    # Compute the locations and weights of the integration points in 1D:
    ρ_X_max = maximum(abs.(ρ_X - LinearAlgebra.I))
    num_ip = ρ_X_max ≤ 0.9 ? 64 : 1024
    ip_l_1d, ip_w_1d = gausslegendre(num_ip)

    # Transform the locations and weights of the integration points from 1D into 2D:
    ξ = Vector{Float64}(undef, num_ip^2)
    η = Vector{Float64}(undef, num_ip^2)
    W = Vector{Float64}(undef, num_ip^2)
    for i in 1:num_ip
        for j in 1:num_ip
            ξ[(i - 1) * num_ip + j] = ip_l_1d[i]
            η[(i - 1) * num_ip + j] = ip_l_1d[j]
            W[(i - 1) * num_ip + j] = ip_w_1d[i] * ip_w_1d[j]
        end
    end

    # Set the bounds of integration:
    z_min = -6
    z_max = +6

    # Perform change of interval:
    z_i = ((z_max - z_min) / 2) * ξ .+ (z_max + z_min) / 2
    z_j = ((z_max - z_min) / 2) * η .+ (z_max + z_min) / 2

    # Determine the common parameter type:
    parameters = params.(X)
    parameters = [parameters[i][j] for i in eachindex(X) for j in eachindex(parameters[i])]
    parameter_types = typeof.(parameters)
    common_parameter_type = promote_type(parameter_types...)

    # Compute the entries of the distorted correlation matrix:
    ρ_Z = Matrix{common_parameter_type}(I, N_d, N_d)
    ϕ = Normal()
    for i in 1:N_d
        for j in (i + 1):N_d
            # Check if the marginal distributions are uncorrelated:
            if ρ_X[i, j] == 0
                continue
            end

            # Define a function from which we will compute the entries of the distorted correlation matrix:
            h_i = (quantile.(X[i], cdf.(ϕ, z_i)) .- mean(X[i])) / std(X[i])
            h_j = (quantile.(X[j], cdf.(ϕ, z_j)) .- mean(X[j])) / std(X[j])
            F(ρ_Z, p) =
                ((z_max - z_min) / 2) ^ 2 * dot(
                    W .* (h_i .* h_j),
                    (
                        (1 / (2 * π * sqrt(1 - ρ_Z ^ 2))) * exp.(
                            (2 * ρ_Z * (z_i .* z_j) - z_i .^ 2 - z_j .^ 2) /
                            (2 * (1 - ρ_Z ^ 2)),
                        )
                    ),
                ) - ρ_X[i, j]

            # Compute the entries of the distorted correlation matrix:
            try
                problem = NonlinearSolve.NonlinearProblem(F, ρ_X[i, j])
                solution = NonlinearSolve.solve(problem, nothing)
                ρ_Z[i, j] = solution.u
            catch
                problem = NonlinearSolve.IntervalNonlinearProblem(
                    F, (-(1 - 1E-3), +(1 - 1E-3))
                )
                solution = NonlinearSolve.solve(problem, nothing)
                ρ_Z[i, j] = solution.u
            end
            ρ_Z[j, i] = ρ_Z[i, j]
        end
    end

    # Compute the lower triangular matrix of the Cholesky decomposition of the distorted correlation matrix and its inverse:
    L = cholesky(ρ_Z).L
    L_inv = inv(L)

    # Return the result:
    return ρ_Z, L, L_inv
end

function _compute_z_sample(X, x, ϕ)
    y = cdf(X, x)

    z = if y == 0
        invlogcdf(ϕ, logcdf(X, x))
    elseif y == 1
        invlogccdf(ϕ, logccdf(X, x))
    else
        quantile(ϕ, y)
    end

    return z
end

function _compute_x_sample(X, z, ϕ)
    y = cdf(ϕ, z)

    x = if y == 0
        invlogcdf(X, logcdf(ϕ, z))
    elseif y == 1
        invlogccdf(X, logccdf(ϕ, z))
    else
        quantile(X, y)
    end

    return x
end

"""
    transformsamples(transf_obj::NatafTransformation, samples::AbstractVector{<:Real}, transformation_dir::Symbol)

Function used to transform samples from ``X``- to ``U``-space and vice versa. \\
If `transformation_dir is:

  - `:X2U`, then the function transforms samples ``\\vec{x}`` from ``X``- to ``U``-space.
  - `:U2X`, then the function transforms samples ``\\vec{u}`` from ``U``- to ``X``-space.
"""
function transformsamples(
    transf_obj::NatafTransformation,
    samples::AbstractVector{<:Real},
    transformation_dir::Symbol,
)
    # Compute number of dimensions:
    N_d = length(samples)

    ϕ = Normal()

    if transformation_dir != :X2U && transformation_dir != :U2X
        throw(
            ArgumentError(
                "Invalid transformation direction! Available options are: `:X2U` and `:U2X`!",
            ),
        )
    elseif transformation_dir == :X2U
        # Extract data:
        X = transf_obj.X
        L_inv = transf_obj.L_inv

        # Convert samples of the marginal distributions X into the space of correlated standard normal random variables Z:
        Z_samples = [_compute_z_sample(X[i], samples[i], ϕ) for i in 1:N_d]

        # Convert samples from the space of correlated standard normal random variables Z into the space of uncorrelated standard normal random variables U:
        U_samples = L_inv * Z_samples

        # Return the result:
        return U_samples
    elseif transformation_dir == :U2X
        # Extract data:
        X = transf_obj.X
        L = transf_obj.L

        # Convert samples to the space of correlated standard normal random variables Z:
        Z_samples = L * samples

        # Convert samples of the correlated standard normal random variables Z into samples of the marginals:
        X_samples = [_compute_x_sample(X[i], Z_samples[i], ϕ) for i in 1:N_d]

        # Return the result:
        return X_samples
    end
end

function transformsamples(
    transf_obj::NatafTransformation,
    samples::AbstractMatrix{<:Real},
    transformation_dir::Symbol,
)
    # Compute number of samples and dimensions:
    N_d = size(samples, 1)

    # Error-catching:
    length(transf_obj.X) == N_d ||
        throw(ArgumentError("Number of dimensions of the provided samples is incorrect!"))

    trans_samples = transformsamples.(transf_obj, eachcol(samples), transformation_dir)
    trans_samples = hcat(trans_samples...)

    # Return the result:
    return trans_samples
end

"""
    getjacobian(transf_obj::NatafTransformation, samples::AbstractVector{<:Real}, transformation_dir::Symbol)

Function used to compute the Jacobians of the transformations of samples from ``X``- to ``U``-space and vice versa. \\
If `transformation_dir` is:

  - `:X2U`, then the function returns the Jacobians of the transformations of samples ``\\vec{x}`` from ``X``- to ``U``-space.
  - `:U2X`, then the function returns the Jacobians of the transformations of samples ``\\vec{u}`` from ``U``- to ``X``-space.
"""
function getjacobian(
    transf_obj::NatafTransformation,
    samples::AbstractVector{<:Real},
    transformation_dir::Symbol,
)
    # Compute number of dimensions:
    N_d = length(samples)

    ϕ = Normal()

    if transformation_dir != :X2U && transformation_dir != :U2X
        throw(
            ArgumentError(
                "Invalid transformation direction! Available options are: `:X2U` and `:U2X`!",
            ),
        )
    elseif transformation_dir == :X2U
        # Extract data:
        X = transf_obj.X
        L = transf_obj.L

        # Convert samples to the space of correlated standard normal random variables Z:
        Z_samples = [_compute_z_sample(X[i], samples[i], ϕ) for i in 1:N_d]

        # Compute the Jacobian:
        D = [exp(logpdf(ϕ, Z_samples[i]) - logpdf(X[i], samples[i])) for i in 1:N_d]
        D = diagm(D)
        J = D * L
    elseif transformation_dir == :U2X
        # Extract data:
        X = transf_obj.X
        L = transf_obj.L
        L_inv = transf_obj.L_inv

        # Convert samples to the space of correlated standard normal random variables Z:
        Z_samples = L * samples

        # Convert samples of the correlated standard normal random variables Z into samples of the marginals:
        X_samples = [_compute_x_sample(X[i], Z_samples[i], ϕ) for i in 1:N_d]

        # Compute the Jacobian:
        D = [exp(logpdf(X[i], X_samples[i]) - logpdf(ϕ, Z_samples[i])) for i in 1:N_d]
        D = diagm(D)
        J = L_inv * D
    end

    # Return the result:
    return J
end

function getjacobian(
    transf_obj::NatafTransformation,
    samples::AbstractMatrix{<:Real},
    transformation_dir::Symbol,
)
    # Compute number of dimensions and samples:
    N_d = size(samples, 1)

    # Error-catching:
    length(transf_obj.X) == N_d ||
        throw(ArgumentError("Number of dimensions of the provided samples is incorrect!"))

    # Compute the Jacobians:
    J = getjacobian.(transf_obj, eachcol(samples), transformation_dir)

    # Return the result:
    return J
end

"""
    pdf(transf_obj::NatafTransformation, x::AbstractVector{<:Real})

Function used to compute the joint PDF in ``X``-space.
"""
function Distributions.pdf(transf_obj::NatafTransformation, x::AbstractVector{<:Real})
    # Extract data:
    X = transf_obj.X
    ρ_Z = transf_obj.ρ_Z

    # Compute the number of samples and number of marginal distributions:
    N_d = length(x)

    ϕ = Normal()

    # Convert samples to the space of correlated standard normal random variables Z:
    z = [_compute_z_sample(X[i], x[i], ϕ) for i in 1:N_d]
    f_x_log = [logpdf(X[i], x[i]) for i in 1:N_d]
    ϕ_z_log = [logpdf(ϕ, z[i]) for i in 1:N_d]

    # Compute the joint PDF of samples in the space of correlated standard normal random variables Z: 
    pdf_z_log = logpdf(MvNormal(ρ_Z), z)

    # Compute the joint PDF:
    pdf_x = exp(pdf_z_log + sum(f_x_log) - sum(ϕ_z_log))

    # Clean up:
    if !isfinite(pdf_x)
        pdf_x = 0
    end

    # Return the result:
    return pdf_x
end

function Distributions.pdf(transf_obj::NatafTransformation, x::AbstractMatrix{<:Real})
    # Compute number of dimensions and samples:
    N_d = size(x, 1)

    # Error-catching:
    length(transf_obj.X) == N_d ||
        throw(ArgumentError("Number of dimensions of the provided samples is incorrect!"))

    # Compute the joint PDF:
    pdf_x = pdf.(transf_obj, eachcol(x))

    # Return the result:
    return pdf_x
end
