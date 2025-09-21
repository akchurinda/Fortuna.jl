"""
    SSM <: AbstractReliabililyAnalysisMethod

Type used to perform reliability analysis using Subset Simulation Method (SSM).

$(TYPEDFIELDS)
"""
Base.@kwdef struct SSM <: AbstractReliabililyAnalysisMethod
    "Probability of failure for each subset ``P_{0}``"
    p_0::Real = 0.10
    "Number of samples generated within each subset ``N``"
    num_samples::Integer = 1E6
    "Maximum number of subsets ``M``"
    max_num_subsets::Integer = 50
end

"""
    SSMCache

Type used to perform reliability analysis using Subset Simulation Method (SSM).

$(TYPEDFIELDS)
"""
struct SSMCache
    "Samples generated within each subset in ``X``-space"
    X_samples_subset::Vector{Matrix{Float64}}
    "Samples generated within each subset in ``U``-space"
    U_samples_subset::Vector{Matrix{Float64}}
    "Threshold for each subset ``C_{i}``"
    C_subset::Vector{Float64}
    "Probability of failure for each subset ``P_{f_{i}}``"
    PoF_subset::Vector{Float64}
    "Probability of failure ``P_{f}``"
    PoF::Float64
    "convergence status"
    convergence::Bool
end

"""
    solve(problem::ReliabilityProblem, analysis_method::SSM)

Function used to solve reliability problems using Subset Simulation Method (SSM).
"""
function solve(problem::ReliabilityProblem, analysis_method::SSM)
    # Extract analysis details:
    p_0 = analysis_method.p_0
    num_samples = analysis_method.num_samples
    max_num_subsets = analysis_method.max_num_subsets

    # Extract problem data:
    X = problem.X
    ρ_X = problem.ρ_X
    g = problem.g

    # Perform Nataf Transformation:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Compute number of dimensions: 
    num_dims = length(X)

    # Compute number of Markov chains within each subset:
    num_mcs = floor(Integer, p_0 * num_samples)

    # Compute number of samples to generate within each Markov chain:
    num_samples_chain = floor(Integer, num_samples / num_mcs)

    # Preallocate:
    U_samples_subset = Vector{Matrix{Float64}}()
    X_samples_subset = Vector{Matrix{Float64}}()
    C_subset = Vector{Float64}(undef, max_num_subsets)
    PoF_subset = Vector{Float64}(undef, max_num_subsets)
    convergence = true

    # Loop through each subset:
    for i in 1:max_num_subsets
        if i == 1
            # Generate samples in the standard normal space:
            U_samples = Distributions.randn(num_dims, num_samples)
        else
            # Preallocate:
            U_samples = zeros(num_dims, num_mcs * num_samples_chain)

            # Generate samples using the Modified Metropolis-Hastings algorithm:
            for j in 1:num_mcs
                U_samples[:, (num_samples_chain * (j - 1) + 1):(num_samples_chain * j)] = ModifiedMetropolisHastings(
                    U_samples_subset[i - 1][:, j],
                    C_subset[i - 1],
                    num_dims,
                    num_samples_chain,
                    nataf_obj,
                    g,
                )
            end
        end

        # Evaluate the limit state function at the generated samples:
        G_samples = G(g, nataf_obj, U_samples)

        # Sort the values of the limit state function:
        G_sample_sorted = sort(G_samples)

        # Compute the threshold:
        C_subset[i] = Distributions.quantile(G_sample_sorted, p_0)

        # Check for convergance:
        if C_subset[i] ≤ 0 || i == max_num_subsets
            # Check for convergance:
            if i == max_num_subsets
                @warn """
                SSM did not converge in the given maximum number of subsets (max_num_subsets = $max_num_subsets)! 
                Try increasing the maximum number of subsets (max_num_subsets), increasing the number samples generated within each subset (num_samples), or changing the probability of failure for each subset (p_0))!
                """

                convergence = false
            end

            # Redefine the threshold:
            C_subset[i] = 0

            # Retain samples below the threshold:
            idx = findall(x -> x ≤ C_subset[i], G_samples)
            push!(U_samples_subset, U_samples[:, idx])
            push!(X_samples_subset, transformsamples(nataf_obj, U_samples[:, idx], :U2X))

            # Compute the probability of failure:
            PoF_subset[i] = length(idx) / size(G_samples)[1]

            # Clean up the result:
            C_subset = C_subset[1:i]
            PoF_subset = PoF_subset[1:i]

            # Compute the final probability of failure:
            PoF = prod(PoF_subset)

            # Return the result:
            return SSMCache(
                X_samples_subset, U_samples_subset, C_subset, PoF_subset, PoF, convergence
            )
        else
            # Retain samples below the threshold:
            idx = findall(x -> x ≤ C_subset[i], G_samples)
            push!(U_samples_subset, U_samples[:, idx])
            push!(X_samples_subset, transformsamples(nataf_obj, U_samples[:, idx], :U2X))

            # Compute the probability of failure:
            PoF_subset[i] = length(idx) / size(G_samples)[1]
        end
    end
end

function ModifiedMetropolisHastings(
    start_point::Vector{Float64},
    curr_threshold::Float64,
    num_dims::Integer,
    num_samples_chain::Integer,
    nataf_obj::NatafTransformation,
    g::Function,
)
    # Preallocate:
    chain_samples = zeros(num_dims, num_samples_chain)
    chain_samples[:, 1] = start_point

    # Define a standard multivariate normal PDF:
    M = zeros(num_dims)
    Σ = Matrix(1.0 * LinearAlgebra.I, num_dims, num_dims)
    ϕ = Distributions.MvNormal(M, Σ)

    # Pregenerate uniformly-distributed samples:
    U = Distributions.rand(num_samples_chain)

    # Generate samples:
    for i in 1:(num_samples_chain - 1)
        # Define a proposal density:
        M = chain_samples[:, i]
        Σ = Matrix(1.0 * LinearAlgebra.I, num_dims, num_dims)
        q = Distributions.MvNormal(M, Σ)

        # Propose a new state:
        prop_state = Distributions.rand(q, 1)
        prop_state = vec(prop_state)

        # Compute the indicator function:
        X_prop_state = transformsamples(nataf_obj, prop_state, :U2X)
        G_prop_state = g(X_prop_state)
        I = G_prop_state ≤ curr_threshold ? 1 : 0

        # Compute the acceptance ratio:        
        α = (pdf(ϕ, prop_state) * I) / pdf(ϕ, chain_samples[:, i])

        # Accept or reject the proposed state:
        chain_samples[:, i + 1] = U[i] <= α ? prop_state : chain_samples[:, i]
    end

    # Return the result:
    return chain_samples
end

function G(g::Function, nataf_obj::NatafTransformation, U_samples::AbstractMatrix)
    # Transform the samples:
    X_samples = transformsamples(nataf_obj, U_samples, :U2X)

    # Clean up the transformed samples:
    X_samples_clean = eachcol(X_samples)
    X_samples_clean = Vector.(X_samples_clean)

    # Evaluate the limit state function at the transform samples:
    G_samples = g.(X_samples_clean)

    # Return the result:
    return G_samples
end
