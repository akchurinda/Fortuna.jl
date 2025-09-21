"""
    IS <: AbstractReliabililyAnalysisMethod

Type used to perform reliability analysis using Importance Sampling (IS) method.
"""
Base.@kwdef struct IS <: AbstractReliabililyAnalysisMethod
    "Proposal probability density function ``q``"
    q::Distributions.ContinuousMultivariateDistribution
    "Number of samples to generate ``N``"
    num_sims::Integer = 1E6
end

"""
    ISCache

Type used to store results of reliability analysis performed using Importance Sampling (IS) method.
"""
struct ISCache
    "samples generated from the proposal probability density function ``\\vec{x}_{i}``"
    samples::Matrix{Float64}
    "Target probability density function evaluated at each sample ``f(\\vec{x}_{i})``"
    f_vals::Vector{Float64}
    "Proposal probability density function evaluated at each sample ``q(\\vec{x}_{i})``"
    q_vals::Vector{Float64}
    "Limit state function evalued at each sample ``g(\\vec{x}_{i})``"
    g_vals::Vector{Float64}
    "Probability of failure ``P_{f}``"
    PoF::Float64
end

"""
    solve(problem::ReliabilityProblem, analysis_method::IS; showprogressbar = false)

Function used to solve reliability problems using Importance Sampling (IS) method.
"""
function solve(problem::ReliabilityProblem, analysis_method::IS; showprogressbar=false)
    # Extract the analysis details:
    q = analysis_method.q
    num_sims = analysis_method.num_sims

    # Extract data:
    g = problem.g
    X = problem.X
    ρ_X = problem.ρ_X

    # Error-catching:
    length(q) == length(X) || throw(
        DimensionMismatch(
            "Dimensionality of the proposal distribution does not match the dimensionality of the random vector!",
        ),
    )

    # If the marginal distrbutions are correlated, define a Nataf object:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Generate samples:
    samples = rand(q, num_sims)

    # Evaluate the target and proposal probability density functions at the generate samples:
    f_vals = pdf(nataf_obj, samples)
    q_vals = pdf(q, samples)

    # Evaluate the limit state function at the generate samples:
    g_vals = Vector{Float64}(undef, num_sims)
    ProgressMeter.@showprogress desc="Evaluating the limit state function..." enabled=showprogressbar for i in
                                                                                                          axes(
        samples, 2
    )
        g_vals[i]=g(samples[:, i])
    end

    # Evaluate the indicator function at the generate samples:
    I_vals = g_vals .≤ 0
    I_vals = Int.(I_vals)

    # Compute the probability of failure:
    PoF = mean((I_vals .* f_vals) ./ q_vals)

    # Return results:
    return ISCache(samples, f_vals, q_vals, g_vals, PoF)
end
