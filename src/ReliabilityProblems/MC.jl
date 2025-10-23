"""
    MC <: AbstractReliabililyAnalysisMethod

Type used to perform reliability analysis using Monte Carlo (MC) simulations.

$(TYPEDFIELDS)
"""
Base.@kwdef struct MC <: AbstractReliabililyAnalysisMethod
    "Number of simulations ``N``"
    num_sims::Integer = 1E6
    "Sampling technique"
    sampling_technique::Symbol = :LHS
end

"""
    MCCache

Type used to store results of reliability analysis performed using Monte Carlo (MC) simulations.

$(TYPEDFIELDS)
"""
struct MCCache
    "Generated samples ``\\vec{x}_{i}``"
    samples::Matrix{Float64}
    "Limit state function evalued at each sample ``g(\\vec{x}_{i})``"
    g_vals::Vector{Float64}
    "Probability of failure ``P_{f}``"
    PoF::Float64
end

"""
    solve(problem::ReliabilityProblem, analysis_method::MC; showprogressbar = false)

Function used to solve reliability problems using Monte Carlo (MC) simulations.
"""
function solve(problem::ReliabilityProblem, analysis_method::MC; showprogressbar=false)
    # Extract the analysis details:
    num_sims = analysis_method.num_sims
    sampling_technique = analysis_method.sampling_technique

    # Extract data:
    g = problem.g
    X = problem.X
    ρ_X = problem.ρ_X

    # If the marginal distrbutions are correlated, define a Nataf object:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Generate samples:
    if (sampling_technique != :ITS) && (sampling_technique != :LHS)
        throw(ArgumentError("Provided sampling technique is not supported!"))
    else
        samples, _, _ = rand(nataf_obj, num_sims, sampling_technique)
    end

    # Evaluate the limit state function at the generate samples:
    g_vals = Vector{Float64}(undef, num_sims)
    ProgressMeter.@showprogress desc="Evaluating the limit state function..." enabled=showprogressbar for i in
                                                                                                          axes(
        samples, 2
    )
        g_vals[i]=g(samples[:, i])
    end

    # Compute the probability of failure:
    PoF = count(x -> x ≤ 0, g_vals) / num_sims

    # Return results:
    return MCCache(samples, g_vals, PoF)
end
