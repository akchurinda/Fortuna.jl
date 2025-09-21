# # Inverse Transform Sampling:
# """
#     struct ITS <: AbstractSamplingTechnique

# Type used to perform the Inverse Transform Sampling.
# """
# struct ITS <: AbstractSamplingTechnique end

# # Latin Hypercube Sampling:
# """
#     struct LHS <: AbstractSamplingTechnique

# Type used to perform the Latin Hypercube Sampling.
# """
# struct LHS <: AbstractSamplingTechnique end

# --------------------------------------------------
# GENERATE SAMPLES FROM A RANDOM VARIABLE
# --------------------------------------------------
"""
    rand(rng::Distributions.AbstractRNG, random_variable::Distributions.UnivariateDistribution, num_samples::Integer, sampling_technique::Symbol)

Function used to generate samples from an *random variable*.
If `sampling_technique` is:

  - `:ITS` samples are generated using Inverse Transform Sampling technique.
  - `:LHS` samples are generated using Latin Hypercube Sampling technique.
"""
function Distributions.rand(
    rng::Distributions.AbstractRNG,
    random_variable::Distributions.ContinuousUnivariateDistribution,
    num_samples::Integer,
    sampling_technique::Symbol,
)
    # Generate samples:
    if (sampling_technique != :ITS) && (sampling_technique != :LHS)
        throw(ArgumentError("Provided sampling technique is not supported!"))
    elseif sampling_technique == :ITS
        # Generate samples:
        samples = Distributions.rand(rng, random_variable, num_samples)
    elseif sampling_technique == :LHS
        # Define the lower limits of each strata:
        lb = collect(range(0, (num_samples - 1) / num_samples, num_samples))

        # Generate samples within each strata:
        uniform_samples =
            lb +
            Distributions.rand(rng, Distributions.Uniform(0, 1 / num_samples), num_samples)

        # Shuffle samples:
        uniform_samples = Random.shuffle(rng, uniform_samples)

        # Generate samples:
        samples = Distributions.quantile.(random_variable, uniform_samples)
    end

    # Return the result:
    return samples
end

function Distributions.rand(
    random_variable::Distributions.ContinuousUnivariateDistribution,
    num_samples::Integer,
    sampling_technique::Symbol,
)
    Distributions.rand(
        Distributions.default_rng(), random_variable, num_samples, sampling_technique
    )
end

# --------------------------------------------------
# GENERATE SAMPLES FROM A RANDOM VECTOR
# --------------------------------------------------
"""
    rand(rng::Distributions.AbstractRNG, random_vector::Vector{<:Distributions.UnivariateDistribution}, num_samples::Integer, sampling_technique::Symbol)

Function used to generate samples from a *random vector with uncorrelated marginals*.
If `sampling_technique` is:

  - `:ITS` samples are generated using Inverse Transform Sampling technique.
  - `:LHS` samples are generated using Latin Hypercube Sampling technique.
"""
function Distributions.rand(
    rng::Distributions.AbstractRNG,
    random_vector::Vector{<:Distributions.ContinuousUnivariateDistribution},
    num_samples::Integer,
    sampling_technique::Symbol,
)
    # Compute number of dimensions:
    num_dims = length(random_vector)

    # Generate samples:
    samples = [
        Distributions.rand(rng, random_vector[i], num_samples, sampling_technique) for
        i in 1:num_dims
    ]
    samples = vcat(samples'...)

    # Return the result:
    return samples
end

function Distributions.rand(
    random_vector::Vector{<:Distributions.ContinuousUnivariateDistribution},
    num_samples::Integer,
    sampling_technique::Symbol,
)
    Distributions.rand(
        Distributions.default_rng(), random_vector, num_samples, sampling_technique
    )
end

# --------------------------------------------------
# GENERATE SAMPLES FROM A TRANSFORMATION OBJECT
# --------------------------------------------------
# Nataf Transformation:
"""
    rand(rng::Distributions.AbstractRNG, trans_obj::NatafTransformation, num_samples::Integer, sampling_technique::Symbol)

Function used to generate samples from a *random vector with correlated marginals* using Nataf Transformation object.
If `sampling_technique` is:

  - `:ITS` samples are generated using Inverse Transform Sampling technique.
  - `:LHS` samples are generated using Latin Hypercube Sampling technique.
"""
function Distributions.rand(
    rng::Distributions.AbstractRNG,
    trans_obj::NatafTransformation,
    num_samples::Integer,
    sampling_technique::Symbol,
)
    # Extract data:
    X = trans_obj.X
    L = trans_obj.L

    # Compute number of dimensions:
    num_dims = length(X)

    # Generate samples of uncorrelated normal random variables U:
    U_samples = [
        Distributions.rand(rng, Distributions.Normal(), num_samples, sampling_technique) for
        _ in 1:num_dims
    ]
    U_samples = vcat(U_samples'...)

    # Generate samples of correlated normal random variables Z:
    Z_samples = L * U_samples

    # Generate samples of correlated non-normal random variables X:
    X_samples = [
        Distributions.quantile.(
            X[i], Distributions.cdf.(Distributions.Normal(), Z_samples[i, :])
        ) for i in 1:num_dims
    ]
    X_samples = vcat(X_samples'...)

    return X_samples, Z_samples, U_samples
end

function Distributions.rand(
    trans_obj::NatafTransformation, num_samples::Integer, sampling_technique::Symbol
)
    Distributions.rand(
        Distributions.default_rng(), trans_obj, num_samples, sampling_technique
    )
end
