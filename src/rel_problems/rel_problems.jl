"""
    ReliabilityProblem <: AbstractReliabilityProblem

Type used to define reliability problems.

$(TYPEDFIELDS)
"""
mutable struct ReliabilityProblem <: AbstractReliabilityProblem
    "Random vector ``\\vec{X}``"
    X::AbstractVector{<:Distributions.UnivariateDistribution}
    "Correlation matrix ``\\rho^{X}``"
    ρ_X::AbstractMatrix{<:Real}
    "Limit state function ``g(\\vec{X})``"
    g::Function
end

include("mc.jl")
include("is.jl")
include("form.jl")
include("sorm.jl")
include("ssm.jl")
