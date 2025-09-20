using Test
using Fortuna
using Random
using Distributions

@testset "Random Variables" begin
    include("test_rvs.jl")
end

@testset "Sampling Techniques" begin
    include("test_sampling_techniques.jl")
end

@testset "Nataf Transformation" begin
    include("test_nataf_transf.jl")
end

@testset "Monte Carlo" begin
    include("test_mc.jl")
end

@testset "Importance Sampling" begin
    include("test_is.jl")
end

@testset "First-Order Reliability Method" begin
    include("test_form.jl")
end

@testset "Second-Order Reliability Method" begin
    include("test_sorm.jl")
end

@testset "Subset Simulation Method" begin
    include("test_ssm.jl")
end

@testset "Sensitivity Problems" begin
    include("test_sen_rel_problems.jl")
end

@testset "Inverse Reliability Problems" begin
    include("test_inv_rel_problems.jl")
end