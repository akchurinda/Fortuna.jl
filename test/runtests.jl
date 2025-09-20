using Test
using Fortuna
using Random
using Distributions

@testset "Random Variables" begin
    include("TestRandomVariables.jl")
end

@testset "Sampling Techniques" begin
    include("TestSamplingTechniques.jl")
end

@testset "Nataf Transformation" begin
    include("TestNatafTransformation.jl")
end

@testset "Monte Carlo" begin
    include("TestMC.jl")
end

@testset "Importance Sampling" begin
    include("TestIS.jl")
end

@testset "First-Order Reliability Method" begin
    include("TestFORM.jl")
end

@testset "Second-Order Reliability Method" begin
    include("TestSORM.jl")
end

@testset "Subset Simulation Method" begin
    include("TestSSM.jl")
end

@testset "Sensitivity Problems" begin
    include("TestSensitivityProblems.jl")
end

@testset "Inverse Reliability Problems" begin
    include("TestInverseReliabilityProblems.jl")
end