module Fortuna
# --------------------------------------------------
# IMPORT PACKAGES
# --------------------------------------------------
using Base
using Distributions
using FastGaussQuadrature
using DifferentiationInterface, ADTypes
using LinearAlgebra
using NonlinearSolve
using Random
using SpecialFunctions
using DocStringExtensions
using ProgressMeter

# --------------------------------------------------
# REEXPORT PACKAGES
# --------------------------------------------------
using Reexport
Reexport.@reexport using Distributions: rand, pdf              # Extended functions
Reexport.@reexport using Distributions: mean, std, cor, params # Useful functions
Reexport.@reexport using LinearAlgebra: I
Reexport.@reexport using ADTypes: AutoForwardDiff, AutoReverseDiff, AutoFiniteDiff

# --------------------------------------------------
# DEFINE ABSTRACT TYPES
# --------------------------------------------------
"""
    abstract type AbstractIsoprobabilisticTransformation end

Abstract type for isoprobabilistic transformations.
"""
abstract type AbstractIsoprobabilisticTransformation end

# """
#     abstract type AbstractSamplingTechnique end

# Abstract type for sampling techniques.
# """
# abstract type AbstractSamplingTechnique end

"""
    abstract type AbstractReliabilityProblem end

Abstract type for reliability problems.
"""
abstract type AbstractReliabilityProblem end

"""
    abstract type AbstractReliabililyAnalysisMethod end

Abstract type for reliability analysis methods.
"""
abstract type AbstractReliabililyAnalysisMethod end

"""
    abstract type FORMSubmethod end

Abstract type for First-Order Reliability Method's (FORM) submethods.
"""
abstract type FORMSubmethod end

"""
    abstract type SORMSubmethod end

Abstract type for Second-Order Reliability Method's (FORM) submethods.
"""
abstract type SORMSubmethod end

# --------------------------------------------------
# EXPORT TYPES AND FUNCTIONS
# --------------------------------------------------
include("transf/nataf_transf.jl")
include("transf/rosen_transf.jl")
include("rvs/define_rvs.jl")
include("rvs/sample_rvs.jl")
include("rel_problems/rel_problems.jl")
include("inv_rel_problems.jl")
include("sen_rel_problems.jl")
export AbstractSamplingTechnique
export ITS, LHS
export AbstractTransformation
export NatafTransformation, RosenblattTransformation
export AbstractReliabilityProblem
export ReliabilityProblem
export MC, MCCache
export IS, ISCache
export FORM, MCFOSM, MCFOSMCache, RF, RFCache, HLRF, HLRFCache, iHLRF, iHLRFCache
export SORM, CF, CFCache, PF, PFCache
export SSM, SSMCache
export SensitivityProblemTypeI, SensitivityProblemTypeII, SensitivityProblemCache
export InverseReliabilityProblem, InverseReliabilityProblemCache
export randomvariable
export getdistortedcorrelation, transformsamples, getjacobian
export solve
end
