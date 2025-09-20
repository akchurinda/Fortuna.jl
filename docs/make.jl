using Fortuna
using Distributions, Random
using Documenter
using DocumenterCitations

bibliography = CitationBibliography(
    joinpath(@__DIR__, "src/references.bib"),
    style = :authoryear)

makedocs(
    sitename = "Fortuna.jl",
    authors  = "Damir Akchurin, akchurinda@gmail.com",
    format   = Documenter.HTML(
        assets = [
            "assets/favicon.ico",
            "assets/custom.css"]),
    pages    = [
        "Home" => "index.md",
        "Random Variables" => [
            "Defining Random Variables" => "Random Variables/DefineRandomVariables.md",
            "Sampling Random Variables" => "Random Variables/SampleRandomVariables.md"
        ],
        "Isoprobabilistic Transformations" => [
            "Nataf Transformation" => "Isoprobabilistic Transformations/NatafTransformation.md",
            "Rosenblatt Transformation" => "Isoprobabilistic Transformations/RosenblattTransformation.md"
        ],
        "Reliability Problems" => [
            "Reliability Problems" => "Reliability Problems/ReliabilityProblems.md",
            "Monte Carlo Methods" => [
                "Direct Monte Carlo Simulations" => "Reliability Problems/MC.md",
                "Importance Sampling" => "Reliability Problems/IS.md"
            ],
            "First-Order Reliability Methods" => "Reliability Problems/FORM.md",
            "Second-Order Reliability Methods" => "Reliability Problems/SORM.md",
            "Subset Simulation Method" => "Reliability Problems/SSM.md"
        ],
        "Inverse Reliability Problems" => "InverseReliabilityProblems.md",
        "Sensitivity Problems" => "SensitivityProblems.md",
        "Basic Examples" => "BasicExamples.md",
        "Research" => "Research.md",
        "References" => "References.md"
    ],
    plugins = [bibliography])

deploydocs(
    repo = "github.com/akchurinda/Fortuna.jl")