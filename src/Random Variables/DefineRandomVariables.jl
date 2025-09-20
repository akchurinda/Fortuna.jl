"""
    randomvariable(distribution::AbstractString, by::AbstractString, values::Union{Real, AbstractVector{<:Real}})

Function used to define random variables.
"""
function randomvariable(distribution::AbstractString, by::AbstractString, values::Union{Real, AbstractVector{<:Real}})
    # Supported distribution:
    supp_distributions = ["Exponential", "Frechet", "Gamma", "Gumbel", "LogNormal", "Normal", "Poisson", "Uniform", "Weibull"]

    # Error-catching:
    distribution in supp_distributions || throw(ArgumentError("Provided distribution is not supported!"))
    (by == "M" || by == "P")   || throw(ArgumentError("""Random variables can only be defined by "moments" ("M") and "params ("P")"!"""))

    # Convert moments into parameters if needed:
    by == "M" && (values = convertmoments(distribution, values))

    # Define a random variable:
    random_variables = getfield((@__MODULE__).Distributions, Symbol(distribution))(values...)

    # Return the result:
    return random_variables
end

function convertmoments(distribution::AbstractString, moments::Union{Real, AbstractVector{<:Real}})
    # Error-catching:
    length(moments) == 2 || throw(ArgumentError("Too few or many moments are provided! Provide only the mean (μ) and standard deviation (σ) in a vector format (i.e., [μ, σ])!"))

    # Convert moments to parameters:
    if distribution == "Exponential"
        # Extract moments:
        mean = moments[1]
        std  = moments[2]

        # Error catching:
        if mean != std
            throw(DomainError(moments, "mean and standard deviation values of must be the same!"))
        end

        # Convert moments to parameters:
        θ          = mean
        params = θ
    end

    if distribution == "Frechet"
        # Extract moments:
        mean = moments[1]
        std  = moments[2]

        # Convert moments to parameters:
        FFrechet(u, p) = sqrt(SpecialFunctions.gamma(1 - 2 / u) - SpecialFunctions.gamma(1 - 1 / u) ^ 2) / SpecialFunctions.gamma(1 - 1 / u) - p
        u_0             = (2 + 1E-1, 1E+6)
        p_0             = std / mean
        problem        = NonlinearSolve.IntervalNonlinearProblem(FFrechet, u_0, p_0)
        solution       = NonlinearSolve.solve(problem, nothing, abstol = 1E-9, reltol = 1E-9)
        α              = solution.u
        if !isapprox(FFrechet(α, p_0), 0, atol = 1E-9)
            throw(DomainError(moments, "Conversion of the provided moments to parameters has failed!"))
        end
        θ              = mean / SpecialFunctions.gamma(1 - 1 / α)
        params     = [α, θ]
    end

    if distribution == "Gamma"
        # Extract moments:
        mean = moments[1]
        std  = moments[2]

        # Convert moments to parameters:
        α          = mean ^ 2 / std ^ 2
        θ          = std ^ 2 / mean
        params = [α, θ]
    end
    
    if distribution == "Gumbel"
        # Extract moments:
        mean = moments[1]
        std  = moments[2]

        # Convert moments to parameters:
        γ          = Base.MathConstants.eulergamma
        μ          = mean - (std * γ * sqrt(6)) / π
        θ          = (std * sqrt(6)) / π
        params = [μ, θ]
    end
    
    if distribution == "LogNormal"
        # Extract moments:
        mean = moments[1]
        std  = moments[2]

        # Convert moments to parameters:
        μ          = log(mean) - log(sqrt(1 + (std / mean) ^ 2))
        σ          = sqrt(log(1 + (std / mean) ^ 2))
        params = [μ, σ]
    end
    
    if distribution == "Normal"
        # Extract moments:
        mean = moments[1]
        std  = moments[2]

        # Convert moments to parameters:
        μ          = mean
        σ          = std
        params = [μ, σ]
    end
    
    if distribution == "Poisson"
        # Extract moments:
        mean = moments[1]
        std  = moments[2]

        # Error catching:
        if !(mean ≈ (std ^ 2))
            throw(DomainError(moments, "Standard deviation must be equal to square root of mean!"))
        end

        # Convert moments to parameters:
        λ          = mean
        params = λ
    end
    
    if distribution == "Uniform"
        # Extract moments:
        mean = moments[1]
        std  = moments[2]

        # Convert moments to parameters:
        a          = mean - std * sqrt(3)
        b          = mean + std * sqrt(3)
        params = [a, b]
    end
    
    if distribution == "Weibull"
        # Extract moments:
        mean = moments[1]
        std  = moments[2]

        # Convert moments to parameters:
        FWeibull(u, p) = sqrt(SpecialFunctions.gamma(1 + 2 / u) - SpecialFunctions.gamma(1 + 1 / u) ^ 2) / SpecialFunctions.gamma(1 + 1 / u) - p
        u_0             = (1E-1, 1E+6)
        p_0             = std / mean
        problem        = NonlinearSolve.IntervalNonlinearProblem(FWeibull, u_0, p_0)
        solution       = NonlinearSolve.solve(problem, nothing, abstol = 1E-9, reltol = 1E-9)
        α              = solution.u
        if !isapprox(FWeibull(α, p_0), 0, atol = 1E-9)
            throw(DomainError(moments, "Conversion of the provided moments to parameters has failed!"))
        end
        θ          = mean / SpecialFunctions.gamma(1 + 1 / α)
        params = [α, θ]
    end

    # Return the result:
    return params
end