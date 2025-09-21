@testset "Random Variables" begin
    @testset "Exponential" begin
        moments = zeros(10, 2)
        for i in 1:10
            moments[i, 1] = i
            moments[i, 2] = i
        end

        rvs = [randomvariable("Exponential", "M", moments[i, :]) for i in axes(moments, 1)]

        @test isapprox(hcat(mean.(rvs), std.(rvs)), moments, rtol=10 ^ (-9))
    end

    @testset "Frechet" begin
        μ = 1:10
        σ = 0.1:0.1:1
        moments = zeros(length(μ) * length(σ), 2)
        for i in eachindex(μ)
            for j in eachindex(σ)
                moments[(i - 1) * length(σ) + j, 1] = μ[i]
                moments[(i - 1) * length(σ) + j, 2] = σ[j]
            end
        end

        rvs = [randomvariable("Frechet", "M", moments[i, :]) for i in axes(moments, 1)]

        @test isapprox(hcat(mean.(rvs), std.(rvs)), moments, rtol=10 ^ (-9))
    end

    @testset "Gamma" begin
        μ = 1:10
        σ = 0.1:0.1:1
        moments = zeros(length(μ) * length(σ), 2)
        for i in eachindex(μ)
            for j in eachindex(σ)
                moments[(i - 1) * length(σ) + j, 1] = μ[i]
                moments[(i - 1) * length(σ) + j, 2] = σ[j]
            end
        end

        rvs = [randomvariable("Gamma", "M", moments[i, :]) for i in axes(moments, 1)]

        @test isapprox(hcat(mean.(rvs), std.(rvs)), moments, rtol=10 ^ (-9))
    end

    @testset "Gumbel" begin
        μ = 1:10
        σ = 0.1:0.1:1
        moments = zeros(length(μ) * length(σ), 2)
        for i in eachindex(μ)
            for j in eachindex(σ)
                moments[(i - 1) * length(σ) + j, 1] = μ[i]
                moments[(i - 1) * length(σ) + j, 2] = σ[j]
            end
        end

        rvs = [randomvariable("Gumbel", "M", moments[i, :]) for i in axes(moments, 1)]

        @test isapprox(hcat(mean.(rvs), std.(rvs)), moments, rtol=10 ^ (-9))
    end

    @testset "LogNormal" begin
        μ = 1:10
        σ = 0.1:0.1:1
        moments = zeros(length(μ) * length(σ), 2)
        for i in eachindex(μ)
            for j in eachindex(σ)
                moments[(i - 1) * length(σ) + j, 1] = μ[i]
                moments[(i - 1) * length(σ) + j, 2] = σ[j]
            end
        end

        rvs = [randomvariable("LogNormal", "M", moments[i, :]) for i in axes(moments, 1)]

        @test isapprox(hcat(mean.(rvs), std.(rvs)), moments, rtol=10 ^ (-9))
    end

    @testset "Normal" begin
        μ = 1:10
        σ = 0.1:0.1:1
        moments = zeros(length(μ) * length(σ), 2)
        for i in eachindex(μ)
            for j in eachindex(σ)
                moments[(i - 1) * length(σ) + j, 1] = μ[i]
                moments[(i - 1) * length(σ) + j, 2] = σ[j]
            end
        end

        rvs = [randomvariable("Normal", "M", moments[i, :]) for i in axes(moments, 1)]

        @test isapprox(hcat(mean.(rvs), std.(rvs)), moments, rtol=10 ^ (-9))
    end

    @testset "Poisson" begin
        moments = zeros(10, 2)
        for i in 1:10
            moments[i, 1] = i
            moments[i, 2] = sqrt(i)
        end

        rvs = [randomvariable("Poisson", "M", moments[i, :]) for i in axes(moments, 1)]

        @test isapprox(hcat(mean.(rvs), std.(rvs)), moments, rtol=10 ^ (-9))
    end

    @testset "Uniform" begin
        μ = 1:10
        σ = 0.1:0.1:1
        moments = zeros(length(μ) * length(σ), 2)
        for i in eachindex(μ)
            for j in eachindex(σ)
                moments[(i - 1) * length(σ) + j, 1] = μ[i]
                moments[(i - 1) * length(σ) + j, 2] = σ[j]
            end
        end

        rvs = [randomvariable("Uniform", "M", moments[i, :]) for i in axes(moments, 1)]

        @test isapprox(hcat(mean.(rvs), std.(rvs)), moments, rtol=10 ^ (-9))
    end

    @testset "Weibull" begin
        μ = 1:10
        σ = 0.1:0.1:1
        moments = zeros(length(μ) * length(σ), 2)
        for i in eachindex(μ)
            for j in eachindex(σ)
                moments[(i - 1) * length(σ) + j, 1] = μ[i]
                moments[(i - 1) * length(σ) + j, 2] = σ[j]
            end
        end

        rvs = [randomvariable("Weibull", "M", moments[i, :]) for i in axes(moments, 1)]

        @test isapprox(hcat(mean.(rvs), std.(rvs)), moments, rtol=10 ^ (-9))
    end
end
