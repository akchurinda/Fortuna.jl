@testset "Sampling Techniques #1" begin
    # Set an RNG seed:
    Random.seed!(123)

    # Generate a random vector X with uncorrelated marginal random variables X_1 and X_2:
    X_1 = randomvariable("Gamma", "M", [10, 1.5])
    X_2 = randomvariable("Gamma", "M", [15, 2.5])
    X  = [X_1, X_2]

    # Generate samples:
    num_samples  = 10 ^ 6
    x_samples_its = rand(X, num_samples, :ITS)
    x_samples_lhs = rand(X, num_samples, :LHS)

    # Test the results:
    @test isapprox(mean(x_samples_its[1, :]),    10,         rtol = 1E-2) # Inverse Transform Sampling
    @test isapprox(mean(x_samples_its[2, :]),    15,         rtol = 1E-2)
    @test isapprox(std(x_samples_its[1, :]),     1.5,        rtol = 1E-2)
    @test isapprox(std(x_samples_its[2, :]),     2.5,        rtol = 1E-2)
    @test isapprox(cor(x_samples_its, dims = 2), [1 0; 0 1], rtol = 1E-2)
    @test isapprox(mean(x_samples_lhs[1, :]),    10,         rtol = 1E-2) # Latin Hypercube Sampling
    @test isapprox(mean(x_samples_lhs[2, :]),    15,         rtol = 1E-2)
    @test isapprox(std(x_samples_lhs[1, :]),     1.5,        rtol = 1E-2)
    @test isapprox(std(x_samples_lhs[2, :]),     2.5,        rtol = 1E-2)
    @test isapprox(cor(x_samples_lhs, dims = 2), [1 0; 0 1], rtol = 1E-2)
end

@testset "Sampling Techniques #2" begin
    # Set an RNG seed:
    Random.seed!(123)

    # Define a list of reliability indices of interest:
    ρ_list = (-0.75):(0.25):(+0.75)

    for i in eachindex(ρ_list)
        # Generate a random vector X of correlated marginal distributions:
        X_1 = randomvariable("Gamma", "M", [10, 1.5])
        X_2 = randomvariable("Gamma", "M", [15, 2.5])
        X  = [X_1, X_2]
        ρ_X = [1 ρ_list[i]; ρ_list[i] 1]

        # Perform Nataf transformation of the correlated marginal random variables:
        nataf_obj = NatafTransformation(X, ρ_X)

        # Generate samples:
        num_samples        = 10 ^ 6
        x_samples_its, _, _ = rand(nataf_obj, num_samples, :ITS)
        x_samples_lhs, _, _ = rand(nataf_obj, num_samples, :LHS)

        # Test the results:
        @test isapprox(mean(x_samples_its[1, :]),    10,  rtol = 1E-2) # Inverse Transform Sampling
        @test isapprox(mean(x_samples_its[2, :]),    15,  rtol = 1E-2)
        @test isapprox(std(x_samples_its[1, :]),     1.5, rtol = 1E-2)
        @test isapprox(std(x_samples_its[2, :]),     2.5, rtol = 1E-2)
        @test isapprox(cor(x_samples_its, dims = 2), ρ_X,  rtol = 1E-2)
        @test isapprox(mean(x_samples_lhs[1, :]),    10,  rtol = 1E-2) # Latin Hypercube Sampling
        @test isapprox(mean(x_samples_lhs[2, :]),    15,  rtol = 1E-2)
        @test isapprox(std(x_samples_lhs[1, :]),     1.5, rtol = 1E-2)
        @test isapprox(std(x_samples_lhs[2, :]),     2.5, rtol = 1E-2)
        @test isapprox(cor(x_samples_lhs, dims = 2), ρ_X,  rtol = 1E-2)
    end
end

@testset "Sampling Techniques #3" begin
    # Generate a random vector X of correlated marginal distributions:
    X_1 = randomvariable("Gamma", "M", [10, 1.5])
    X_2 = randomvariable("Gamma", "M", [15, 2.5])
    X  = [X_1, X_2]
    ρ_X = [1 0.75; 0.75 1]

    # Perform Nataf transformation of the correlated marginal random variables:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Define number of samples:
    num_samples = 10 ^ 6

    # Generate samples from a random variable:
    Random.seed!(123)
    x_samples_its_1 = rand(X_1, num_samples, :ITS)
    x_samples_lhs_1 = rand(X_1, num_samples, :LHS)

    Random.seed!(123)
    x_samples_its_2 = rand(X_1, num_samples, :ITS)
    x_samples_lhs_2 = rand(X_1, num_samples, :LHS)

    # Test the results:
    @test x_samples_its_1 == x_samples_its_2
    @test x_samples_lhs_1 == x_samples_lhs_2

    # Generate samples from a random variable:
    Random.seed!(123)
    x_samples_its_1 = rand(X_2, num_samples, :ITS)
    x_samples_lhs_1 = rand(X_2, num_samples, :LHS)

    Random.seed!(123)
    x_samples_its_2 = rand(X_2, num_samples, :ITS)
    x_samples_lhs_2 = rand(X_2, num_samples, :LHS)

    # Test the results:
    @test x_samples_its_1 == x_samples_its_2
    @test x_samples_lhs_1 == x_samples_lhs_2

    # Generate samples from a random vector:
    Random.seed!(123)
    x_samples_its_1 = rand(X, num_samples, :ITS)
    x_samples_lhs_1 = rand(X, num_samples, :LHS)

    Random.seed!(123)
    x_samples_its_2 = rand(X, num_samples, :ITS)
    x_samples_lhs_2 = rand(X, num_samples, :LHS)

    # Test the results:
    @test x_samples_its_1 == x_samples_its_2
    @test x_samples_lhs_1 == x_samples_lhs_2

    # Generate samples from a transformation object:
    Random.seed!(123)
    x_samples_its_1, z_samples_its_1, u_samples_its_1 = rand(nataf_obj, num_samples, :ITS)
    x_samples_lhs_1, z_samples_lhs_1, u_samples_lhs_1 = rand(nataf_obj, num_samples, :LHS)

    Random.seed!(123)
    x_samples_its_2, z_samples_its_2, u_samples_its_1 = rand(nataf_obj, num_samples, :ITS)
    x_samples_lhs_2, z_samples_lhs_2, u_samples_lhs_1 = rand(nataf_obj, num_samples, :LHS)

    # Test the results:
    @test x_samples_its_1 == x_samples_its_2
    @test z_samples_its_1 == z_samples_its_2
    @test u_samples_its_1 == u_samples_its_1
    @test x_samples_lhs_1 == x_samples_lhs_2
    @test z_samples_lhs_1 == z_samples_lhs_2
    @test u_samples_lhs_1 == u_samples_lhs_1
end