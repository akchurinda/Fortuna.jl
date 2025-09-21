@testset "Nataf Transformation: Distorted Correlation Matrix #1 (Identity)" begin
    # Test from UQPy package (https://github.com/SURGroup/UQpy/tree/master)

    # Define a random vector:
    X_1  = randomvariable("Normal", "M", [0, 1])
    X_2  = randomvariable("Normal", "M", [0, 1])
    X   = [X_1, X_2]
    ρ_X  = [1 0; 0 1]

    # Perform Nataf transformation:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Test the results:
    ρ_Z = nataf_obj.ρ_Z
    @test ρ_Z == ρ_X
end

@testset "Nataf Transformation: Distorted Correlation Matrix #2 (Identity)" begin
    # Test from UQPy package (https://github.com/SURGroup/UQpy/tree/master)

    # Define a random vector:
    X_1  = randomvariable("Uniform", "P", [0, 1])
    X_2  = randomvariable("Uniform", "P", [0, 1])
    X   = [X_1, X_2]
    ρ_X  = [1 0; 0 1]

    # Perform Nataf transformation:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Test the results:
    ρ_Z = nataf_obj.ρ_Z
    @test ρ_Z == ρ_X
end

@testset "Nataf Transformation: Distorted Correlation Matrix #3 (Non-Identity)" begin
    # Test from UQPy package (https://github.com/SURGroup/UQpy/tree/master)

    # Define a random vector:
    X_1  = randomvariable("Normal", "M", [0, 1])
    X_2  = randomvariable("Normal", "M", [0, 1])
    X   = [X_1, X_2]
    ρ_X  = [1 0.8; 0.8 1]

    # Perform Nataf transformation:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Test the results:
    ρ_Z = nataf_obj.ρ_Z
    @test isapprox(ρ_Z, [1 0.8; 0.8 1], rtol = 1E-6)
end

@testset "Nataf Transformation: Distorted Correlation Matrix #4 (Non-Identity)" begin
    # Test from UQPy package (https://github.com/SURGroup/UQpy/tree/master)

    # Define a random vector:
    X_1  = randomvariable("Uniform", "P", [0, 1])
    X_2  = randomvariable("Uniform", "P", [0, 1])
    X   = [X_1, X_2]
    ρ_X  = [1 0.8; 0.8 1]

    # Perform Nataf transformation:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Test the results:
    ρ_Z = nataf_obj.ρ_Z
    @test isapprox(ρ_Z, [1 0.8134732861515996; 0.8134732861515996 1], rtol = 1E-6)
end

@testset "Nataf Transformation: Transformed samples #1 (Identity)" begin
    # Define a random vector:
    X_1  = randomvariable("Normal", "M", [0, 1])
    X_2  = randomvariable("Normal", "M", [0, 1])
    X    = [X_1, X_2]
    ρ_X  = [1 0; 0 1]

    # Perform Nataf transformation:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Define samples:
    x_1_range = range(-6, 6, length = 100)
    x_2_range = range(-6, 6, length = 100)
    samples = Matrix{Float64}(undef, 2, 100 * 100)
    for i in 1:100
        for j in 1:100
            samples[1, (i - 1) * 100 + j] = x_1_range[i]
            samples[2, (i - 1) * 100 + j] = x_2_range[j]
        end
    end

    # Perform transformation:
    trans_samples_x_to_u = transformsamples(nataf_obj, samples, :X2U)
    trans_samples_u_to_x = transformsamples(nataf_obj, samples, :U2X)

    # Test the results:
    @test all([isapprox(trans_samples_x_to_u[:, i], samples[:, i], rtol = 1E-6) for i in 1:10000])
    @test all([isapprox(trans_samples_u_to_x[:, i], samples[:, i], rtol = 1E-6) for i in 1:10000])
    @test trans_samples_x_to_u == trans_samples_u_to_x
end

@testset "Nataf Transformation: Jacobians #1 (Identity)" begin
    # Define a random vector:
    X_1  = randomvariable("Normal", "M", [0, 1])
    X_2  = randomvariable("Normal", "M", [0, 1])
    X   = [X_1, X_2]
    ρ_X  = [1 0; 0 1]

    # Perform Nataf transformation:
    nataf_obj = NatafTransformation(X, ρ_X)

    # Define samples:
    x_1_range = range(-6, 6, length = 100)
    x_2_range = range(-6, 6, length = 100)
    samples = Matrix{Float64}(undef, 2, 100 * 100)
    for i in 1:100
        for j in 1:100
            samples[1, (i - 1) * 100 + j] = x_1_range[i]
            samples[2, (i - 1) * 100 + j] = x_2_range[j]
        end
    end

    # Perform transformation:
    J_x_to_u = getjacobian(nataf_obj, samples, :X2U)
    J_u_to_x = getjacobian(nataf_obj, samples, :U2X)

    # Test the results:
    @test all([isapprox(J_x_to_u[i], [1 0; 0 1], rtol = 1E-6) for i in 1:10000])
    @test all([isapprox(J_u_to_x[i], [1 0; 0 1], rtol = 1E-6) for i in 1:10000])
    @test J_x_to_u == J_u_to_x
end