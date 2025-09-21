using Fortuna
using DelimitedFiles

# Define the random variables:
X_1 = randomvariable("Normal", "M", [29000, 0.05 * 29000]) # Young's modulus
X_2 = randomvariable("Normal", "M", [110, 0.05 * 110]) # Moment of inertia about major axis
X = [X_1, X_2]

# Define the correlation matrix:
ρ_X = [1 0; 0 1]

# Define the FE model of the cantilever beam:
opensees_path = "C:\\Users\\...\\bin\\OpenSees"           # This must be an absolute path!
work_dir = "C:\\Users\\...\\Fortuna.jl\\examples\\OpenSees" # This must be an absolute path!
inp_filename = "beam.tcl"
out_filename = "output.out"
placeholders = [":E", ":I"]
function beam(x::Vector)
    # Inject values into the input file:
    inp_file_string = read(joinpath(work_dir, inp_filename), String)
    for (placeholder, val) in zip(placeholders, x)
        inp_file_string = replace(inp_file_string, placeholder => string(val))
    end

    # Write the modified input file:
    temp_inp_filename = replace(inp_filename, ".tcl" => "temp.tcl")
    write(joinpath(work_dir, temp_inp_filename), inp_file_string)

    # Run the model from the work directory:
    cd(work_dir)
    run(
        pipeline(
            `$(opensees_path) $(joinpath(work_dir, temp_inp_filename))`;
            stdout=devnull,
            stderr=devnull,
        ),
    )

    # Extract the output:
    Δ = -readdlm(joinpath(work_dir, out_filename))[end]

    # Delete the created files to prevent cluttering the work directory:
    rm(joinpath(work_dir, temp_inp_filename))
    rm(joinpath(work_dir, out_filename))

    # Return the result:
    return Δ
end

# Define the limit state function:
g(x::Vector) = 1 - beam(x)

# Define the reliability problem:
problem = ReliabilityProblem(X, ρ_X, g)

# Perform the reliability analysis using the FORM:
form_solution = solve(problem, FORM(); backend=AutoFiniteDiff())
println("FORM:")
println("β: $(form_solution.β)")
println("PoF: $(form_solution.PoF)")

# Perform the reliability analysis using the SORM:
sorm_solution = solve(
    problem, SORM(); form_solution=form_solution, backend=AutoFiniteDiff()
)
println("SORM:")
println("β: $(sorm_solution.β_2[1]) (Hohenbichler and Rackwitz)")
println("β: $(sorm_solution.β_2[2]) (Breitung)")
println("PoF: $(sorm_solution.PoF_2[1]) (Hohenbichler and Rackwitz)")
println("PoF: $(sorm_solution.PoF_2[2]) (Breitung)")
