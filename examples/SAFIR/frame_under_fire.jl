using Fortuna
using DelimitedFiles

# Define the random variables:
X_1 = randomvariable("Normal", "M", [29000, 0.10 * 29000]) # Young's modulus
X_2 = randomvariable("Normal", "M", [2, 0.10 * 2]) # Lateral load
X = [X_1, X_2]

# Define the correlation matrix:
ρ_X = [1 0; 0 1]

# Define the FE model of the frame under fire:
work_dir = "C:\\Users\\...\\Fortuna.jl\\examples\\SAFIR" # This must be an absolute path!
inp_filename = "frame_under_fire.IN"
out_filename = "FrameUnderFireTemp.OUT"
placeholders = [":E", ":F"]
function frame_under_fire(x::Vector)
    # Inject values into the input file:
    inp_file_string = read(joinpath(work_dir, inp_filename), String)
    for (placeholder, val) in zip(placeholders, x)
        inp_file_string = replace(inp_file_string, placeholder => string(val))
    end

    # Write the modified input file:
    temp_inp_filename = replace(inp_filename, ".IN" => "temp.IN")
    write(joinpath(work_dir, temp_inp_filename), inp_file_string)

    # Run the model from the work directory:
    cd(work_dir)
    run(
        pipeline(
            `cmd /C "SAFIR $(replace(temp_inp_filename, ".IN" => ""))"`;
            stdout=devnull,
            stderr=devnull,
        ),
    )

    # Extract the output:
    out_file_string = read(out_filename, String)
    s_idx = findlast(
        "TOTAL DISPLACEMENTS.\r\n --------------------\r\n NODE    DOF 1     DOF 2     DOF 3     DOF 4     DOF 5     DOF 6     DOF 7\r\n",
        out_file_string,
    )
    out_file_string = out_file_string[(s_idx[end] + 1):end]
    f_idx = findfirst("\r\n\r\n", out_file_string)
    out_file_string = out_file_string[1:(f_idx[1] - 1)]
    write(joinpath(work_dir, out_filename), out_file_string)
    Δ = -readdlm(joinpath(work_dir, out_filename))[38, 3]

    # Delete the created files to prevent cluttering the work directory:
    rm(joinpath(work_dir, temp_inp_filename))
    rm(joinpath(work_dir, out_filename))

    # Return the result:
    return Δ
end

# Define the limit state function:
g(x::Vector) = 0.075 - frame_under_fire(x)

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
