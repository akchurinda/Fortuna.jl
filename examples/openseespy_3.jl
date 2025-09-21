# Preamble:
using Fortuna
using PyCall

# Load OpenSeesPy package:
ops = pyimport("openseespy.opensees")

# Define the random variables:
X_1 = randomvariable("Normal", "M", [29000, 0.05 * 29000]) # Young's modulus
X_2 = randomvariable("Normal", "M", [110, 0.05 * 110]) # Moment of inertia about major-axis
X = [X_1, X_2]

# Define the correlation matrix:
ρ_X = [1 0; 0 1]

# Define the limit state function:
function g(x::Vector)
    # Remove any previous models:
    ops.wipe()

    # Define the model parameters:
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # Define the nodes:
    ops.node(1, 0 * 18, 0)
    ops.node(2, 1 * 18, 0)
    ops.node(3, 2 * 18, 0)
    ops.node(4, 3 * 18, 0)
    ops.node(5, 4 * 18, 0)
    ops.node(6, 5 * 18, 0)
    ops.node(7, 6 * 18, 0)
    ops.node(8, 7 * 18, 0)
    ops.node(9, 8 * 18, 0)
    ops.node(10, 9 * 18, 0)
    ops.node(11, 10 * 18, 0)

    # Define the boundary conditions:
    ops.fix(1, 1, 1, 1)

    # Define the material properties:
    ops.uniaxialMaterial("Elastic", 1, 29000)

    # Define the cross-sectional properties:
    A = 9.12
    ops.section("Elastic", 1, x[1], A, x[2])

    # Define the transformation:
    ops.geomTransf("PDelta", 1)

    # Define the elements:
    ops.element("elasticBeamColumn", 1, 1, 2, 1, 1)
    ops.element("elasticBeamColumn", 2, 2, 3, 1, 1)
    ops.element("elasticBeamColumn", 3, 3, 4, 1, 1)
    ops.element("elasticBeamColumn", 4, 4, 5, 1, 1)
    ops.element("elasticBeamColumn", 5, 5, 6, 1, 1)
    ops.element("elasticBeamColumn", 6, 6, 7, 1, 1)
    ops.element("elasticBeamColumn", 7, 7, 8, 1, 1)
    ops.element("elasticBeamColumn", 8, 8, 9, 1, 1)
    ops.element("elasticBeamColumn", 9, 9, 10, 1, 1)
    ops.element("elasticBeamColumn", 10, 10, 11, 1, 1)

    # Define the solver parameters:
    ops.system("BandSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")

    # Define the loads in the first step:
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(11, 0, -1, 0)

    # Define the solver parameters and solve:
    ops.integrator("LoadControl", 0.01)
    ops.analysis("Static")
    ops.analyze(100)

    # Propagate the loads from the previous step and define the loads in the second step:
    ops.loadConst("-time", 0.0)
    ops.pattern("Plain", 2, 1)
    ops.load(11, -50, 0, 0)

    # Define the solver parameters and solve:
    ops.integrator("LoadControl", 0.01)
    ops.analysis("Static")
    ops.analyze(100)

    # Get the vertical displacement at the free end:
    Δ = -ops.nodeDisp(11, 2)

    return 1 - Δ
end

# Define the reliability problem:
problem = ReliabilityProblem(X, ρ_X, g)

# Perform the reliability analysis using the FORM:
solution = solve(problem, FORM(); backend=AutoFiniteDiff())
println("FORM:")
println("β:   $(solution.β)  ")
println("PoF: $(solution.PoF)")

# Perform the reliability analysis using the SORM:
solution = solve(problem, SORM(); backend=AutoFiniteDiff())
println("SORM:")
println("β:   $(solution.β_2)  ")
println("PoF: $(solution.PoF_2)")
