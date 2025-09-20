# https://ascelibrary.org/doi/epdf/10.1061/%28ASCE%290733-9399%281991%29117%3A12%282904%29

# Preamble:
using Fortuna
using PyCall

# Load the OpenSeesPy module:
ops = pyimport("openseespy.opensees")

# Define the random vector:
P_1     = randomvariable("Frechet",   "M", [   60,    60 * 0.1])
P_2     = randomvariable("Frechet",   "M", [   60,    60 * 0.1])
P_3     = randomvariable("Frechet",   "M", [   60,    60 * 0.1])
P_4     = randomvariable("Frechet",   "M", [   60,    60 * 0.1])
P_5     = randomvariable("Frechet",   "M", [   60,    60 * 0.1])
P_6     = randomvariable("Frechet",   "M", [   30,    30 * 0.1])
P_7     = randomvariable("Frechet",   "M", [  100,   100 * 0.1])
P_8     = randomvariable("Frechet",   "M", [  200,   200 * 0.1])
P_9     = randomvariable("Frechet",   "M", [  100,   100 * 0.1])
E_C_L_1 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1]) # Outer columns (left side)
E_C_L_2 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1])
E_C_L_3 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1])
E_C_L_4 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1])
E_C_L_5 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1])
E_C_L_6 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1])
E_C_C_1 = randomvariable("LogNormal", "M", [30000, 30000 * 0.1]) # Central columns 
E_C_C_2 = randomvariable("LogNormal", "M", [30000, 30000 * 0.1])
E_C_C_3 = randomvariable("LogNormal", "M", [30000, 30000 * 0.1])
E_C_C_4 = randomvariable("LogNormal", "M", [30000, 30000 * 0.1])
E_C_C_5 = randomvariable("LogNormal", "M", [30000, 30000 * 0.1])
E_C_C_6 = randomvariable("LogNormal", "M", [30000, 30000 * 0.1])
E_C_R_1 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1]) # Outer columns (right side)
E_C_R_2 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1])
E_C_R_3 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1])
E_C_R_4 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1])
E_C_R_5 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1])
E_C_R_6 = randomvariable("LogNormal", "M", [28000, 28000 * 0.1])
E_B_1   = randomvariable("LogNormal", "M", [26000, 26000 * 0.1]) # Beams
E_B_2   = randomvariable("LogNormal", "M", [26000, 26000 * 0.1])
E_B_3   = randomvariable("LogNormal", "M", [26000, 26000 * 0.1])
E_B_4   = randomvariable("LogNormal", "M", [26000, 26000 * 0.1])
E_B_5   = randomvariable("LogNormal", "M", [26000, 26000 * 0.1])
E_B_6   = randomvariable("LogNormal", "M", [26000, 26000 * 0.1])
E_B_7   = randomvariable("LogNormal", "M", [26000, 26000 * 0.1])
E_B_8   = randomvariable("LogNormal", "M", [26000, 26000 * 0.1])
E_B_9   = randomvariable("LogNormal", "M", [26000, 26000 * 0.1])
E_B_10  = randomvariable("LogNormal", "M", [26000, 26000 * 0.1])
E_B_11  = randomvariable("LogNormal", "M", [26000, 26000 * 0.1])
E_B_12  = randomvariable("LogNormal", "M", [26000, 26000 * 0.1])
A_C_L_1 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1]) # Outer columns (left side)
A_C_L_2 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1])
A_C_L_3 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1])
A_C_L_4 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1])
A_C_L_5 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1])
A_C_L_6 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1])
A_C_C_1 = randomvariable("LogNormal", "M", [ 26.9,  26.9 * 0.1]) # Central columns 
A_C_C_2 = randomvariable("LogNormal", "M", [ 26.9,  26.9 * 0.1])
A_C_C_3 = randomvariable("LogNormal", "M", [ 26.9,  26.9 * 0.1])
A_C_C_4 = randomvariable("LogNormal", "M", [ 26.9,  26.9 * 0.1])
A_C_C_5 = randomvariable("LogNormal", "M", [ 26.9,  26.9 * 0.1])
A_C_C_6 = randomvariable("LogNormal", "M", [ 26.9,  26.9 * 0.1])
A_C_R_1 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1]) # Outer columns (right side)
A_C_R_2 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1])
A_C_R_3 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1])
A_C_R_4 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1])
A_C_R_5 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1])
A_C_R_6 = randomvariable("LogNormal", "M", [ 21.0,  21.0 * 0.1])
A_B_1   = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1]) # Beams
A_B_2   = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1])
A_B_3   = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1])
A_B_4   = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1])
A_B_5   = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1])
A_B_6   = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1])
A_B_7   = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1])
A_B_8   = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1])
A_B_9   = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1])
A_B_10  = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1])
A_B_11  = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1])
A_B_12  = randomvariable("LogNormal", "M", [ 16.0,  16.0 * 0.1])
I_C_L_1 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1]) # Outer columns (left side)
I_C_L_2 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1])
I_C_L_3 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1])
I_C_L_4 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1])
I_C_L_5 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1])
I_C_L_6 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1])
I_C_C_1 = randomvariable("LogNormal", "M", [ 2690,  2690 * 0.1]) # Central columns 
I_C_C_2 = randomvariable("LogNormal", "M", [ 2690,  2690 * 0.1])
I_C_C_3 = randomvariable("LogNormal", "M", [ 2690,  2690 * 0.1])
I_C_C_4 = randomvariable("LogNormal", "M", [ 2690,  2690 * 0.1])
I_C_C_5 = randomvariable("LogNormal", "M", [ 2690,  2690 * 0.1])
I_C_C_6 = randomvariable("LogNormal", "M", [ 2690,  2690 * 0.1])
I_C_R_1 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1]) # Outer columns (right side)
I_C_R_2 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1])
I_C_R_3 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1])
I_C_R_4 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1])
I_C_R_5 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1])
I_C_R_6 = randomvariable("LogNormal", "M", [ 2100,  2100 * 0.1])
I_B_1   = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1]) # Beams
I_B_2   = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1])
I_B_3   = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1])
I_B_4   = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1])
I_B_5   = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1])
I_B_6   = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1])
I_B_7   = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1])
I_B_8   = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1])
I_B_9   = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1])
I_B_10  = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1])
I_B_11  = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1])
I_B_12  = randomvariable("LogNormal", "M", [ 1600,  1600 * 0.1])
X = [
    P_1, P_2, P_3, P_4, P_5, P_6, P_7, P_8, P_9,
    E_C_L_1, E_C_L_2, E_C_L_3, E_C_L_4, E_C_L_5, E_C_L_6,
    E_C_C_1, E_C_C_2, E_C_C_3, E_C_C_4, E_C_C_5, E_C_C_6,
    E_C_R_1, E_C_R_2, E_C_R_3, E_C_R_4, E_C_R_5, E_C_R_6,
    E_B_1, E_B_2, E_B_3, E_B_4, E_B_5, E_B_6, E_B_7, E_B_8, E_B_9, E_B_10, E_B_11, E_B_12,
    A_C_L_1, A_C_L_2, A_C_L_3, A_C_L_4, A_C_L_5, A_C_L_6,
    A_C_C_1, A_C_C_2, A_C_C_3, A_C_C_4, A_C_C_5, A_C_C_6,
    A_C_R_1, A_C_R_2, A_C_R_3, A_C_R_4, A_C_R_5, A_C_R_6,
    A_B_1, A_B_2, A_B_3, A_B_4, A_B_5, A_B_6, A_B_7, A_B_8, A_B_9, A_B_10, A_B_11, A_B_12,
    I_C_L_1, I_C_L_2, I_C_L_3, I_C_L_4, I_C_L_5, I_C_L_6,
    I_C_C_1, I_C_C_2, I_C_C_3, I_C_C_4, I_C_C_5, I_C_C_6,
    I_C_R_1, I_C_R_2, I_C_R_3, I_C_R_4, I_C_R_5, I_C_R_6,
    I_B_1, I_B_2, I_B_3, I_B_4, I_B_5, I_B_6, I_B_7, I_B_8, I_B_9, I_B_10, I_B_11, I_B_12]

# Define the correlation matrix:
ρ_X = Matrix(1.0 * I, length(X), length(X))

# Define the limit state function:
function g(x::Vector)
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # [Node tag | X | Y]
    ops.node( 1,   0,   0)
    ops.node( 2,   0, 120)
    ops.node( 3,   0, 240)
    ops.node( 4,   0, 360)
    ops.node( 5,   0, 480)
    ops.node( 6,   0, 600)
    ops.node( 7,   0, 720)
    ops.node( 8, 300,   0)
    ops.node( 9, 300, 120)
    ops.node(10, 300, 240)
    ops.node(11, 300, 360)
    ops.node(12, 300, 480)
    ops.node(13, 300, 600)
    ops.node(14, 300, 720)
    ops.node(15, 600,   0)
    ops.node(16, 600, 120)
    ops.node(17, 600, 240)
    ops.node(18, 600, 360)
    ops.node(19, 600, 480)
    ops.node(20, 600, 600)
    ops.node(21, 600, 720)

    # [Node tag | UX | UY | URZ]
    ops.fix( 1, 1, 1, 1)
    ops.fix( 8, 1, 1, 1)
    ops.fix(15, 1, 1, 1)

    # [Section type | Section tag | E | A | I]
    ops.section("Elastic",  1, x[10], x[40], x[70]) # Outer columns (left side)
    ops.section("Elastic",  2, x[11], x[41], x[71])
    ops.section("Elastic",  3, x[12], x[42], x[72])
    ops.section("Elastic",  4, x[13], x[43], x[73])
    ops.section("Elastic",  5, x[14], x[44], x[74])
    ops.section("Elastic",  6, x[15], x[45], x[75])
    ops.section("Elastic",  7, x[16], x[46], x[76]) # Outer columns (right side)
    ops.section("Elastic",  8, x[17], x[47], x[77])
    ops.section("Elastic",  9, x[18], x[48], x[78])
    ops.section("Elastic", 10, x[19], x[49], x[79])
    ops.section("Elastic", 11, x[20], x[50], x[80])
    ops.section("Elastic", 12, x[21], x[51], x[81])
    ops.section("Elastic", 13, x[22], x[52], x[82]) # Central columns
    ops.section("Elastic", 14, x[23], x[53], x[83])
    ops.section("Elastic", 15, x[24], x[54], x[84])
    ops.section("Elastic", 16, x[25], x[55], x[85])
    ops.section("Elastic", 17, x[26], x[56], x[86])
    ops.section("Elastic", 18, x[27], x[57], x[87])
    ops.section("Elastic", 19, x[28], x[58], x[88]) # Beams
    ops.section("Elastic", 20, x[29], x[59], x[89])
    ops.section("Elastic", 21, x[30], x[60], x[90])
    ops.section("Elastic", 22, x[31], x[61], x[91])
    ops.section("Elastic", 23, x[32], x[62], x[92])
    ops.section("Elastic", 24, x[33], x[63], x[93])
    ops.section("Elastic", 25, x[34], x[64], x[94])
    ops.section("Elastic", 26, x[35], x[65], x[95])
    ops.section("Elastic", 27, x[36], x[66], x[96])
    ops.section("Elastic", 28, x[37], x[67], x[97])
    ops.section("Elastic", 29, x[38], x[68], x[98])
    ops.section("Elastic", 30, x[39], x[69], x[99])

    # [Transformation type | Transformation tag]
    ops.geomTransf("PDelta", 1)
    
    # [Element type | Element tag | Node (i) | Node (j) | Section tag | Transformation tag]
    ops.element("elasticBeamColumn",  1,  1,  2,  1, 1) # Outer columns (left side)
    ops.element("elasticBeamColumn",  2,  2,  3,  2, 1)
    ops.element("elasticBeamColumn",  3,  3,  4,  3, 1)
    ops.element("elasticBeamColumn",  4,  4,  5,  4, 1)
    ops.element("elasticBeamColumn",  5,  5,  6,  5, 1)
    ops.element("elasticBeamColumn",  6,  6,  7,  6, 1)
    ops.element("elasticBeamColumn",  7,  8,  9,  7, 1) # Outer columns (right side)
    ops.element("elasticBeamColumn",  8,  9, 10,  8, 1)
    ops.element("elasticBeamColumn",  9, 10, 11,  9, 1)
    ops.element("elasticBeamColumn", 10, 11, 12, 10, 1)
    ops.element("elasticBeamColumn", 11, 12, 13, 11, 1)
    ops.element("elasticBeamColumn", 12, 13, 14, 12, 1)
    ops.element("elasticBeamColumn", 13, 15, 16, 13, 1) # Central columns
    ops.element("elasticBeamColumn", 14, 16, 17, 14, 1)
    ops.element("elasticBeamColumn", 15, 17, 18, 15, 1)
    ops.element("elasticBeamColumn", 16, 18, 19, 16, 1)
    ops.element("elasticBeamColumn", 17, 19, 20, 17, 1)
    ops.element("elasticBeamColumn", 18, 20, 21, 18, 1)
    ops.element("elasticBeamColumn", 19,  2,  9, 19, 1) # Beams
    ops.element("elasticBeamColumn", 20,  3, 10, 20, 1)
    ops.element("elasticBeamColumn", 21,  4, 11, 21, 1)
    ops.element("elasticBeamColumn", 22,  5, 12, 22, 1)
    ops.element("elasticBeamColumn", 23,  6, 13, 23, 1)
    ops.element("elasticBeamColumn", 24,  7, 14, 24, 1)
    ops.element("elasticBeamColumn", 25,  9, 16, 25, 1)
    ops.element("elasticBeamColumn", 26, 10, 17, 26, 1)
    ops.element("elasticBeamColumn", 27, 11, 18, 27, 1)
    ops.element("elasticBeamColumn", 28, 12, 19, 28, 1)
    ops.element("elasticBeamColumn", 29, 13, 20, 29, 1)
    ops.element("elasticBeamColumn", 30, 14, 21, 30, 1)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(11, x[1], 0, 0)

    ops.system("BandSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 0.01)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(100)

    ops.loadConst("-time", 0.0)
    ops.pattern("Plain", 2, 1)
    ops.load( 7, 0, -x[7], 0)
    ops.load(14, 0, -x[8], 0)
    ops.load(21, 0, -x[9], 0)

    ops.integrator("LoadControl", 0.01)
    ops.analyze(100)

    return 5 - ops.nodeDisp(21, 1)
end

# Define the reliability problem:
problem = ReliabilityProblem(X, ρ_X, g)

# Solve the reliability problem:
solution = solve(problem, FORM(), backend = AutoFiniteDiff())
println("β   = $(solution.β)  ")
println("PoF = $(solution.PoF)")