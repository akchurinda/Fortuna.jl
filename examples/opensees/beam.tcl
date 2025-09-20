# Remove existing models:
wipe

# Define the model parameters:
model BasicBuilder -ndm 2 -ndf 3

# Define the nodes:
node  1   0 0
node  2  18 0
node  3  36 0
node  4  54 0
node  5  72 0
node  6  90 0
node  7 108 0
node  8 126 0
node  9 144 0
node 10 162 0
node 11 180 0

# Define the boundary conditions:
fix 1 1 1 1

# Define the cross-sectional properties:
section Elastic 1 :E 9.120 :I

# Define the transformation:
geomTransf PDelta 1

# Define the elements:
element elasticBeamColumn  1  1  2 1 1
element elasticBeamColumn  2  2  3 1 1
element elasticBeamColumn  3  3  4 1 1
element elasticBeamColumn  4  4  5 1 1
element elasticBeamColumn  5  5  6 1 1
element elasticBeamColumn  6  6  7 1 1
element elasticBeamColumn  7  7  8 1 1
element elasticBeamColumn  8  8  9 1 1
element elasticBeamColumn  9  9 10 1 1
element elasticBeamColumn 10 10 11 1 1

# Define the loads:
timeSeries Linear 1
pattern Plain 1 1 {
    load 11   0 -1 0
    load 11 -50  0 0
}

# Define the solver parameters:
system BandSPD
numberer RCM
constraints Plain
algorithm Linear

# Define the recorder:
recorder Node -file "Output.out" -node 11 -dof 2 disp

# Solve:
integrator LoadControl 0.01
analysis Static
analyze 100