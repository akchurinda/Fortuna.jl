from abaqus          import *
from abaqusConstants import *
from odbAccess       import *

TrackedNode = 11

Jobname = 'CantileverBeamTemp'
ODBFile = openOdb(path = Jobname + '.odb')

with open('Output.txt.', 'w') as FileID:
    Delta = ODBFile.steps['LOADS'].frames[-1].fieldOutputs['U'].values[11 - 1].data[1]
    FileID.write(str(Delta))