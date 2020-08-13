#!/usr/bin/python3

# Walk through - https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
#
# A quick run through setting up a simulation with random objects and producing images from some camera array.
# NOTE: it might be good to consider using urdf/obj files as input so that obj modifications don't need 
#       to occur in the code, but rather in the model description. Maybe it's not so useful afterall.

import pybullet as p
import numpy as np
import pybullet_data
import time
import os
import sys
from io import StringIO
import SnakeObj
import RopeObj


urdf_file = "/home/bilkit/Workspace/ModelFreeLearning/sim/little_spheres.urdf"
pr2_urdf_file = "/home/bilkit/Workspace/RobotModels/pr2_common/pr2_description/robots/pr2.urdf"
obj_file = "/home/bilkit/Workspace/pybullet/models/random_urdfs/spheres/sphere.obj"
SIM_DURATION = 10000
SIM_SELECTION = 0

#
# Parse commandline args
#
if len(sys.argv) != 2:
    print("Usage:\n .py <0 or 1>")
    sys.exit()
elif sys.argv[1] == '0':
    SIM_SELECTION = 0
elif sys.argv[1] == '1':
    SIM_SELECTION = 1
else:
    print(f"(!) Unknown arg {sys.argv[1]} \nUsage:\n .py <0 or 1>")
    sys.exit()

#
# Setup physics server
#
print("Connecting to gui")
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
planeId = p.loadURDF('plane.urdf')
p.setGravity(0, 0, -9.81)

pr2 = p.loadURDF(pr2_urdf_file)

#
# Run
#
if SIM_SELECTION == 0:
    RopeObj.simulate(urdf_file, SIM_DURATION)
elif SIM_SELECTION == 1:
    SnakeObj.simulate(SIM_DURATION)
else:
    assert f"(!) Bad simulation selection {SIM_SELECTION}"

##
#
# TODO: use example of camera recoding instead of the commented code below 
#
##


p.disconnect()
print("Disconnecting from gui")
