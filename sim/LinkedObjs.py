#!/usr/bin/python3

# Walk through - https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
#
# A quick run through setting up a simulation with random objects and producing images from some camera array.
# NOTE: it might be good to consider using urdf/obj files as input so that obj modifications don't need 
#       to occur in the code, but rather in the model description. Maybe it's not so useful afterall.




# Some notes about setting up models
# setup urdf example - http://wiki.ros.org/pr2_description
# rosrun xacro xacro --inorder `rospack find pr2_description`/robots/pr2.urdf.xacro > pr2.urdf
# 
# install rospack and xacro
# $ sudo apt install rospack-tools rosbash
# $ sudo apt-get install -y ros-<distro>-xacro
#
# clone pr2_common src
# add source path to ros package path env var
#   e.g., export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:"/home/bilkit/Workspace/PR2"
# 
# $ sudo apt install ros-lunar-pr2-common

import pybullet as p
import time
import os, glob, random, sys
import numpy as np
from matplotlib import pyplot as plt
from math import cos, sin
import pybullet_data
import numpy as np
import cv2 as cv
import math

pb_random_urdfs_dir = "/home/bilkit/Workspace/pybullet/models/random_urdfs"
pb_textures_dir = "/home/bilkit/Workspace/pybullet/textures/dtd/images/stratified"
IMG_WIDTH = 300
IMG_HEIGHT = 300
MODEL_INITIAL_POS = [0,0,1]

class ModelFormats: 
    Obj = "obj"
    Urdf = "urdf"

class RopeObj:
    def __init__(self, size):
        self.m_sphereUid = -1
        self.m_massSpringModel = None 
        return 

    class PointMass:
        def __init__(self, massId):
            self.m_id = massId
            self.m_mass = 0 
            self.m_inertia = 0 
            self.m_position = [0,0,0]

        def SetPhysicalProperties(mass, inertia):
            self.m_mass = mass
            self.m_inertia = inertia

        # todo: accessors
        def SetPosition(pos):
            self.m_position = pos

        def Str():
            return "id: {}\nmass: {}\nintertia: {}\npos: {}\nlink_id: {}\n".format(
                self.m_id, self.m_mass, self.m_inertial, self.m_position, self.m_linkId)

    class Spring:
        # NOTE: each spring is associated with a point mass
        # 
        def __init__(self, parentMassId):
            self.m_massId = parentMassId
            self.m_kSpring = 100000 
            self.m_kBend = 100000
            self.m_length = 0 
            self.m_orientation = [0,0,0] 

        def SetPhysicalProperties(springConstant, bendConstant, restPosition=0):
            self.m_kSpring = springConstant
            self.m_kBend = bendConstant
            self.m_length = restPosition

        def SetOrientation(orientationVec):
            self.m_orientation = orientationVec

        def VectorAngle(vector):
            a = vector
            b = self.m_orientation
            # Should we worry about the sign? in which case maybe sine+cross is better?
            return math.acos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))

        def Str():
            return "id: {}\nmid: {}\nmass: {}\nspringK: {}\nbendK: {}\nlength: {}\norientation: {}\n".format(
                self.m_id, self.m_massId, self.m_kSpring, self.m_kBend, self.m_length, self.m_orientation)

#        def ComputeBeta(otherLinkOrientation):

    def Move(self, heading):
        # Use Wang_Bleuler physics model to compute new position for each point mass


        # get mouse position to use as grasp point?
        # find the link id that was selected (is closes to it?)
        # set corresponding node position to new link id position
        # update all other nodes and links
        # update all joints corresponding to nodes
        return

    def CreateRopeModel(self, numJoints):

        self.m_massSpringModel = np.empty(numJoints)
        for i in np.arange(numJoints):
            pm = PointMass(i)
            lk = Spring(i)
            pm.SetPhysicalProperties(2, 0.0495)  # Should read from file
            self.m_massSpringModel[i] = (pm, lk)  

        return

    # What if there's no collision model?
    def SetObjectParameters(self, modelGeometryFile, modelCollisionFile="", texturePath=""):
        print("Loading urdf")
        if (not os.path.exists(modelGeometryFile) and not os.path.exists(modelCollisionFile)):
            print("Failed to find model paths")
            return 

        multiBodyId = ""
        visId = ""
        colId = ""
        if (modelGeometryFile.endswith(ModelFormats.Obj)):
            # Compute size of mesh to determine collision shape size and link distance
            # Specify visual model
            meshScale = [0.1, 0.1, 0.1]
            visId = p.createVisualShape(
                shapeType = p.GEOM_MESH,
                fileName = modelGeometryFile,
                rgbaColor = None,
                meshScale = meshScale)

            # Specify collision model
            if modelCollisionFile == "" and False:
                colId = p.createCollisionShape(
                    shapeType = p.GEOM_SPHERE,
                    radius = 0.2,
                    meshScale = meshScale)
            else:
                colId = p.createCollisionShape(
                    shapeType = p.GEOM_MESH,
                    fileName = modelCollisionFile,
                    meshScale = meshScale)

            # Create joint model
            multiBodyId = p.createMultiBody(
                baseMass = 1.0,
                baseCollisionShapeIndex = colId,
                baseVisualShapeIndex = visId,
                basePosition = MODEL_INITIAL_POS,
                baseOrientation = p.getQuaternionFromEuler([0,0,0]))
            print("(INFO) Loaded new model '{}'".format(multiBodyId))
        elif (modelGeometryFile.endswith(ModelFormats.Urdf)): 
            multiBodyId = p.loadURDF(modelGeometryFile)
            print("(INFO) Loaded new model '{}'".format(multiBodyId))
        else:
            print("Unknown model format '{}'".format(modelGeometryFile))

        # Override textures
        if (texturePath != ""):
            p.changeVisualShape(multiBodyId, -1, textureUniqueId = p.loadTexture(texturePath))

        # anistropicFriction = [1, 0.01, 0.01]
        # p.changeDynamics(multiBodyId, -1, lateralFriction=2, anisotropicFriction=anistropicFriction)
        # for i in range(p.getNumJoints(multiBodyId)):
        #     p.getJointInfo(multiBodyId, i)
        #     p.changeDynamics(multiBodyId, i, lateralFriction=2, anisotropicFriction=anistropicFriction)

        print("generated model:\n{}\n{}\n{}".format(str(visId), str(colId), str(multiBodyId)))

        CreateRopeModel(p.getNumJoints(sphereUid))

        # update member components
        self.m_sphereUid = multiBodyId
        return

class SnakeObj:

    m_dt = 1. / 240.
    m_sphereUid = -1
    m_sphereRadius = 0.25
    m_waveLength = 4
    m_wavePeriod = 1.5
    m_waveAmplitude = 0.1
    m_waveFront = 0.0

    def __init__(self, sphereRadius, useMaximalCoordinates=True):
        # colBoxId = p.createCollisionShapeArray([p.GEOM_BOX, p.GEOM_SPHERE],radii=[sphereRadius+0.03,sphereRadius+0.03], halfExtents=[[sphereRadius,sphereRadius,sphereRadius],[sphereRadius,sphereRadius,sphereRadius]])
        colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[sphereRadius, sphereRadius, sphereRadius])

        mass = 1
        visualShapeId = -1

        link_Masses = []
        linkCollisionShapeIndices = []
        linkVisualShapeIndices = []
        linkPositions = []
        linkOrientations = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        indices = []
        jointTypes = []
        axis = []

        for i in range(36):
            link_Masses.append(1)
            linkCollisionShapeIndices.append(colBoxId)
            linkVisualShapeIndices.append(-1)
            linkPositions.append([0, sphereRadius * 2.0 + 0.01, 0])
            linkOrientations.append([0, 0, 0, 1])
            linkInertialFramePositions.append([0, 0, 0])
            linkInertialFrameOrientations.append([0, 0, 0, 1])
            indices.append(i)
            jointTypes.append(p.JOINT_REVOLUTE)
            axis.append([0, 0, 1])

        basePosition = [0, 0, 1]
        baseOrientation = [0, 0, 0, 1]

        self.m_sphereRadius = sphereRadius
        self.m_sphereUid = p.createMultiBody(mass,
                                      colBoxId,
                                      visualShapeId,
                                      basePosition,
                                      baseOrientation,
                                      linkMasses=link_Masses,
                                      linkCollisionShapeIndices=linkCollisionShapeIndices,
                                      linkVisualShapeIndices=linkVisualShapeIndices,
                                      linkPositions=linkPositions,
                                      linkOrientations=linkOrientations,
                                      linkInertialFramePositions=linkInertialFramePositions,
                                      linkInertialFrameOrientations=linkInertialFrameOrientations,
                                      linkParentIndices=indices,
                                      linkJointTypes=jointTypes,
                                      linkJointAxis=axis,
                                      useMaximalCoordinates=useMaximalCoordinates)

        anistropicFriction = [1, 0.01, 0.01]
        p.changeDynamics(self.m_sphereUid, -1, lateralFriction=2, anisotropicFriction=anistropicFriction)
        for i in np.arange(p.getNumJoints(self.m_sphereUid)):
            p.getJointInfo(self.m_sphereUid, i)
            p.changeDynamics(self.m_sphereUid, i, lateralFriction=2, anisotropicFriction=anistropicFriction)

    def SetObjectParameters(self, dt, length, period, amplitude, wavefront):
        self.m_dt = dt
        self.m_waveLength = length
        self.m_wavePeriod = period
        self.m_waveAmplitude = amplitude
        self.m_waveFront = wavefront

    def Move(self, heading):
        # our steering value
        self.m_segmentLength = self.m_sphereRadius * 2.0

        sphereUid = self.m_sphereUid
        amp = 0.2
        offset = 0.6
        numJoints = p.getNumJoints(sphereUid)
        scaleStart = 1.0

        # start of the snake with smaller waves.
        # I think starting the wave at the tail would work better ( while it still goes from head to tail )
        if (self.m_waveFront < self.m_segmentLength * 4.0):
            scaleStart = self.m_waveFront / (self.m_segmentLength * 4.0)

        # we simply move a sin wave down the body of the snake.
        # this snake may be going backwards, but who can tell ;)
        for joint in np.arange(numJoints):
            segment = joint  # numMuscles-1-joint
            # map segment to phase
            phase = (self.m_waveFront - (segment + 1) * self.m_segmentLength) / self.m_waveLength
            phase -= math.floor(phase)
            phase *= math.pi * 2.0

            # map phase to curvature
            targetPos = math.sin(phase) * scaleStart * self.m_waveAmplitude

            # // steer snake by squashing +ve or -ve side of sin curve
            if (heading > 0 and targetPos < 0):
                targetPos *= 1.0 / (1.0 + heading)

            if (heading < 0 and targetPos > 0):
                targetPos *= 1.0 / (1.0 - heading)

            # set our motor
            p.setJointMotorControl2(sphereUid,
                                    joint,
                                    p.POSITION_CONTROL,
                                    targetPosition=targetPos, #+ heading,
                                    force=30)

        # wave keeps track of where the wave is in time
        self.m_waveFront += self.m_dt / self.m_wavePeriod * self.m_waveLength

def PlaceCamera(position, target):
    print("new cam pos: {}".format(position))
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=position,
        cameraTargetPosition=target,
        cameraUpVector=[0, 1, 0])

    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=3.1)

    return viewMatrix, projectionMatrix

def GetImageFromCamera(viewMatrix, projMatrix):
    width, height, rgbaImg, depthImg, segImg = p.getCameraImage(
        width=IMG_WIDTH, 
        height=IMG_HEIGHT,
        viewMatrix=viewMatrix,
        projectionMatrix=projMatrix)
    return np.array(rgbaImg).reshape(height, width, 4)

def MoveCamera(deltaPos, deltaTarg):

    return viewMatrix


SIM_DURATION = 10000
print("Connecting to gui")
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
planeId = p.loadURDF('plane.urdf')
p.setGravity(0,0,-9.81)

# if (len(sys.argv) == 1 or not os.path.exists(sys.argv[1])):
#    print("ERROR: Must existing urdf file for a model")
#    sys.exit(1)

# # Setup cameras
# A = 30    # angular resolution in degrees 
# R = 2     # radial distance in meters
# N = 135 // A
# cams = [None] * N 
# print("Creating {} cams".format(N))
# for i in np.arange(N):
#   x = R * cos(i * A * np.pi / 180) 
#   y = R * sin(i * A * np.pi / 180) 
#   z = 0

#   view, proj = PlaceCamera([x, y, z], MODEL_INITIAL_POS)
#   cams[i] = [view, proj]

# # Record images
# cid = 0
# for c in cams:
#   rgbaImg = GetImageFromCamera(c[0], c[1])
#   rgbImg = rgbaImg[:,:,:3]
#   cv.imwrite("{}_rgb.png".format(cid), rgbImg)
#   cid += 1

# Better to use loop for batches of objs
# NOTE: will get "XIO:  fatal IO error 11" when closing window before timeout
shouldExit = False
sleepDuration = 1. / 240. 
if (len(sys.argv) == 1):
    print("Please provide model urdf file")
    sys.exit()

#targetObj = RopeObj()
#targetObj.SetObjectParameters(sys.argv[1])
targetObj = SnakeObj(0.2)
targetObj.SetObjectParameters(sleepDuration, 2, 2, 1, 0)

toggleFreq = SIM_DURATION / 500 
heading = 0.4 
for i in np.arange(SIM_DURATION):
    p.stepSimulation()
    # periodically flip heading sign
    heading *= -2 * (i % toggleFreq == 0) + 1
    targetObj.Move(heading)
    time.sleep(sleepDuration)

p.disconnect()
print("Disconnecting from gui")
