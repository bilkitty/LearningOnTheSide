#!/usr/bin/python3

# Walk through - https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
#
# A quick run through setting up a simulation with random objects and producing images from some camera array.
# NOTE: it might be good to consider using urdf/obj files as input so that obj modifications don't need 
#       to occur in the code, but rather in the model description. Maybe it's not so useful afterall.

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
import random

pb_random_urdfs_dir = "/home/bilkit/Workspace/pybullet/models/random_urdfs"
pb_textures_dir = "/home/bilkit/Workspace/pybullet/textures/dtd/images/stratified"
IMG_WIDTH = 300
IMG_HEIGHT = 300
MODEL_INITIAL_POS = [0,0,1]

ENABLE_MODEL = False 

class ModelFormats: 
    Obj = "obj"
    Urdf = "urdf"

class PointMass:
    # TODO: figure out how to represent inertia
    #       this'll become relevant when updating the 
    #       torsional/twisting forces of the springs
    #       However, for now our model can account for
    #       bending and stretching of the rope.
    def __init__(self, massId):
        self.m_id = massId
        self.m_mass = 0 
        self.m_inertia = np.identity(3) 
        self.m_position = [0,0,0]
        self.m_velocity = [0,0,0]

    def SetPhysicalProperties(self, mass, inertia):
        self.m_mass = mass
        self.m_inertia = inertia

    def SetPositionAndVelocity(self, pos, vel):
        self.m_position = np.array(pos)
        self.m_velocity = np.array(vel)

    def Str(self):
        return "id: {}\nmass: {}\nintertia: {}\npos: {}\n".format(
            self.m_id, self.m_mass, self.m_inertia, self.m_position)

class Spring:
    # NOTE: each spring is associated with a point mass
    #       still unsure exactly what data structure makes the most sense...
    #       so, I'm maintaining ids for corresponding point masses and springs
    #       in case they are not stored in as pairs in an array.
    def __init__(self, parentMassId):
        self.m_massId = parentMassId
        self.m_kSpring = 100000 
        self.m_kBend = 100000
        self.m_restLength = 0.05
        self.m_orientation = np.array([np.pi,0,0]) #[0,0,0, 1] 

    def SetPhysicalProperties(self, springConstant, bendConstant, nodeDistance=0.05):
        self.m_kSpring = springConstant
        self.m_kBend = bendConstant
        self.m_restLength = nodeDistance

    def SetOrientation(self, masses):
        a = masses[self.m_massId]
        b = masses[self.m_massId + 1]
        self.m_orientation = b.m_position - a.m_position

    def Length(self, pointMasses):
        return abs(pointMasses[self.m_massId].m_position - pointMasses[self.m_massId + 1].m_position)

    def Str(self):
        return "mid: {}\nspringK: {}\nbendK: {}\nlength: {}\norientation: {}\n".format(
            self.m_massId, self.m_kSpring, self.m_kBend, self.m_restLength, self.m_orientation)

    # TODO: use proper quaternion ops
    def ComputeAngle(self, orientation):
        a = np.array(orientation)
        b = self.m_orientation
        # Should we worry about the sign? in which case maybe sine+cross is better?
        return math.acos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
        #return 0.0

class RopeObj:
    DEFAULT_MASS = 0.1
    DEFAULT_INERTIA = np.identity(3)
    DEFAULT_INERTIA[2,2] = 0
    DEFAULT_KSPRING = 10 
    DEFAULT_KBEND = 10
    USE_MAXIMAL_COORDINATES = True
    MASS_IDX = 0
    SPR_IDX = 1
    INVALID_UID = -1

    def __init__(self, geometryFile, sphereRadius=0.2):
        #
        # Geometric model
        #

        #self.m_sphereUid, linkPositions = RopeObj.CreateGeometry(geometryFile, sphereRadius)
        #self.m_sphereUid, linkPositions = RopeObj.CreateGeometry2(geometryFile, sphereRadius)
        self.m_sphereUid, linkPositions = RopeObj.LoadGeometryFromUrdf(geometryFile)
        assert(linkPositions.shape[0] > 0), "Failed to create rope geometry"

        # Override physical properties
        # Using large values is highly unstable
        for l in np.arange(linkPositions.shape[0]):
            p.changeDynamics(self.m_sphereUid,
                             l,
                             RopeObj.DEFAULT_MASS,
                             lateralFriction=0.2,
                             spinningFriction=0.2, 
                             rollingFriction=0.2,
                             restitution=100,   # (bouncyness)
                             linearDamping=0.1,
                             angularDamping=0.01,
                             contactStiffness=1,
                             contactDamping=1, 
                             anisotropicFriction=[1, 0.01, 0.01]) 

        #
        # Physical model
        #
        ropeLength = linkPositions.shape[0]
        self.m_massSpringModel = np.empty((2, ropeLength), dtype=object) 
        for i in np.arange(ropeLength):
            pm = PointMass(i)
            pm.SetPhysicalProperties(self.DEFAULT_MASS, self.DEFAULT_INERTIA)  # Should read from file
            pm.SetPositionAndVelocity(linkPositions[i], 0)
            self.m_massSpringModel[self.MASS_IDX][i] = pm

        # There should be N - 1 springs for an N length rope
        for i in np.arange(ropeLength - 1):
            lk = Spring(i)
            #absDistPts = 0 if i == 0 else math.abs(linkPositions[i] - linkPositions[i - 1])
            lk.SetPhysicalProperties(self.DEFAULT_KSPRING, self.DEFAULT_KBEND)
            lk.SetOrientation(self.m_massSpringModel[self.MASS_IDX])
            self.m_massSpringModel[self.SPR_IDX][i] = lk
        self.m_massSpringModel[self.SPR_IDX][ropeLength - 1] = None # will this come back to bite me? could just duplicate the last spring... hmmm 

        #length = self.Size()
        length = p.getNumJoints(self.m_sphereUid)
        print("geometry:\n{}\n{}\nphysical:\n{}\n{}".format(
            self.m_sphereUid, length, self.m_massSpringModel[self.MASS_IDX][0].Str(), self.m_massSpringModel[self.SPR_IDX][0].Str()))

        return 

    # NOTE: Using Wang,Bleuler 2005 physics model to compute new position for each point mass
    def Move(self, movedLinkId, newPosition, dt):
        nJoints = p.getNumJoints(self.m_sphereUid)
        positions = np.array([np.zeros(4)] * nJoints)

        # Here's a stupid algorithm weeeee!
        #
        # set moved node  to new link position
        # for left nodes
        #     Compute stretching force
        #     Compute bending force
        #     
        #     Compute linear acceleration      
        #     Compute linear velocity
        #     Compute new node position

        leftLinkId = movedLinkId - 1
        i = max(1, leftLinkId)
        j = i - 1

        # We are about to cascade the effects of moved link changing position
        # It should be fine to use spring lengths as is - i.e., as diff between (new) parent mass pos and other mass pos 
        # Later, all springs will be updated after all new mass locations are determined.
        springs = self.m_massSpringModel[self.SPR_IDX]
        masses = self.m_massSpringModel[self.MASS_IDX]
        x = masses[i].m_position
        xd = masses[i].m_velocity
        #for i in np.arange(0, movedLinkId):
        fsi = springs[i].m_kSpring * (springs[i].Length(masses) - springs[i].m_restLength) * springs[i].m_orientation
        fsj = springs[j].m_kSpring * (springs[j].Length(masses) - springs[j].m_restLength) * springs[j].m_orientation
        fs = fsi - fsj

        print("force: {1} - {2} = {0}".format(fs, fsi, fsj))
        assert(len(fs) == 3), "Improper force type {} of length".format(type(fs), len(fs))
        xdd = fs / masses[i].m_mass
        xd = xd + xdd * dt
        x = x + xd * dt
        print("numJoints: {}\tid: {}\tpos: {}".format(p.getNumJoints(self.m_sphereUid), i, x))

        positions = p.calculateInverseKinematics(self.m_sphereUid, i, x)
        print("new joint pos:\n{}\n{}".format(x, positions[i]))
        for n in np.arange(nJoints):
            p.setJointMotorControl2(self.m_sphereUid,
                                    n,
                                    p.POSITION_CONTROL,
                                    targetPosition=positions[i],
                                    force=30)

        # update model - masses first then springs
        masses[i].SetPositionAndVelocity(x, xd)
        springs[i].SetOrientation(masses)


        # rightLinkId = movedLinkId + 1
        # i = min(rightLinkId, self.Size() - 2)
        # # for right nodes
        # #     do the same

        # for i in np.arange(nJoints):
        #     p.setJointMotorControl2(self.m_sphereUid,
        #                             i,
        #                             p.POSITION_CONTROL,
        #                             targetPosition=positions[i],
        #                             force=30)
        return

    def Size(self):
        return self.m_massSpringModel.shape[1]

    def ObjUid(self):
        return self.m_sphereUid

    @staticmethod
    def CreateGeometry(modelGeometryFile, sphereRadius):
        visualShapeId = -1
        if (os.path.exists(modelGeometryFile) and modelGeometryFile.endswith(ModelFormats.Obj)):
            # Compute size of mesh to determine collision shape size and link distance
            # Specify visual model
            meshScale = [0.1, 0.1, 0.1]
            visualShapeId = p.createVisualShape(
                shapeType = p.GEOM_MESH,
                fileName = modelGeometryFile,
                rgbaColor = None,
                meshScale = meshScale)
        else:
            print("Geometry file '{}' should be existing .obj".format(modelGeometryFile))
            # continue with default visualization


        # colBoxId = p.createCollisionShapeArray([p.GEOM_BOX, p.GEOM_SPHERE],radii=[sphereRadius+0.03,sphereRadius+0.03], halfExtents=[[sphereRadius,sphereRadius,sphereRadius],[sphereRadius,sphereRadius,sphereRadius]])
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sphereRadius, sphereRadius, sphereRadius])

        linkMasses = []
        linkCollisionShapeIndices = []
        linkVisualShapeIndices = []
        linkPositions = []
        linkOrientations = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        indices = []
        jointTypes = []
        axis = []

        ropeLength = 36
        for i in range(ropeLength):
            linkMasses.append(1)
            linkCollisionShapeIndices.append(colBoxId)
            linkVisualShapeIndices.append(visualShapeId)
            linkPositions.append([0, sphereRadius * 2.0 + 0.01, 0])
            #linkOrientations.append([0, 0, 0, 1])
            linkOrientations.append([0, 0, np.pi])
            linkInertialFramePositions.append([0, 0, 0])
            linkInertialFrameOrientations.append([0, 0, 0, 1])
            indices.append(i)
            jointTypes.append(p.JOINT_REVOLUTE)
            axis.append([0, 0, 1])

        basePosition = [0, 0, 1]
        baseOrientation = [0, 0, 0, 1]

        # Geometric model
        multiBodyId = p.createMultiBody(RopeObj.DEFAULT_MASS,
                                      colBoxId,
                                      visualShapeId,
                                      basePosition,
                                      baseOrientation,
                                      linkMasses=np.array(linkMasses, dtype=float),
                                      linkCollisionShapeIndices=np.array(linkCollisionShapeIndices, dtype=int),
                                      linkVisualShapeIndices=np.array(linkVisualShapeIndices, dtype=int),
                                      linkPositions=linkPositions,
                                      linkOrientations=linkOrientations,
                                      linkInertialFramePositions=linkInertialFramePositions,
                                      linkInertialFrameOrientations=linkInertialFrameOrientations,
                                      linkParentIndices=np.array(indices, dtype=int),
                                      linkJointTypes=jointTypes,
                                      linkJointAxis=axis,
                                      useMaximalCoordinates=RopeObj.USE_MAXIMAL_COORDINATES)

        anistropicFriction = [1, 0.01, 0.01]
        p.changeDynamics(multiBodyId, -1, lateralFriction=2, anisotropicFriction=anistropicFriction)
        for i in np.arange(p.getNumJoints(multiBodyId)):
            p.getJointInfo(multiBodyId, i)
            p.changeDynamics(multiBodyId, i, lateralFriction=2, anisotropicFriction=anistropicFriction)

        return multiBodyId, np.array(linkPositions)

    @staticmethod
    def CreateGeometry2(modelGeometryFile, sphereRadius):
        massVisualShapeId = -1
        springVisualShapeId = -1
        if (os.path.exists(modelGeometryFile) and modelGeometryFile.endswith(ModelFormats.Obj)):
            # Compute size of mesh to determine collision shape size and link distance
            # Specify visual model
            meshScale = [0.1, 0.1, 0.1]
            massVisualShapeId = p.createVisualShape(
                shapeType = p.GEOM_MESH,
                fileName = modelGeometryFile,
                rgbaColor = None,
                meshScale = meshScale)
            meshScale = [0.03, 0.03, 0.03]
            springVisualShapeId = p.createVisualShape(
                shapeType = p.GEOM_MESH,
                fileName = modelGeometryFile,
                rgbaColor = None,
                meshScale = meshScale)
        else:
            print("Geometry file '{}' should be existing .obj".format(modelGeometryFile))
            # continue with default visualization

        colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[sphereRadius/3, sphereRadius/3, sphereRadius/3])

        lps = []
        linkMasses = [1]
        linkCollisionShapeIndices = [colBoxId]
        linkVisualShapeIndices = [-1]
        linkPositions = [[0, 0, 0.11]]
        linkOrientations = [[0, 0, 0, 1]]
        linkInertialFramePositions = [[0, 0, 0]]
        linkInertialFrameOrientations = [[0, 0, 0, 1]]
        indices = [0]
        jointTypes = [p.JOINT_REVOLUTE]
        axis = [[0, 0, 1]]

        massMultiBodyIds = []
        springMultiBodyIds = []
        for i in range(3):
          for j in range(3):
            for k in range(3):
              basePosition = [
                  1 + i * 5 * sphereRadius, 1 + j * 5 * sphereRadius, 1 + k * 5 * sphereRadius + 1
              ]
              baseOrientation = [0, 0, 0, 1]
              if (k & 2):
                # Masses
                massMultiBodyIds.append(p.createMultiBody(RopeObj.DEFAULT_MASS, colSphereId, massVisualShapeId, basePosition,
                                              baseOrientation))
              else:
                # Springs
                springMultiBodyIds.append(p.createMultiBody(RopeObj.DEFAULT_MASS,
                                              colBoxId,
                                              springVisualShapeId,
                                              basePosition,
                                              baseOrientation,
                                              linkMasses=linkMasses,
                                              linkCollisionShapeIndices=linkCollisionShapeIndices,
                                              linkVisualShapeIndices=linkVisualShapeIndices,
                                              linkPositions=linkPositions,
                                              linkOrientations=linkOrientations,
                                              linkInertialFramePositions=linkInertialFramePositions,
                                              linkInertialFrameOrientations=linkInertialFrameOrientations,
                                              linkParentIndices=indices,
                                              linkJointTypes=jointTypes,
                                              linkJointAxis=axis))
              lps.append(basePosition)

              # p.changeDynamics(multiBodyId,
              #                  -1,
              #                  spinningFriction=0.001,
              #                  rollingFriction=0.001,
              #                  linearDamping=0.0)

        print("test: {}".format(massMultiBodyIds[0]))
        print("      {}".format(springMultiBodyIds[0]))
        RopeObj.LinkState(massMultiBodyIds[0])
        RopeObj.LinkState(springMultiBodyIds[0])
        return massMultiBodyIds, springMultiBodyIds, np.array(lps)

    # NOTE: some api functions that involve inspecting the multibody don't work with
    #       programatically generated bodies. So, prioritize loading models from urdf,
    #       or otherwise. Besides, it's probably better to store model descriptions in
    #       files rather than creating them on the fly...
    #       example failing apis: 
    #       getLinkStates, calcultateInverseKinematics
    @staticmethod
    def LoadGeometryFromUrdf(modelUrdfFile, texturePath=""):
        print("Loading urdf '{}'".format(modelUrdfFile))
        if (not os.path.exists(modelUrdfFile)):
            print("Failed to find model paths '{}'".format(modelUrdfFile))
            return RopeObj.INVALID_UID, np.array([])

        multiBodyId = ""
        if (modelUrdfFile.endswith(ModelFormats.Urdf)): 
            multiBodyId = p.loadURDF(modelUrdfFile)
            print("(INFO) Loaded new model '{}'".format(multiBodyId))
        else:
            print("Unknown model format '{}'".format(modelUrdfFile))
            return RopeObj.INVALID_UID, np.array([])

        # Override textures
        if (texturePath != ""):
            p.changeVisualShape(multiBodyId, -1, textureUniqueId = p.loadTexture(texturePath))

        linkPositions = RopeObj.LinkStates(multiBodyId, p.getNumJoints(multiBodyId))[0]
        if (not linkPositions.any()):
            print("Failed to get link states.")
            return RopeObj.INVALID_UID, np.array([])

        return multiBodyId, np.array(linkPositions)

    @staticmethod
    def JointStates(uid):
        # Just for fun, query joint states
        # WARN: segfault occurs when there's an if statement enclosing prints...
        failedQueries = 0
        pos, vels, forces, torques = [], [], [], []
        for i in np.arange(p.getNumJoints(uid)):
            jointState = p.getJointState(uid, i)
            jointInfo = p.getJointInfo(uid, i)
            print("================ ")
            if (jointState == None or jointInfo == None):
                failedQueries += 1
            else:
                jointPos, jointVel, jointForces, jointTorque = p.getJointState(uid, i)
                index, name, jtype, qindex, uindex = p.getJointInfo(uid, i)[:5]
                pos.append(jointPos)
                vels.append(jointVel)
                forces.append(jointForces)
                torques.append(jointTorque)

                print("query joint info:\n{}\n{}\n{}\n{}\n{}".format(
                    index,
                    name,
                    jtype,
                    qindex,
                    uindex))
                print("query joint states:\n{}\n{}\n{}\n{}".format(
                    jointPos,
                    jointVel,
                    jointForces,
                    jointTorque))

        if failedQueries > 0:
            print("================ ")
            print("Failed joint state queries: {}".format(failedQueries))

        return np.array(pos), np.array(vels), np.array(forces), np.array(torques)

    @staticmethod
    def LinkStates(uid, numLinks):
        failedQueries = 0
        pos, orientations, inertialPos, inertialOrientations = [], [], [], []
        for i in np.arange(numLinks):
            linkState = p.getLinkState(uid, i)
            print("================ ")
            if (linkState == None):
                failedQueries += 1
            else:
                posWorld, orientationWorld, inertialPosLocal, inertialOrientationLocal = linkState[:4]  
                pos.append(posWorld)
                orientations.append(orientationWorld)
                inertialPos.append(inertialPosLocal)
                inertialOrientations.append(inertialOrientationLocal)

                print("query link info:\n{}\n{}\n{}\n{}\n{}".format(
                    i,
                    posWorld, 
                    orientationWorld, 
                    inertialPosLocal, 
                    inertialOrientationLocal))
        if failedQueries > 0:
            print("================ ")
            print("Failed link state queries: {}".format(failedQueries))

        return np.array(pos), np.array(orientations), np.array(inertialPos), np.array(inertialOrientations)

    @staticmethod
    def LinkState(uid):
        linkState = p.getLinkState(uid, 0)
        print("================ ")
        if (linkState == None):
            print("Failed link state query")
            return None, None, None, None
        else:
            posWorld, orientationWorld, inertialPosLocal, inertialOrientationLocal = linkState[:4]

            print("query link info:\n{}\n{}\n{}\n{}\n{}".format(
                uid,
                posWorld, 
                orientationWorld, 
                inertialPosLocal, 
                inertialOrientationLocal))
            return posWorld, orientationWorld, inertialPosLocal, inertialOrientationLocal

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

        linkMasses = []
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
            linkMasses.append(1)
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

        self.m_waveLength = 0 
        self.m_wavePeriod = 0 
        self.m_waveAmplitude = 0 
        self.m_waveFront = 0 
        self.m_sphereRadius = sphereRadius
        self.m_sphereUid = p.createMultiBody(mass,
                                      colBoxId,
                                      visualShapeId,
                                      basePosition,
                                      baseOrientation,
                                      linkMasses=linkMasses,
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

    def SetObjectParameters(self, length, period, amplitude, wavefront):
        self.m_waveLength = length
        self.m_wavePeriod = period
        self.m_waveAmplitude = amplitude
        self.m_waveFront = wavefront

    def Move(self, heading, dt):
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
        self.m_waveFront += dt / self.m_wavePeriod * self.m_waveLength

SIM_DURATION = 10000
print("Connecting to gui")
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
planeId = p.loadURDF('plane.urdf')
p.setGravity(0,0,-9.81)


##
#
# TODO: use example of camera recoding instead of the commented code below 
#
##

# Better to use loop for batches of objs
# NOTE: will get "XIO:  fatal IO error 11" when closing window before timeout
shouldExit = False
sleepDuration = 1. / 240. 
sphereRadius = 0.2
if (len(sys.argv) == 1):
    print("Please provide model obj/urdf file")
    sys.exit()

urdffile = "/home/bilkit/Workspace/pybullet/models/random_urdfs/spheres/spheres.urdf"
objfile = "/home/bilkit/Workspace/pybullet/models/random_urdfs/spheres/sphere.obj"
targetObj = RopeObj(objfile, sphereRadius) if sys.argv[1] == '0' else RopeObj(urdffile)
#targetObj = SnakeObj(sphereRadius)
#targetObj.SetObjectParameters(sleepDuration, 2, 2, 1, 0)

toggleFreq = SIM_DURATION / 500 
heading = 0.4 
for i in np.arange(SIM_DURATION):
    p.stepSimulation()

    if ENABLE_MODEL:
        # periodically flip heading sign
        heading *= -2 * (i % toggleFreq == 0) + 1

        # if mouse pointer intersects body
        # find link closest to mouse pos
        mousePos = [0.5, 0, 1]
        refMousePointId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=0, basePosition=mousePos)
        pts = p.getClosestPoints(bodyA=targetObj.ObjUid(), bodyB=refMousePointId, distance=1)

        # binary search for closest link

        selectedLink = 5
        targetObj.Move(selectedLink, mousePos, sleepDuration)

    time.sleep(sleepDuration)

p.disconnect()
print("Disconnecting from gui")
