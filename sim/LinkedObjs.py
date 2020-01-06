#!/usr/bin/python3

# Walk through - https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
#
# A quick run through setting up a simulation with random objects and producing images from some camera array.
# NOTE: it might be good to consider using urdf/obj files as input so that obj modifications don't need 
#       to occur in the code, but rather in the model description. Maybe it's not so useful afterall.


# TODO: remove mouse pos and reference id. hmmmm, how are the joints actually moving?
#       it seems like the positions of the links are just given by the sim. There's no
#       need to really compute them. So, we should just observe these positions and compute
#       the new joint positions, then update the link positions with those from the
#       resulting state. I guess this is simpler?

import pybullet as p
import time
import os, sys
from io import StringIO
import numpy as np
import pybullet_data
import math

ENABLE_MODEL = True


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
        self.m_inertia = np.identity(3, dtype=np.float128)
        self.m_position = np.zeros(3, dtype=np.float128)
        self.m_velocity = np.zeros(3, dtype=np.float128)

    def SetPhysicalProperties(self, mass, inertia):
        self.m_mass = mass
        self.m_inertia = inertia

    def SetPositionAndVelocity(self, pos, vel):
        self.m_position = np.array(pos, dtype=np.float128)
        self.m_velocity = np.array(vel, dtype=np.float128)

    def Str(self):
        return "id: {}\nmass: {}\ninertia: {}\npos: {}\n".format(
            self.m_id, self.m_mass, self.m_inertia, self.m_position)


class Spring:
    # NOTE: each spring is associated with a point mass
    #       still unsure exactly what data structure makes the most sense...
    #       so, I'm maintaining ids for corresponding point masses and springs
    #       in case they are not stored in as pairs in an array.
    def __init__(self, parentMassId):
        self.m_massId = parentMassId
        self.m_kSpring = 10
        self.m_kBend = 10
        self.m_restLength = 0.05
        self.m_orientation = np.array([np.pi, 0, 0], dtype=np.float128)  # [0,0,0, 1]

    def SetPhysicalProperties(self, springConstant, bendConstant, nodeDistance=0.05):
        self.m_kSpring = springConstant
        self.m_kBend = bendConstant
        self.m_restLength = nodeDistance

    def SetOrientation(self, masses):
        a = masses[self.m_massId - 1]
        b = masses[self.m_massId]
        self.m_orientation = b.m_position - a.m_position

    def Length(self, pointMasses):
        return np.linalg.norm(pointMasses[self.m_massId].m_position - pointMasses[self.m_massId - 1].m_position)

    def Str(self):
        return "mid: {}\nspringK: {}\nbendK: {}\nlength: {}\norientation: {}\n".format(
            self.m_massId, self.m_kSpring, self.m_kBend, self.m_restLength, self.m_orientation)

    # TODO: use proper quaternion ops
    def ComputeAngle(self, orientation):
        a = np.array(orientation, dtype=np.float128)
        b = self.m_orientation
        # Should we worry about the sign? in which case maybe sine+cross is better?
        return math.acos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
        # return 0.0


class RopeObj:
    DEFAULT_MASS = 1
    DEFAULT_INERTIA = np.identity(3)
    DEFAULT_INERTIA[2, 2] = 0
    DEFAULT_KSPRING = 1
    DEFAULT_KBEND = 1
    USE_MAXIMAL_COORDINATES = True
    MASS_IDX = 0
    SPR_IDX = 1
    INVALID_UID = -1

    def __init__(self, geometryFile, sphereRadius=0.2):
        #
        # Geometric model
        #

        # self.m_sphereUid, linkPositions = RopeObj.CreateGeometry(geometryFile, sphereRadius)
        # self.m_sphereUid, linkPositions = RopeObj.CreateGeometry2(geometryFile, sphereRadius)
        self.m_sphereUid, linkPositions = RopeObj.LoadGeometryFromUrdf(geometryFile)
        assert (linkPositions.shape[0] > 0), "Failed to create rope geometry"

        # Override physical properties
        # Using large values is highly unstable
        for l in np.arange(linkPositions.shape[0]):
            p.changeDynamics(self.m_sphereUid,
                             l,
                             RopeObj.DEFAULT_MASS,
                             lateralFriction=2,
                             spinningFriction=2,
                             rollingFriction=2,
                             restitution=0.001,  # (bounciness)
                             linearDamping=1,
                             angularDamping=1,
                             contactStiffness=1,
                             contactDamping=1,
                             anisotropicFriction=[1, 0.01, 0.01])

        #
        # Physical model
        #
        ropeLength = linkPositions.shape[0]
        self.m_massSpringModel = np.empty((2, ropeLength), dtype=object)
        masses = self.m_massSpringModel[self.MASS_IDX]
        springs = self.m_massSpringModel[self.SPR_IDX]
        for i in np.arange(ropeLength):
            pm = PointMass(i)
            pm.SetPhysicalProperties(self.DEFAULT_MASS, self.DEFAULT_INERTIA)  # Should read from file
            pm.SetPositionAndVelocity(linkPositions[i], 0)
            masses[i] = pm

        # There should be N - 1 springs for an N length rope
        # SIGNIFICANT ASSUMPTION HERE, the initial rope configuration is used to set the resting position
        #                              of the spring. That is, the rope should initially be unwound and relaxed.
        springs[-1] = None  # spring index refers to spring on the left side of the corresponding mass index
        for i in np.arange(ropeLength - 1):
            lk = Spring(parentMassId=i + 1)
            # absDistPts = 0 if i == 0 else math.abs(linkPositions[i] - linkPositions[i - 1])
            lk.SetPhysicalProperties(self.DEFAULT_KSPRING, self.DEFAULT_KBEND, lk.Length(masses))
            lk.SetOrientation(masses)
            springs[i] = lk

        length = p.getNumJoints(self.m_sphereUid)
        print("geometry:\n{}\n{}\nphysical:\n{}\n{}".format(
            self.m_sphereUid, length, masses[0].Str(),
            springs[0].Str()))

        # Debug with ghost geometry model
        orientations = [s.m_orientation for s in springs[:len(springs) - 1]]
        self.m_ghostMassBody, self.m_ghostSpringBody = RopeObj.CreateGeometry3(0.2, linkPositions, orientations)

        return

    """
    Parameters:
        i       joint index
        refDir  direction of reference joint; +1 for right, -1 for left
    """
    # HUGE BUG: mass positions, and thus forces, explode causing numerical overflow... spring consts too large?
    def ComputeNewJointPosition(self, i, j, dt, outfile):
        springs = self.m_massSpringModel[self.SPR_IDX]
        masses = self.m_massSpringModel[self.MASS_IDX]

        x = masses[i].m_position
        xd = masses[i].m_velocity
        di = springs[i].Length(masses)
        dj = springs[j].Length(masses)
        fsi = springs[i].m_kSpring * (di - springs[i].m_restLength) * springs[i].m_orientation
        fsj = springs[j].m_kSpring * (dj - springs[j].m_restLength) * springs[j].m_orientation
        fs = fsj - fsi

        if (i < j):
            fs *= -1.

        outfile.write("joint: {}\n".format(i))
        outfile.write("\tf| {1} - {2} = {0}\n".format(fsi, fsj, fs))
        assert (len(fs) == 3), "Improper force type {} of length".format(type(fs), len(fs))
        xdd = fs / masses[i].m_mass
        xdp = xd + xdd * dt
        xp = x + xdp * dt

        # update model - masses first then springs
        masses[i].SetPositionAndVelocity(xp, xdp)
        springs[i].SetOrientation(masses)

        jointPositions = p.calculateInverseKinematics(self.m_sphereUid, i, xp)
        outfile.write("\tx| {}\n\tv| {} | {}\n\tjp| {}\n".format(xp, xdp, xp - x, jointPositions[i]))
        return jointPositions[i]

    def LazyUpdate(self, i, j):
        self.m_massSpringModel[self.MASS_IDX][i].m_position = self.m_massSpringModel[self.MASS_IDX][j].m_position
        self.m_massSpringModel[self.MASS_IDX][i].m_velocity = self.m_massSpringModel[self.MASS_IDX][j].m_velocity

    # NOTE: Using Wang,Bleuler 2005 physics model to compute new position for each point mass
    # Here's a stupid algorithm weeeee!
    # for left nodes
    #     Compute all spring forces
    #     Compute linear acceleration and velocity, and new node position
    def Move(self, movedLinkId, newPosition, dt, verbose):

        if verbose:
            outfile = sys.stdout
        else:
            outfile = StringIO()

        nJoints = p.getNumJoints(self.m_sphereUid)
        # jointPositions = np.array([np.zeros(4)] * nJoints)
        jointPositions = np.zeros(nJoints, dtype=np.float128)
        newPosition = np.array(newPosition, dtype=np.float128)

        # Update moved link in physical model
        position = self.m_massSpringModel[self.MASS_IDX][movedLinkId].m_position
        self.m_massSpringModel[self.MASS_IDX][movedLinkId].m_position = newPosition
        self.m_massSpringModel[self.MASS_IDX][movedLinkId].m_velocity = newPosition - position
        ikResult = p.calculateInverseKinematics(self.m_sphereUid, movedLinkId, newPosition)
        jointPositions[movedLinkId] = ikResult[movedLinkId]

        # Update other links in physical model:
        # We are about to cascade the effects of moved link changing position
        # It should be fine to use spring lengths as is - i.e., as diff between (new) parent mass pos and other mass pos 
        # Later, all springs will be updated after all new mass locations are determined.
        for i in range(movedLinkId - 1, 1, -1):
            jointPositions[i] = self.ComputeNewJointPosition(i, i + 1, dt, outfile=outfile)

        for i in range(movedLinkId + 1, self.Size() - 1):
            jointPositions[i] = self.ComputeNewJointPosition(i, i - 1, dt, outfile=outfile)

        # TODO: figure out how to handle the last node as it doesn't have a right spring as reference
        #       generally this scheme should be able to handle the edge nodes similarly
        #       for now, do a lazy copy from endpoints
        self.LazyUpdate(0, 1)
        jointPositions[0] = jointPositions[1]
        self.LazyUpdate(self.Size() - 1, self.Size() - 2)
        jointPositions[self.Size() - 1] = jointPositions[self.Size() - 2]

        # Update joint positions in geometric model
        # [unsure] and feed link positions back to physical model
        springs = self.m_massSpringModel[self.SPR_IDX]
        masses = self.m_massSpringModel[self.MASS_IDX]
        positions = []
        for i in np.arange(nJoints):
            p.setJointMotorControl2(self.m_sphereUid,
                                    i,
                                    p.POSITION_CONTROL,
                                    targetPosition=jointPositions[i],
                                    force=30)

            positions.append(masses[i].m_position)
            x = RopeObj.LinkState(self.m_sphereUid, i, False)[0]
            print("joint: {}\nx0={}\nx1={}\n".format(i, x, masses[i].m_position))
            masses[i].SetPositionAndVelocity(x, masses[i].m_velocity)

        orientations = []
        for i in np.arange(nJoints - 1):
            orientations.append(springs[i].m_orientation)
            springs[i].SetOrientation(masses)

        # Debug view of mass spring model
        RopeObj.DeleteGeometry3(self.m_ghostMassBody, self.m_ghostSpringBody)
        self.m_ghostMassBody, self.m_ghostSpringBody = RopeObj.CreateGeometry3(0.2, positions, orientations)

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
                shapeType=p.GEOM_MESH,
                fileName=modelGeometryFile,
                rgbaColor=None,
                meshScale=meshScale)
        else:
            print("Geometry file '{}' should be existing .obj".format(modelGeometryFile))
            # continue with default visualization

        # colBoxId = p.createCollisionShapeArray([p.GEOM_BOX, p.GEOM_SPHERE],radii=[sphereRadius+0.03,sphereRadius+0.03]
        # , halfExtents=[[sphereRadius,sphereRadius,sphereRadius],[sphereRadius,sphereRadius,sphereRadius]])
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
            # linkOrientations.append([0, 0, 0, 1])
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
                                        linkMasses=np.array(linkMasses, dtype=np.float128),
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

        @staticmethod
        def CreateGeometry2(modelGeometryFile, sphereRadius):
            massVisualShapeId = -1
            springVisualShapeId = -1
            if (os.path.exists(modelGeometryFile) and modelGeometryFile.endswith(ModelFormats.Obj)):
                # Compute size of mesh to determine collision shape size and link distance
                # Specify visual model
                meshScale = [0.1, 0.1, 0.1]
                massVisualShapeId = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=modelGeometryFile,
                    rgbaColor=None,
                    meshScale=meshScale)
                meshScale = [0.03, 0.03, 0.03]
                springVisualShapeId = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=modelGeometryFile,
                    rgbaColor=None,
                    meshScale=meshScale)
            else:
                print("Geometry file '{}' should be existing .obj".format(modelGeometryFile))
                # continue with default visualization

            colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
            colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                              halfExtents=[sphereRadius / 3, sphereRadius / 3, sphereRadius / 3])

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
                            massMultiBodyIds.append(
                                p.createMultiBody(RopeObj.DEFAULT_MASS, colSphereId, massVisualShapeId, basePosition,
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
            return massMultiBodyIds, springMultiBodyIds, np.array(lps, dtype=np.float128)

        return multiBodyId, np.array(linkPositions, dtype=float)

    @staticmethod
    def CreateGeometry3(sphereRadius, positions, orientations):
        massVisualShapeId = -1
        springVisualShapeId = -1

        colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
        colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[sphereRadius / 3, sphereRadius / 3, sphereRadius / 3])

        massMultiBodyIds = []
        springMultiBodyIds = []
        for i, basePosition in enumerate(positions):
            if (i % (len(positions) / 4) == 0):
                baseOrientation = [0, 0, 0, 1]
                # Masses
                massMultiBodyIds.append(
                    p.createMultiBody(0, colSphereId, massVisualShapeId, basePosition,
                                      baseOrientation))

        linkMasses = [0]
        linkCollisionShapeIndices = [colBoxId]
        linkVisualShapeIndices = [massVisualShapeId]
        linkPositions = [[0, 0, 0.11]]
        linkOrientations = [[0, 0, 0, 1]]
        linkInertialFramePositions = [[0, 0, 0]]
        linkInertialFrameOrientations = [[0, 0, 0, 1]]
        indices = [0]
        jointTypes = [p.JOINT_REVOLUTE]
        axis = [[0, 0, 1]]
        for i in range(len(orientations)):
            if (i % (len(orientations) / 4) == 0):
                basePosition = positions[i] # shift along correct axis by sphereRadius
                baseOrientation = orientations[i]
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
                # p.changeDynamics(multiBodyId,
                #                  -1,
                #                  spinningFriction=0.001,
                #                  rollingFriction=0.001,
                #                  linearDamping=0.0)

        print("test: {}".format(massMultiBodyIds[0]))
        print("      {}".format(springMultiBodyIds[0]))
        return massMultiBodyIds, springMultiBodyIds

    @staticmethod
    def DeleteGeometry3(massMultiBodyIds, springMultiBodyIds):
        for m in massMultiBodyIds:
            p.removeBody(m)
        for s in springMultiBodyIds:
            p.removeBody(s)
        return

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
            p.changeVisualShape(multiBodyId, -1, textureUniqueId=p.loadTexture(texturePath))

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
    def LinkStates(uid, numLinks, verbose=True):
        failedQueries = 0
        pos, orientations, inertialPos, inertialOrientations = [], [], [], []
        for i in np.arange(numLinks):
            linkState = p.getLinkState(uid, i)
            if (linkState == None):
                failedQueries += 1
            else:
                posWorld, orientationWorld, inertialPosLocal, inertialOrientationLocal = linkState[:4]
                pos.append(posWorld)
                orientations.append(orientationWorld)
                inertialPos.append(inertialPosLocal)
                inertialOrientations.append(inertialOrientationLocal)

                if verbose:
                    print("================ ")
                    print("query link info:\n{}\n{}\n{}\n{}\n{}".format(
                        i,
                        posWorld,
                        orientationWorld,
                        inertialPosLocal,
                        inertialOrientationLocal))
        if verbose and failedQueries > 0:
            print("================ ")
            print("Failed link state queries: {}".format(failedQueries))

        return np.array(pos), np.array(orientations), np.array(inertialPos), np.array(inertialOrientations)

    @staticmethod
    def LinkState(uid, linkId, verbose=True):
        linkState = p.getLinkState(uid, linkId)
        if (verbose and linkState == None):
            print("Failed link state query")
            return None, None, None, None
        else:
            posWorld, orientationWorld, inertialPosLocal, inertialOrientationLocal = linkState[:4]

            if verbose:
                print("================ ")
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
                                    targetPosition=targetPos,  # + heading,
                                    force=30)

        # wave keeps track of where the wave is in time
        self.m_waveFront += dt / self.m_wavePeriod * self.m_waveLength


SIM_DURATION = 10000
print("Connecting to gui")
physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
planeId = p.loadURDF('plane.urdf')
p.setGravity(0, 0, -9.81)

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

urdffile = "/home/bilkit/Workspace/pybullet/scripts/sim/spheres_partial.urdf"
objfile = "/home/bilkit/Workspace/pybullet/models/random_urdfs/spheres/sphere.obj"
targetObj = RopeObj(objfile, sphereRadius) if sys.argv[1] == '0' else RopeObj(urdffile)
# targetObj = SnakeObj(sphereRadius)
# targetObj.SetObjectParameters(sleepDuration, 2, 2, 1, 0)

toggleFreq = SIM_DURATION / 100
heading = 0.4
pts = []
for i in np.arange(SIM_DURATION):
    p.stepSimulation()
    verbose = False

    if ENABLE_MODEL:
        # periodically flip heading sign
        heading *= -2 * (i % toggleFreq == 0) + 1

        # if mouse pointer intersects body
        # find link closest to mouse pos
        mousePos = [0, -0.8, 0]
        refMousePointId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, basePosition=mousePos)
        #pts = np.array(p.getClosestPoints(bodyA=targetObj.ObjUid(), bodyB=refMousePointId, distance=1))
        pts = np.zeros(3)

        # binary search for closest link

        # print some stuff every so often
        if (i % (toggleFreq) == 0):
            #verbose = True
            print("================================================")
            print("Mouse\npos: {}\nclosepts: {}, {}\n".format(mousePos, pts.shape[0], pts.mean()))

        selectedLink = 0
        targetObj.Move(selectedLink, mousePos, sleepDuration, verbose)

    time.sleep(sleepDuration)

p.disconnect()
print("Disconnecting from gui")
