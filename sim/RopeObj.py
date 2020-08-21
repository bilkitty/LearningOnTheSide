import pybullet as p
import numpy as np
import pybullet_data
import math
import time
import os
import sys
from io import StringIO
import RopeModel


class ModelFormats:
    Obj = "obj"
    Urdf = "urdf"


class RopeObj:
    # Note that large masses result in "explosive" interactions with other objs - in other words,
    # rope will falcon punch. However, changing masses will change the dynamics req to maintain
    # similar rope-like behaviour.

    class DynamicsProfile:
        def __init__(self, obj_uid, n_links, mass, restitution, linear_damping, angular_damping):
            pass

        def generate_change_dynamics_options(self):
            pass



    DEFAULT_MASS = 0.5
    MASS_0P01_SPHERE = 1
    INVALID_UID = -1

    def __init__(self, geometryFile, fileType="obj", sphereRadius=0.3, xyz=[0, 0, 0], overrideDynamics=True):
        #
        # Geometric model
        #

        if fileType.lower() == "obj":
            self.m_sphere_uid, link_positions = create_rope_geometry(geometryFile, sphereRadius)
        else:
            self.m_sphere_uid, link_positions = load_rope_geometry_from_urdf(geometryFile, xyz)

        assert (link_positions.shape[0] > 0), "Failed to create rope geometry"
        self.m_length = link_positions.shape[0]
        self.m_mass = RopeObj.MASS_0P01_SPHERE

        # Override physical properties
        # Using large values is highly unstable
        if overrideDynamics:
            for link in np.arange(self.m_length):
                p.changeDynamics(self.m_sphere_uid,
                                 link,
                                 self.m_mass,
                                 lateralFriction=1.0,
                                 rollingFriction=1.0,
                                 restitution=0.9,  # (bounciness)
                                 )

        #
        # Physical model
        #
        self.m_model = RopeModel.RopeModel(self.m_length, self.m_mass, link_positions)

    """
    NOTE:
    change dynamics options
        lateralFriction=2,                         // linear contact friction
        spinningFriction=2,                        // torsional friction about normal
        rollingFriction=2,                         // torsional friction about tangent
        restitution=0.001,  # (bounciness)         // bounciness (^ less elastic)
        linearDamping=0.1,                         // (0.04)                       
        angularDamping=0.01,                       // (0.04) mvmt btw links (^ appears stiffer)
        contactStiffness=1,                        // high value -> surface intersections
        contactDamping=1,                          // should set together w/ stiffness
        anisotropicFriction=[0.01, 0.01, 0.01]     // scales friction along different dirs
    """


    """
    TODO: 
        perhaps we need our mass spring model to compute new coordinates per link.
        then we compute joint position updates (inverse kin solver) and set them.
        
    """
    def move(self, dt, verbose):
        if verbose:
            outfile = sys.stdout
        else:
            outfile = StringIO()

        # Set new joint positions
        n_joints = self.m_length - 2
        for i in np.arange(n_joints):
            joints_pos = self.m_model.ComputeNewJointPosition(self.m_sphere_uid, i, i + 1, dt, outfile)
            p.setJointMotorControl2(self.m_sphere_uid,
                                    i,
                                    p.POSITION_CONTROL,
                                    targetPosition=joints_pos[i],
                                    force=30)

        # Update physical model
        ls, _, _ = link_states(self.m_sphere_uid, n_joints, verbose)
        self.m_model.update(ls, dt, verbose)


"""
Helper functions
"""

ENABLE_MODEL = True
SLEEP_DURATION = 1. / 4800.
SPHERE_RADIUS = 0.2


def make_rope(file, xyz=[0, 0, 0]):
    return RopeObj(file, "obj", SPHERE_RADIUS, xyz=xyz) if "obj" in file else RopeObj(file, "urdf", xyz=xyz)


def simulate(file, duration=10000, verbose=False):
    rope = make_rope(file)
    for i in np.arange(duration):
        p.stepSimulation()
        if ENABLE_MODEL:
            rope.move(SLEEP_DURATION, verbose)

        time.sleep(SLEEP_DURATION)


"""
    NOTE: some api functions that involve inspecting the multibody don't work with
          programatically generated bodies. So, prioritize loading models from urdf,
          or otherwise. Besides, it's probably better to store model descriptions in
          files rather than creating them on the fly...
          example failing apis:
          getlink_states, calcultateInverseKinematics
"""


# TODO: refactor
def link_states(uid, num_links, verbose=False):
    failedQueries = 0
    pos, orientations, inertialPos, inertialOrientations = [], [], [], []
    for i in np.arange(num_links):
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

    return np.column_stack((pos, orientations)), inertialPos, inertialOrientations


# TODO: refactor
def load_rope_geometry_from_urdf(modelUrdfFile, xyz=[0, 0, 0], texturePath=""):
    print("Loading urdf '{}'".format(modelUrdfFile))
    if (not os.path.exists(modelUrdfFile)):
        print("Failed to find model paths '{}'".format(modelUrdfFile))
        return RopeObj.INVALID_UID, np.array([])

    multiBodyId = ""
    if (modelUrdfFile.endswith(ModelFormats.Urdf)):
        multiBodyId = p.loadURDF(modelUrdfFile, *xyz)
        print("(INFO) Loaded new model '{}'".format(multiBodyId))
    else:
        print("Unknown model format '{}'".format(modelUrdfFile))
        return RopeObj.INVALID_UID, np.array([])

    # Override textures
    if (texturePath != ""):
        p.changeVisualShape(multiBodyId, -1, textureUniqueId=p.loadTexture(texturePath))

    link_poses, _, _ = link_states(multiBodyId, p.getNumJoints(multiBodyId))
    linkPositions = link_poses[:, 0:3]
    if len(linkPositions) > 0:
        print("(INFO) Found {} links and {} joints".format(len(linkPositions), p.getNumJoints(multiBodyId)))
    else:
        print("Failed to get link states.")
        return RopeObj.INVALID_UID, np.array([])

    return multiBodyId, np.array(linkPositions)


"""
Don't look below! 
"""


def create_rope_geometry(modelGeometryFile, sphereRadius):
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
    RopeObj.GetLinkState(massMultiBodyIds[0], 0)
    RopeObj.GetLinkState(springMultiBodyIds[0], 0)
    return massMultiBodyIds[0], np.array(lps, dtype=np.float128)


