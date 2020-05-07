
import pybullet as p
import numpy as np
import pybullet_data
import math
import time
import os
import sys
from io import StringIO


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

    def move(self, heading, dt):
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
                                    targetPosition=targetPos + heading,
                                    force=30)

        # wave keeps track of where the wave is in time
        self.m_waveFront += dt / self.m_wavePeriod * self.m_waveLength


"""
Helper functions
"""

ENABLE_MODEL = True
SLEEP_DURATION = 1. / 480.
SPHERE_RADIUS = 0.2
HEADING_MAGNITUDE = 0.3


def make_snake(sphere_radius=SPHERE_RADIUS):
    snake = SnakeObj(sphere_radius)
    snake.SetObjectParameters(2, 2, 1, 0)
    return snake


def simulate(duration):
    fake_heading = HEADING_MAGNITUDE
    toggle_freq = duration / 100
    snake = make_snake()
    for i in np.arange(duration):
        p.stepSimulation()
        fake_heading *= -2 * (i % toggle_freq == 0) + 1     # TODO: bit twiddle instead
        snake.move(fake_heading, SLEEP_DURATION)
        time.sleep(SLEEP_DURATION)
