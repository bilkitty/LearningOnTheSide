import pybullet as p
import numpy as np
import pybullet_data
import math
import time
import os
import sys
from io import StringIO


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
        assert(len(pos) == 3 and len(vel) == 3), f"expected 3 and 3, but got {len(pos)} and {len(vel)}"
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


class RopeModel:
    DEFAULT_MASS = 1
    DEFAULT_INERTIA = np.identity(3)
    DEFAULT_INERTIA[2, 2] = 0
    DEFAULT_KSPRING = 1
    DEFAULT_KBEND = 1
    MASS_IDX = 0
    SPR_IDX = 1

    def __init__(self, n_masses, total_mass, link_positions):
        assert(0 < n_masses), "rope model requires at least one point mass"
        self.m_mass_spring_model = np.empty((2, n_masses), dtype=object)
        masses = self.m_mass_spring_model[self.MASS_IDX]
        springs = self.m_mass_spring_model[self.SPR_IDX]

        # Initialize masses
        mass = total_mass / n_masses
        for i in np.arange(n_masses):
            pm = PointMass(i)
            pm.SetPhysicalProperties(mass, self.DEFAULT_INERTIA)  # Should read from file
            pm.SetPositionAndVelocity(link_positions[i], np.zeros(3))
            masses[i] = pm

        # Initialize springs
        # There should be N - 1 springs for an N length rope
        # SIGNIFICANT ASSUMPTION HERE, the initial rope configuration is used to set the resting position
        #                              of the spring. That is, the rope should initially be unwound and relaxed.
        springs[-1] = None  # spring index refers to spring on the left side of the corresponding mass index
        for i in np.arange(n_masses - 1):
            lk = Spring(parentMassId=i + 1)
            # absDistPts = 0 if i == 0 else math.abs(linkPositions[i] - linkPositions[i - 1])
            lk.SetPhysicalProperties(self.DEFAULT_KSPRING, self.DEFAULT_KBEND, lk.Length(masses))
            lk.SetOrientation(masses)
            springs[i] = lk


    # TODO: refactor
    # HUGE BUG: mass positions, and thus forces, explode causing numerical overflow... spring consts too large?
    def ComputeNewJointPosition(self, root_uid, left_link_id, right_link_id, dt, outfile):
        springs = self.m_mass_spring_model[self.SPR_IDX]
        masses = self.m_mass_spring_model[self.MASS_IDX]

        i = left_link_id
        j = right_link_id
        x = masses[i].m_position
        xd = masses[i].m_velocity
        di = springs[i].Length(masses)
        dj = springs[j].Length(masses)
        fsi = springs[i].m_kSpring * (di - springs[i].m_restLength) * springs[i].m_orientation
        fsj = springs[j].m_kSpring * (dj - springs[j].m_restLength) * springs[j].m_orientation
        fs = fsj - fsi

        # Legacy: relevant for implementation that
        # traversed both l<- or ->r instead of r->l
        if (i < j):
            fs *= -1.

        outfile.write("joint: {}\n".format(i))
        outfile.write("\tf| ({1}) - ({2}) = {0}\n".format(fsi, fsj, fs))
        assert (len(fs) == 3), "Improper force type {} of length".format(type(fs), len(fs))
        xdd = fs / masses[i].m_mass
        xdp = xd + xdd * dt
        xp = x + xdp * dt

        # update model - masses first then springs
        masses[i].SetPositionAndVelocity(xp, xdp)
        springs[i].SetOrientation(masses)

        joint_positions = p.calculateInverseKinematics(root_uid, i, xp)
        outfile.write("\tx| {}\n\tv| {} | {}\n\tjp| {}\n".format(xp, xdp, xp - x, joint_positions))
        return joint_positions

    def update_ends(self, i, j):
        self.m_mass_spring_model[self.MASS_IDX][i].m_position = self.m_mass_spring_model[self.MASS_IDX][j].m_position
        self.m_mass_spring_model[self.MASS_IDX][i].m_velocity = self.m_mass_spring_model[self.MASS_IDX][j].m_velocity

    def update(self, obj_link_poses, dt, verbose):
        springs = self.m_mass_spring_model[self.SPR_IDX]
        masses = self.m_mass_spring_model[self.MASS_IDX]
        # Update masses based on object links
        for i in range(len(obj_link_poses)):
            position = obj_link_poses[i][0:3]
            if verbose:
                print("joint: {}\nxo={}\nxm={}\n".format(i, position, masses[i].m_position))

            velocity = (position - masses[i].m_position) / (dt * np.ones(3))
            masses[i].SetPositionAndVelocity(position, velocity)
        # TODO: metrics; log error between obj link and mass positions (and curr vs previous velocities)

        # TODO: figure out how to handle the last node as it doesn't have a right spring as reference
        #       generally this scheme should be able to handle the edge nodes similarly
        #       for now, do a lazy copy from endpoints
        self.update_ends(0, 1)
        self.update_ends(self.Size() - 1, self.Size() - 2)

        # Spring update depends on adjacent masses
        for i in range(len(obj_link_poses)):
            springs[i].SetOrientation(masses)



    # NOTE: Using Wang,Bleuler 2005 physics model to compute new position for each point mass
    # Here's a stupid algorithm weeeee!
    # for left nodes
    #     Compute all spring forces
    #     Compute linear acceleration and velocity, and new node position
    def Move(self, moved_link_id, newPosition, dt, verbose):

        if verbose:
            outfile = sys.stdout
        else:
            outfile = StringIO()

        m_length = p.getNumJoints(self.m_sphere_uid)
        joint_positions = np.array([np.zeros(1)] * m_length, dtype=np.float128)
        newPosition = np.array(newPosition, dtype=np.float128)

        # Update moved link in physical model
        # Note that calculateInverseKinematics returns a tuple of values for each degree of freedom
        position = self.m_mass_spring_model[self.MASS_IDX][moved_link_id].m_position
        self.m_mass_spring_model[self.MASS_IDX][moved_link_id].m_position = newPosition
        self.m_mass_spring_model[self.MASS_IDX][moved_link_id].m_velocity = newPosition - position
        ikResult = p.calculateInverseKinematics(self.m_sphere_uid, moved_link_id, newPosition)
        joint_positions[moved_link_id] = ikResult

        # Update other links in physical model:
        # We are about to cascade the effects of moved link changing position
        # It should be fine to use spring lengths as is - i.e., as diff between (new) parent mass pos and other mass pos
        # Later, all springs will be updated after all new mass locations are determined.
        for i in range(moved_link_id - 1, 1, -1):
            joint_positions[i] = self.ComputeNewJointPosition(i, i + 1, dt, outfile=outfile)

        for i in range(moved_link_id + 1, self.Size() - 1):
            joint_positions[i] = self.ComputeNewJointPosition(i, i - 1, dt, outfile=outfile)

        # TODO: figure out how to handle the last node as it doesn't have a right spring as reference
        #       generally this scheme should be able to handle the edge nodes similarly
        #       for now, do a lazy copy from endpoints
        self.update_ends(0, 1)
        joint_positions[0] = joint_positions[1]
        self.update_ends(self.Size() - 1, self.Size() - 2)
        joint_positions[self.Size() - 1] = joint_positions[self.Size() - 2]

        # Update joint positions in geometric model
        # [unsure] and feed link positions back to physical model
        springs = self.m_mass_spring_model[self.SPR_IDX]
        masses = self.m_mass_spring_model[self.MASS_IDX]
        positions = []
        for i in np.arange(m_length):
            p.setJointMotorControl2(self.rope_obj.m_sphere_uid,
                                    i,
                                    p.POSITION_CONTROL,
                                    targetPosition=joint_positions[i],
                                    force=30)

            positions.append(masses[i].m_position)
            x = RopeObj.link_states(self.m_obj.m_sphere_uid, i, False)[0]
            print("joint: {}\nx0={}\nx1={}\n".format(i, x, masses[i].m_position))
            masses[i].SetPositionAndVelocity(x, masses[i].m_velocity)

        for i in np.arange(m_length - 1):
            springs[i].SetOrientation(masses)


    def Size(self):
        return self.m_mass_spring_model.shape[1]
