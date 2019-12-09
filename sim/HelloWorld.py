#!/usr/bin/python3

# Walk through - https://towardsdatascience.com/simulate-images-for-ml-in-pybullet-the-quick-easy-way-859035b2c9dd
#
# A quick run through setting up a simulation with random objects and producing images from some camera array.




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
import os, glob, random
import numpy as np
from matplotlib import pyplot as plt
from math import cos, sin
import pybullet_data
import cv2 as cv

pb_random_urdfs_dir = "/home/bilkit/Workspace/pybullet/models/random_urdfs"
pb_textures_dir = "/home/bilkit/Workspace/pybullet/textures/dtd/images/stratified"
IMG_WIDTH = 300
IMG_HEIGHT = 300
MODEL_INITIAL_POS = [0,0,1]

class MouseEventEnum:
	EventType = 0
	MousePosX = 1
	MousePosY = 2
	ButtonIndex = 3
	ButtonState = 4
class MouseButtonEnum:
	Left = 0
	Middle = 1
	Right = 2

def LoadModel(modelGeometryFile, modelCollisionFile="", texturePath=""):
	print("Loading urdf")
	if (not os.path.exists(modelGeometryFile) and not os.path.exists(modelCollisionFile)):
		print("Failed to find model paths")
		return 

	meshScale = [0.1, 0.1, 0.1]
	visId = p.createVisualShape(
		shapeType = p.GEOM_MESH,
		fileName = modelGeometryFile,
		rgbaColor = None,
		meshScale = meshScale)
	colId = p.createCollisionShape(
		shapeType = p.GEOM_MESH,
		fileName = modelCollisionFile,
		meshScale = meshScale)
	multiBodyId = p.createMultiBody(
		baseMass = 1.0,
		baseCollisionShapeIndex = colId,
		baseVisualShapeIndex = visId,
		basePosition = MODEL_INITIAL_POS,
		baseOrientation = p.getQuaternionFromEuler([0,0,0]))

	# Override textures
	p.changeVisualShape(multiBodyId, -1, textureUniqueId = p.loadTexture(textPath))
	return

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

print("Connecting to gui")
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
planeId = p.loadURDF('plane.urdf')
p.setGravity(0,0,-9.81)


# Load random visual and physical shapes
modelId = str(random.randint(0, 999)).zfill(3) if modelId < 0 else modelId
texPaths = glob.glob(os.path.join(pb_textures_dir, "**", "*.jpg"), recursive = True)
assert (len(texPaths) > 0), "failed to load texture paths"
LoadModel(
	os.path.join(pb_random_urdfs_dir, "{0}/{0}.obj".format(modelId)),
	os.path.join(pb_random_urdfs_dir, "{0}/{0}_coll.obj".format(modelId)),
	texPaths[random.randint(0, len(texPaths) - 1)])

# Setup cameras
A = 30 		# angular resolution in degrees 
R = 2		# radial distance in meters
N = 135 // A
cams = [None] * N 
print("Creating {} cams".format(N))
for i in range(N):
	x = R * cos(i * A * np.pi / 180) 
	y = R * sin(i * A * np.pi / 180) 
	z = 0

	view, proj = PlaceCamera([x, y, z], MODEL_INITIAL_POS)
	cams[i] = [view, proj]

# Record images
cid = 0
for c in cams:
	rgbaImg = GetImageFromCamera(c[0], c[1])
	rgbImg = rgbaImg[:,:,:3]
	cv.imwrite("{}_rgb.png".format(cid), rgbImg)
	cid += 1

# Better to use loop for batches of objs
# NOTE: will get "XIO:  fatal IO error 11" when closing window before timeout
shouldExit = False
for i in range(500):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()
print("Disconnecting from gui")
