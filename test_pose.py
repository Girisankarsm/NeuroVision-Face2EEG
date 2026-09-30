import cv2
import math
import numpy as np
from neurovision.preprocessing.facial import CANONICAL_FACE_3D
from scipy.spatial.transform import Rotation

# Simulate looking straight at camera
image_points = np.array([
    [320, 240], # Nose
    [320, 300], # Chin
    [280, 200], # Left eye
    [360, 200], # Right eye
    [290, 270], # Left mouth
    [350, 270], # Right mouth
], dtype=np.float64)

focal_length = 640
center = (320, 240)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype=np.float64,
)
dist_coeffs = np.zeros((4, 1), dtype=np.float64)

success, rvec, tvec = cv2.solvePnP(
    CANONICAL_FACE_3D,
    image_points,
    camera_matrix,
    dist_coeffs,
    flags=cv2.SOLVEPNP_ITERATIVE,
)

rmat, _ = cv2.Rodrigues(rvec)
angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
print("RQDecomp3x3:", angles)

# Custom robust method
sy = np.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
singular = sy < 1e-6
if not singular:
    x = math.atan2(rmat[2,1] , rmat[2,2])
    y = math.atan2(-rmat[2,0], sy)
    z = math.atan2(rmat[1,0], rmat[0,0])
else:
    x = math.atan2(-rmat[1,2], rmat[1,1])
    y = math.atan2(-rmat[2,0], sy)
    z = 0
x = np.degrees(x)
y = np.degrees(y)
z = np.degrees(z)
print("Custom:", x, y, z)
