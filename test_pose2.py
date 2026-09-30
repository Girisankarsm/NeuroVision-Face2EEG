import cv2
import numpy as np

rmat, _ = cv2.Rodrigues(np.array([[-3.1], [0], [0]], dtype=float))
angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
pitch, yaw, roll = angles
pitch = (pitch + 180) % 360 - 180
yaw = (yaw + 180) % 360 - 180
roll = (roll + 180) % 360 - 180

if abs(pitch) > 90 and abs(roll) > 90:
    pitch = pitch - 180 if pitch > 0 else pitch + 180
    roll = roll - 180 if roll > 0 else roll + 180
    yaw = -yaw
print(pitch, yaw, roll)
