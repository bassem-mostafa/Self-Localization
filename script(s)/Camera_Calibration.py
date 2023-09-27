# Import required modules
import cv2
import numpy as np
import os
import glob
import time


# Define the dimensions of checkerboard
CHECKERBOARD_CELL = (22.3, 22.3) # height, & width of checkerboard cell in milli-meter
CHECKERBOARD = (7, 7)            # height, & width of checkerboard in cells inner corners for 8x8 checker board there is 7x7 inner corners


# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []


# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0]
                    * CHECKERBOARD[1],
                    3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                            0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None


# Extracting path of individual image stored
# in a given directory. Since no path is
# specified, it will take current directory
# jpg files alone
# images = glob.glob('*.jpg')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception('Camera Open Failed')

while True:
    _timestamp_start = time.time()
    ret, image = cap.read()
    image = cv2.flip(image, 1) # mirror frame horizontly
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    # If desired number of corners are
    # found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If desired number of corners can be detected then,
    # refine the pixel coordinates and display
    # them on the images of checker board
    if ret == True:
        threedpoints.append(objectp3d)
        # Refining pixel coordinates
        # for given 2d points.
        corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
        twodpoints.append(corners2)
        # Draw and display the corners
        image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)

    cv2.imshow('img', image)
    _key = (cv2.waitKey(1000) & 0xFF)
    if _key in [27, ord('q'), ord('Q')]: break

cv2.destroyAllWindows()

print('calibration in progress...')

# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)

print('calibration done')

# Displaying required output
print(" Camera matrix:")
print(matrix)
print("\n Distortion coefficient:")
print(distortion)
print("\n Rotation Vectors:")
print(r_vecs)
print("\n Translation Vectors:")
print(t_vecs)

with open('intrinsic.npy', 'wb') as f:
    np.save(f, matrix)
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, grayColor.shape[::-1], 1, grayColor.shape[::-1])

print(" New Camera matrix:")
print(newCameraMatrix)

with open('intrinsicNew.npy', 'wb') as f:
    np.save(f, newCameraMatrix)

# Undistort
dst = cv2.undistort(grayColor, matrix, distortion, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('caliResult1.png', dst)

# Reprojection Error
mean_error = 0

for i in range(len(threedpoints)):
    imgpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
    error = cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(threedpoints)) )