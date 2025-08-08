import cv2
import numpy as np
import os
import glob

CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
print(cv2.__version__)
objpoints = []
imgpoints=[]

objp=np.zeros((1,CHECKERBOARD[0]*CHECKERBOARD[1],3),np.float32)
objp[0,:,:2]= np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
prev_img_shape=None;

images=glob.glob("/home/srinath/camcalib/nvcamtest_*.jpg")
#images=glob.glob("/home/srinath/camcalib/nvcamtest_7788_s00__*.jpg")
if not images:
    raise ValueError("No images found in camcalib")
for fname in images:
    print("done")
    img=cv2.imread(fname)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if gray is None:
        print(f"Failed to convert image to grayscale: {fname}")
        continue

    if gray.dtype !=np.uint8:
        gray=gray.astype(np.uint8)
        print(f"Converted {fname} to uint8")
    ret,corners=cv2.findChessboardCorners(gray,CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
        print("done")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        img =cv2.drawChessboardCorners(img,CHECKERBOARD,corners2,ret)
    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

if not objpoints or not imgpoints:
    raise ValueError("No valid checkerboard corners were detected in my image")


h,w = img.shape[:2]

try:
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)
except cv2.error as e:
    print(f"Calibration failed {e}")
    raise

print("Camera matrix \n")
print(mtx)
print("dist:\n")
print(dist)
print("rvecs:\n")
print(rvecs)
print("tvecs : \n")
print(tvecs)
np.save("Calibration matrix",mtx)
np.save("distortion_coefficients",dist)
