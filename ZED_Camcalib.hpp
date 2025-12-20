#ifndef ZED_CAMCALIB
#define ZED_CAMCALIB

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>


//Function
//Uses Rectified Parameters
void getZedCalibration(sl::Camera & zed,cv::Mat& cameraMatrix,cv::Mat& distCoeffs);


#endif
