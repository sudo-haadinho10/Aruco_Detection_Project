#ifndef UTILITIES_H
#define UTILITIES_H

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

cv::cuda::GpuMat slMat2cvMatGPU(sl::Mat& input);
cv::Mat slMat2cvMat(sl::Mat& input);

// Mapping between MAT_TYPE and CV_TYPE
int getOCVtype(sl::MAT_TYPE type);

#endif
