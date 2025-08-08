#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <ctime>
#include <cstdint>
#include <map>
#include <string>

struct MarkerData {
	int id; //ID of each aruco marker
	uint16_t pixel_x; 
	uint16_t pixel_y;
	float float_x; //Translation x (cm)
	float float_y; //Translation y (cm)
	float float_z; //Translation z (cm)
	float angle_x; //Rotation x (degrees)
	//
	uint32_t update_time; 
	uint32_t access_time;
};

//Global storage for marker data 

void loadCalibration(const string& filename, Mat& cameraMatrix, Mat& distCoeffs);
void loadCalibration(const string& filename, Mat& cameraMatrix, Mat& distCoeffs);

