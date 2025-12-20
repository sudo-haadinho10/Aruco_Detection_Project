#include "ZED_Camcalib.hpp"
//Uses Rectified Parameters 
//

using namespace sl;
using namespace cv;
using namespace std;


void getZedCalibration(Camera & zed,cv::Mat& cameraMatrix,cv::Mat& distCoeffs) {
	CalibrationParameters calibration_params = zed.getCameraInformation().camera_configuration.calibration_parameters;

	/*if(cameraMatrix.empty() || distCoeffs.empty()) {
		throw runtime_error("Error: Failed to retrieve ZED calibration parameters");
	}*/


	CameraParameters zedParams;
	cameraMatrix = cv::Mat::eye(3,3,CV_64F);

	zedParams = calibration_params.left_cam;

	// Focal length of the left eye in pixels
	//Camera Matrix has the intrinsic Parameters fx,fy,cx,cy
	cameraMatrix.at<double>(0,0) = zedParams.fx;
	cameraMatrix.at<double>(1,1) = zedParams.fy;
	cameraMatrix.at<double>(0,2) = zedParams.cx;
	cameraMatrix.at<double>(1,2) = zedParams.cy;

	//DISTORTION Coefficients 
	//
	distCoeffs = cv::Mat::zeros(1,5,CV_64F);
	distCoeffs.at<double>(0,0) = zedParams.disto[0]; //k1
	distCoeffs.at<double>(0,1) = zedParams.disto[1]; //k2
	distCoeffs.at<double>(0,2) = zedParams.disto[2]; //p1
	distCoeffs.at<double>(0,3) = zedParams.disto[3]; //p2
	distCoeffs.at<double>(0,4) = zedParams.disto[4]; //k3

	return;
}


int main(void) {
	
	Camera zed;
	cv::Mat CameraMatrix;
	cv::Mat distCoeffs;

	InitParameters init_params;
	init_params.camera_resolution = RESOLUTION::HD720;
	init_params.coordinate_units = sl::UNIT::MILLIMETER;

	sl::ERROR_CODE err = zed.open(init_params);
	if(err!=sl::ERROR_CODE::SUCCESS) {
		std::cerr << "Error opening ZED Camera: " << err <<"\n";
		return 1;
	}

	getZedCalibration(zed,CameraMatrix,distCoeffs);

	std::cout<<"Camera Matrix: \n" << CameraMatrix << "\n";
	std::cout << "Distortion Coefficients \n" <<distCoeffs <<"\n";

	zed.close();
	return 0;
}
