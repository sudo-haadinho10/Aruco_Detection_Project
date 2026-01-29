#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/highgui/highgui_c.h>
#include <bits/stdc++.h>

//(k1, k2, p1, p2, k3, k4, k5, k6)


using namespace cv;
using namespace std;
// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 
float sqsize = 25.0f;
int main()
{
  // Creating vector to store vectors of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f> > objpoints;
 
  // Creating vector to store vectors of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f> > imgpoints;
 
  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++)
  {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j*sqsize,i*sqsize,0));
  }
 
 
  // Extracting path of individual image stored in a given directory
  std::vector<cv::String> images;
  // Path of the folder containing checkerboard images
  //std::string path = "/home/srinath/changemotora/op2/*.jpg";
  //
  std::string path = "/home/srinath/arducam_calib_pics/*.jpg";

  cv::glob(path, images);
 
  cv::Mat frame, gray;
  // vector to store the pixel coordinates of detected checker board corners 
  std::vector<cv::Point2f> corner_pts;
  bool success;
 
  // Looping over all the images in the directory
  for(int i=0; i<images.size(); i++)
  {
    frame = cv::imread(images[i]);
    cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
 
    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
    success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, CALIB_CB_ADAPTIVE_THRESH  |CALIB_CB_NORMALIZE_IMAGE);
     
    /* 
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display 
     * them on the images of checker board
    */
    if(success)
    {
      cv::TermCriteria criteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.001);
       
      // refining pixel coordinates for given 2d points.
      cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
       
      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, success);
       
      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }
 
    cv::imshow("Image",frame);
    //std::string output_filename="/home/srinath/op77/"+std::to_string(i)+".jpg";
    std::string output_filename="/home/srinath/OP_arducam_calib_pics/"+std::to_string(i)+".jpg";
    cv::imwrite(output_filename,frame);
    cv::waitKey(0);
  }
 
  cv::destroyAllWindows();
 
  cv::Mat cameraMatrix,distCoeffs;
  std::vector<cv::Mat> R;
  std::vector<cv::Mat> T;
 
  /*
   * Performing camera calibration by 
   * passing the value of known 3D points (objpoints)
   * and corresponding pixel coordinates of the 
   * detected corners (imgpoints)
  */

  if (objpoints.empty() || imgpoints.empty()) {
std::cerr << "No valid checkboard detections. Calibration aborted"<<"\n";
    	return -1;
	}
  //int flags=0;

  int flags = cv::CALIB_RATIONAL_MODEL;
  //flags |= cv::CALIB_FIX_K1;
  //flags |= cv::CALIB_FIX_K2;

  double retval =  cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.cols,gray.rows), cameraMatrix, distCoeffs, R, T,flags); //width,height

  std::cout<<"Image width"<< gray.cols<<"\n";
  std::cout<<"Image height"<<gray.rows<<"\n";
  std::cout<<"retval : "<<retval<<"\n";
  std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
  std::cout << "distCoeffs : " << distCoeffs << std::endl;

  // Save the calibration results to a YAML file
  cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE);
  if (fs.isOpened())
  {
    fs << "cameraMatrix" << cameraMatrix;
    fs << "distCoeffs" << distCoeffs;
    fs.release();
    std::cout << "\nCalibration data saved to calibration.yml" << std::endl;
  }
  else
  {
    std::cerr << "Error: Could not open the file 'calibration.yml' for writing." << std::endl;
  }


  cout<<"Object Points"<<"\n";
  //for(int i=0;i<images.size();i++) {
//	  cout<<objpoints[i]<<"\n";
  //}
  //
  for (int i = 0; i < objpoints.size(); i++)
  {
    // Printing the points for one of the views
    // This will be a long list of coordinates
    cout << "View " << i << ": [";
    for(size_t j = 0; j < objpoints[i].size(); ++j) {
        cout << objpoints[i][j] << (j == objpoints[i].size() - 1 ? "" : ", ");
    }
    cout << "]" << endl;
  }

  std::cout << "\n\n--- All Detected Image Points ---" << std::endl;
  for (size_t i = 0; i < imgpoints.size(); ++i)
  	{
  	std::cout << "View " << i << ": [" << std::endl;
  	for (size_t j = 0; j < imgpoints[i].size(); ++j)
  	{
    		// The cv::Point2f object can be directly sent to cout
    		std::cout << "  " << imgpoints[i][j] << (j == imgpoints[i].size() - 1 ? "" : ",") << std::endl;
  	}
  	std::cout << "]" << std::endl;
	}
  
  // std::cout << "Rotation vector : " << R << std::endl;
  //std::cout << "Translation vector : " << T << std::endl;
 
  return 0;
}
