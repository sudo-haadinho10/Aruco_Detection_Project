#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <cstdint>
#include <map>
#include <thread>
#include <ws.h>
#include "wsServer.h"
#include <opencv2/calib3d.hpp> //for findHomography
#include <opencv2/imgproc.hpp> // for warpPerspective
#include <sl/Camera.hpp>
#include "utilities.hpp"
#include "ZED_Camcalib.hpp"
//#include "poseestimation.hpp"

#include "teraGrid.hpp"

WallDetails_t WallDetails;

int allDetected = false;

void Init_WSSocket(void) {
    // 1. Define the event handlers
	//struct ws_events evs;
	//evs.onopen    = &onopen;
	//evs.onclose   = &onclose;
	//evs.onmessage = &onmessage;

    // 2. Configure the server properties in a ws_server struct
    // This is the key change to fix the error.
    struct ws_server server;
    memset(&server, 0, sizeof(struct ws_server)); // Good practice to zero it out

    server.host = "0.0.0.0";    // Listen on all available network interfaces
    server.port = 9000;         // The port you want to listen on
    //server.events = &evs;       // Point to your event handler functions
    //server.max_clients = 5;     // Set the maximum number of connected clients
    //
    server.timeout_ms = 1000;
    server.thread_loop = 1;
    server.evs.onopen = &onopen;
    server.evs.onclose = &onclose;
    server.evs.onmessage = &onmessage;
	// 3. Call ws_socket() with the single server struct argument
    // This is a blocking call that will run the server.
	ws_socket(&server);
}


using namespace cv;
using namespace std;


// CORRECTED: The map value type is simply 'int'
map<string, int> ARUCO_DICT = {
    {"DICT_4X4_50", aruco::DICT_4X4_50},
    {"DICT_4X4_100", aruco::DICT_4X4_100},
    {"DICT_4X4_250", aruco::DICT_4X4_250},
    {"DICT_4X4_1000", aruco::DICT_4X4_1000},
    {"DICT_5X5_50", aruco::DICT_5X5_50},
    {"DICT_5X5_100", aruco::DICT_5X5_100},
    {"DICT_5X5_250", aruco::DICT_5X5_250},
    {"DICT_5X5_1000", aruco::DICT_5X5_1000},
    {"DICT_6X6_50", aruco::DICT_6X6_50},
    {"DICT_6X6_100", aruco::DICT_6X6_100},
    {"DICT_6X6_250", aruco::DICT_6X6_250},
    {"DICT_6X6_1000", aruco::DICT_6X6_1000},
    {"DICT_7X7_50", aruco::DICT_7X7_50},
    {"DICT_7X7_100", aruco::DICT_7X7_100},
    {"DICT_7X7_250", aruco::DICT_7X7_250},
    {"DICT_7X7_1000", aruco::DICT_7X7_1000},
    {"DICT_ARUCO_ORIGINAL", aruco::DICT_ARUCO_ORIGINAL}
};

// Data structure to hold smoothed marker data
struct MarkerData {
	int id;
	uint16_t pixel_x;
	uint16_t pixel_y;
	float float_x; // Smoothed Translation x (cm)
	float float_y; // Smoothed Translation y (cm)
	float float_z; // Smoothed Translation z (cm)
	float rvec_x;  // Smoothed Rotation vector x
	float rvec_y;  // Smoothed Rotation vector y
	float rvec_z;  // Smoothed Rotation vector z
	uint32_t update_time; //(ms since epoch)
	uint32_t access_time; //(ms since epoch)
};
// Global storage for marker data
map<int,MarkerData> markerDataMap;

// Function to load calibration matrices from YAML file
void loadCalibration(const string& filename, Mat& cameraMatrix, Mat& distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        throw runtime_error("Error: Cannot open calibration file: " + filename);
    }
    fs["camera_matrix_ltu"] >> cameraMatrix;
    fs["dist_coeffs_ltu"] >> distCoeffs;
    fs.release();

    if (cameraMatrix.empty() || distCoeffs.empty()) {
        throw runtime_error("Error: Loaded calibration matrices are empty");
    }
}


/**
 * @brief Calculates a stable average rvec from a list, rejecting outliers.
 * @param rvecs A vector of rvecs, presumably from markers on the same wall.
 * @return A single, averaged rvec.
 */

cv::Vec3d calculateAverageRvec(const std::vector<cv::Vec3d>& rvecs) {
	if(rvecs.empty()) {
		return cv::Vec3d(0,0,0); //
	}

	cv::Mat reference_R;
	cv::Rodrigues(rvecs[0],reference_R);
	
	
	//Starting with the sum of first rvec which is always valid
	cv::Vec3d sum_of_rvecs = rvecs[0];
	int valid_count = 1;

	//rvecs -> magnitude gives angle of rotation in radians
	//rvecs -> direction indicates axis of rotation
	//
	for(size_t i=1;i<rvecs.size();i++) {
		
		//sqrt(x1-x2^2 + y1-y2^2 + z1-z2^2)
		//L2 NORM IS THE EUCILDEAN DISTANCE
		//cv::norm calculates the distance between two vectors?
		
		//Approach 2
		//convert the current vec to a rotation matrix
		cv::Mat current_R;
		cv::Rodrigues(rvecs[i],current_R);

		//Calculate the true angular difference between the two orientations
		//This is the fix to the above 2pi problem ambiguity in rvecs
		//R_camera_to_world

		//multiplication chains transformations, and transpose reverses them!
		//
		cv::Mat R_diff = current_R * reference_R.t();
		//true geometric angle b/w orientations
		//
		cv::Vec3d rvec_diff;
		cv::Rodrigues(R_diff,rvec_diff);
		
		double error = cv::norm(rvec_diff);

		//If orientation differs by more than -60 degrees (~1 rad) 
		//We assume it is an outlier or upside down tilted marker..
		//
		const double outlier_threshold = 1.0;
		if(error < outlier_threshold) {
			sum_of_rvecs +=rvecs[i];
			valid_count+=1;
		}
		else {
			std::cout<<"Warning: Discarding outlier rvec"<<i<<" Error was: "<<error<<"\n";
		}
	}

	std::cout << "Averaging " << valid_count << " of " << rvecs.size() << " visible markers." << std::endl;

	return sum_of_rvecs/valid_count; //argRvec

}

/**
 * @brief Calculates the drone's orientation in the world frame using observations from one wall.
 * @param wall_data The struct containing all observations for a single wall.
 * @return A 3x3 rotation matrix representing the drone's orientation in the world.
 */

cv::Mat getDroneOrientationInWorld(const WallObservations& wall_data) {
	
	//GOAL
	//R_drone_in_world = R_wall_in_world * R_wall_in_camera.t().
	//R_drone_in_world is Camera->Wall->World
	
	//1. Get the stable,average orientation of the Wall in Camera frame
	//
	cv::Vec3d avg_rvec_wall_in_camera = calculateAverageRvec(wall_data.rvecs);
	cv::Mat R_wall_in_camera;
	cv::Rodrigues(avg_rvec_wall_in_camera,R_wall_in_camera); //rotation matrix

	cv::Mat R_wall_in_world;

	//I made a custom wall coordinates axis as per my wish i.e our aluminium foil
	//WALL X AXIS POINTS TOWARDS WORLD'S Z
	//wall y axis points towards world y
	//wall z axis points towards world x+ve
	//
	//
	//
	//
	
	// Each column represents where the wall's X, Y, Z axes point in camera coordinates
	cv::Vec3d wall_X_in_camera(R_wall_in_camera.at<double>(0,0),
			R_wall_in_camera.at<double>(1,0),
			R_wall_in_camera.at<double>(2,0));
	cv::Vec3d wall_Y_in_camera(R_wall_in_camera.at<double>(0,1),
			R_wall_in_camera.at<double>(1,1),
                        R_wall_in_camera.at<double>(2,1));
	cv::Vec3d wall_Z_in_camera(R_wall_in_camera.at<double>(0,2),
                        R_wall_in_camera.at<double>(1,2),
                        R_wall_in_camera.at<double>(2,2));

	std::cout << "Wall X-axis in camera: " << wall_X_in_camera << std::endl;
	std::cout << "Wall Y-axis in camera: " << wall_Y_in_camera << std::endl;
	std::cout << "Wall Z-axis in camera (normal): " << wall_Z_in_camera << std::endl;


	if(wall_data.id ==0) {
		//NORTH WALL
		//
	//	R_wall_in_world = (cv::Mat_<double>(3,3) <<
	//			0,0,-1, //Wall Z axis points towards Worlds x axis
	//			0,1,0,
	//			1,0,0 //Wall x axis points towards worlds z axis
	//	);
	//
		R_wall_in_world = (cv::Mat_<double>(3,3) <<
                                1,0,0, //Wall Z axis points towards Worlds x axis
                                0,-1,0,
                                0,0,-1 //Wall x axis points towards worlds z axis
                );

	}
	//else for future walls / faces
	//
	cv::Mat R_drone_in_world = R_wall_in_world * R_wall_in_camera.t();
	return R_drone_in_world;

}

/**
 * @brief Converts a 3x3 Rotation Matrix to Euler angles (Yaw, Pitch, Roll).
 * @param R The input rotation matrix.
 * @return A cv::Vec3d containing (Roll, Pitch, Yaw) in radians.
 */

cv::Vec3d rotationMatrixToEulerAngles(const cv::Mat &rotation_matrix) {
	//1. The inverse of a 3x3 Rotation matrix is directly it's transpose
	//2. Decompose the rotation matrix to get Euler angles

	double sy = std::sqrt(rotation_matrix.at<double>(0,0) * rotation_matrix.at<double>(0,0) +  rotation_matrix.at<double>(1,0) * rotation_matrix.at<double>(1,0));
        bool singular = sy < 1e-6; // Check for singularity (gimbal lock)
        double roll,pitch,yaw;

       
        if(!singular) 
	{
		pitch  = std::atan2(rotation_matrix.at<double>(2,1) , rotation_matrix.at<double>(2,2));
                yaw = std::atan2(-rotation_matrix.at<double>(2,0), sy);
                roll   = std::atan2(rotation_matrix.at<double>(1,0), rotation_matrix.at<double>(0,0));
        }

        else 
	{
		pitch = std::atan2(-rotation_matrix.at<double>(1,2), rotation_matrix.at<double>(1,1));
                yaw = std::atan2(-rotation_matrix.at<double>(2,0), sy);
                roll   = 0.0f;
        }
	return cv::Vec3d(roll,pitch,yaw);
}




void poseEstimationV2(cv::Mat& frame,int arucoDictType,cv::Mat& matrixCoefficients,cv::Mat& distortionCoefficients) 
{

	//Infinitesimal Plane-Based Pose Estimation 
	//This is a special case suitable for marker pose estimation.
	//
	float markerLength = 10.0;
	float halfMarkerLength = markerLength/2.0;
	std::vector<cv::Point3f> objectPoints;

	//The 4 coplanar object points must be defined in the following order:
	
	objectPoints.push_back({-halfMarkerLength,halfMarkerLength,0});
        objectPoints.push_back({halfMarkerLength,halfMarkerLength,0});
        objectPoints.push_back({halfMarkerLength,-halfMarkerLength,0});
        objectPoints.push_back({-halfMarkerLength,-halfMarkerLength,0});


	cv::Mat gray;
	cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

    	aruco::Dictionary markerDict = aruco::getPredefinedDictionary(arucoDictType);
    	aruco::DetectorParameters paramMarkers = aruco::DetectorParameters();

    	//paramMarkers.cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX; //SUB PIXEL ACCURACY, WE ALSO TRY CORNER_REFINE_APRILTAG for better robustness in detected values
        paramMarkers.cornerRefinementMethod = aruco::CORNER_REFINE_APRILTAG;
    	aruco::ArucoDetector detector(markerDict, paramMarkers);

    	vector<vector<Point2f>> markerCorners;
    	vector<int> markerIDs;
    	detector.detectMarkers(gray, markerCorners, markerIDs);

	std::vector<Vec3d> rvecs,tvecs;


	uint32_t currentTime = static_cast<uint32_t>(chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count());

	if(!markerCorners.empty()) 
	{
		   //aruco::estimatePoseSingleMarkers(markerCorners,10, matrixCoefficients, distortionCoefficients, rvecs, tvecs);  //10cm, tvecs will be in the same uniitt
                //const float alpha = 0.8f;
                //
		//
		//New Additionn
		//

		for(size_t i=0;i<markerCorners.size();i++) 
		{
			cv::Vec3d rvec,tvec;

			// Calculate the pose of the current marker
			cv::solvePnP(objectPoints,markerCorners.at(i),matrixCoefficients,distortionCoefficients,rvec,tvec,false,cv::SOLVEPNP_IPPE_SQUARE);
			rvecs.push_back(rvec);
			tvecs.push_back(tvec);
		}

		std::vector<cv::Vec3d> visible_smoothed_tvecs;
 	        std::vector<cv::Vec3d> visible_smoothed_rvecs;

		//
		std::vector<int> visible_marker_ids;
		std::vector<float> visible_marker_ranges;
		//
		//


		for(size_t i=0;i<markerIDs.size();i++) 
		{
			int current_id = markerIDs[i];
            		Vec3d current_tvec = tvecs[i];
            		Vec3d current_rvec = rvecs[i];
			
			float avg_x=0, avg_y=0;
            		for(const auto& corner : markerCorners[i])
			{
                		avg_x += corner.x;
                		avg_y += corner.y;
            		}
            		avg_x /= 4.0f;
            		avg_y /= 4.0f;

            		MarkerData data;

			//Optional temporal smoothing logic(In case the markers are static and there is lot of noise
		/*	if (markerDataMap.count(current_id)) 
			{
                		MarkerData& prev_data = markerDataMap.at(current_id);
                		// Apply exponential moving average for smoothing
				data.float_x = alpha * current_tvec[0] + (1.0f - alpha) * prev_data.float_x;
                		data.float_y = alpha * current_tvec[1] + (1.0f - alpha) * prev_data.float_y;
                		data.float_z = alpha * current_tvec[2] + (1.0f - alpha) * prev_data.float_z;
                		data.rvec_x = alpha * current_rvec[0] + (1.0f - alpha) * prev_data.rvec_x;
                		data.rvec_y = alpha * current_rvec[1] + (1.0f - alpha) * prev_data.rvec_y;
                		data.rvec_z = alpha * current_rvec[2] + (1.0f - alpha) * prev_data.rvec_z;
            	       }*/
            	       //else 
		       //{
			       // First time seeing this marker, initialize directly
			       data.float_x = current_tvec[0];
			       data.float_y = current_tvec[1];
			       data.float_z = current_tvec[2];
			       data.rvec_x = current_rvec[0];
			       data.rvec_y = current_rvec[1];
			       data.rvec_z = current_rvec[2];
            	      //}
		      data.id = current_id;
	 	      data.pixel_x = static_cast<uint16_t>(avg_x);
            	      data.pixel_y = static_cast<uint16_t>(avg_y);
            	      data.update_time = currentTime;
            	      data.access_time = currentTime;
            	      markerDataMap[current_id] = data;

		      //DEBUG: PRINTING RVECS

		      cout<<"Id:"<<current_id<<"\n"<<current_rvec[0]<<" "<<current_rvec[1]<<" "<<current_rvec[2]<<"\n";

		      float magnitude = std::sqrt(current_rvec[0]*current_rvec[0] + current_rvec[2]*current_rvec[2]);
		      cout<<"ID:"<<current_id<<"\n"<<"Degrees:"<<(magnitude*180)/PI<<"\n";

		      visible_smoothed_tvecs.emplace_back(data.float_x,data.float_y,data.float_z);
		      visible_smoothed_rvecs.emplace_back(data.rvec_x,data.rvec_y,data.rvec_z);
		      visible_marker_ids.emplace_back(current_id);
		      cv::Vec3d current_smoothed_pos(data.float_x,data.float_y,data.float_z);
		      float range = cv::norm(current_smoothed_pos); //Calculates sqrt(x^2 + y^2 + z^2)
		      visible_marker_ranges.emplace_back(range);
		      
		      // Convert Point2f to Point for polylines to prevent runtime crash
 	              vector<Point> corners_int;
            	      corners_int.reserve(markerCorners[i].size());
            	      for (const auto& corner : markerCorners[i]) 
		      {
			      corners_int.push_back(Point((int)corner.x, (int)corner.y));
            	      }
            	      polylines(frame, corners_int, true, Scalar(0, 255, 255), 4, LINE_AA);
		      // Draw axes and text using the smoothed values
  	              cv::drawFrameAxes(frame, matrixCoefficients, distortionCoefficients, visible_smoothed_rvecs.back(), visible_smoothed_tvecs.back(), 5);
		
		      //cv::Point text_origin = {20,30};
		      //
		      cv::Point marker_origin(markerCorners[i][0].x, markerCorners[i][0].y);

		      cv::putText(frame, "ID: " + to_string(data.id),cv::Point(marker_origin.x,marker_origin.y-10), FONT_HERSHEY_PLAIN, 1.3, Scalar(255, 255, 255), 2, LINE_AA);

		      string transText = cv::format("Po: %.0f,%.0f,%.0f",data.float_x,data.float_y,data.float_z);
		      cv::putText(frame,transText,cv::Point(marker_origin.x,marker_origin.y+20),cv::FONT_HERSHEY_PLAIN,1.0,cv::Scalar(0,255,0),1,cv::LINE_AA);
		      
		      //Range/Euclidean distance
		      //
		      string rangeText = cv::format("Range: %.1f",range);
		      cv::putText(frame,rangeText,cv::Point(marker_origin.x,marker_origin.y+40),cv::FONT_HERSHEY_PLAIN,1.0,cv::Scalar(0,255,255),1,cv::LINE_AA);
		      // cv::putText(frame,"ID: " + to_string(data.id),text_origin+50,FONT_HERSHEY_PLAIN,1.3,Scalar(255,0,255),2,LINE_AA);
		}

		const int required_markers = 4; //Minimum 4 markers needed 
		if(visible_smoothed_tvecs.size()==required_markers) 
		{
			int num_visible = visible_marker_ids.size();

			//Group all visible markers by their assigned wall.
			//
			std::map<WallDirection_t,WallObservations> groupedObservations;

			for(size_t i = 0 ;i<visible_marker_ids.size();i++) {
				WallDirection_t wall = getWallForMarker(visible_marker_ids[i]);
				if(wall!=WALL_NONE) {
					groupedObservations[wall].id = wall;
					groupedObservations[wall].rvecs.push_back(visible_smoothed_rvecs[i]);
					groupedObservations[wall].tvecs.push_back(visible_smoothed_tvecs[i]);
				}
			}

			//Proceed if we successfully grouped markers from atleast one wall.
			//
			if(!groupedObservations.empty()) 
			
			{
				//Future strategy pick the wall with most markers?
				//Current implementation, take the first wall
				//const referece to object of type struct
				const WallObservations& testWall = groupedObservations.begin()->second;
				std::cout<<"---Testing Wall ID: "<<testWall.id <<" with "<<
					testWall.rvecs.size()<<" markers---"<<"\n";

				cv::Vec3d avg_rvec_wall_in_camera = calculateAverageRvec(testWall.rvecs);
				cv::Mat R_wall_in_camera;
				cv::Rodrigues(avg_rvec_wall_in_camera,R_wall_in_camera);
				//Use the euler angle conversion function
				cv::Vec3d ypr_angles_rad_camera_frame = rotationMatrixToEulerAngles(R_wall_in_camera);

				// Convert radians to degrees for easier reading
   			//	double camera_yaw_deg   = ypr_angles_rad_camera_frame[2] * 180.0 / CV_PI;
    			//	double camera_pitch_deg = ypr_angles_rad_camera_frame[1] * 180.0 / CV_PI;
    			//	double camera_roll_deg  = ypr_angles_rad_camera_frame[0] * 180.0 / CV_PI;
				cv::Mat R_drone_in_world;

				R_drone_in_world = getDroneOrientationInWorld(testWall);

				// The result is a vector of (Roll, Pitch, Yaw) in RADIANS.
				cv::Vec3d ypr_angles_rad_world_frame = rotationMatrixToEulerAngles(R_drone_in_world);

				// Step 2: Convert the radians to degrees for easy printing and debugging.
				double world_yaw_deg   = ypr_angles_rad_world_frame[2] * 180.0 / CV_PI;
				double world_pitch_deg = ypr_angles_rad_world_frame[1] * 180.0 / CV_PI;
				double world_roll_deg  = ypr_angles_rad_world_frame[0] * 180.0 / CV_PI;


				// Step 5: Display the results.
    				//std::cout << "Average Wall Orientation (in Camera Frame): Yaw=" << camera_yaw_deg<< ", Pitch=" << camera_pitch_deg<< ", Roll=" << camera_roll_deg << std::endl;


			 	cv::Vec3d rvec1 = avg_rvec_wall_in_camera; //avg rvec
                                string transText1 = cv::format("Avg Rvec: [%.2f, %.2f, %.2f]", rvec1[0], rvec1[1], rvec1[2]);
                                cv::putText(frame,transText1,cv::Point(10,70),cv::FONT_HERSHEY_PLAIN,1.2,cv::Scalar(255,255,0),2,cv::LINE_AA);

				string orientationText = cv::format("DRONE IN WORLD YPR: %.2f, %.2f, %.2f", world_yaw_deg, world_pitch_deg, world_roll_deg);
cv::putText(frame, orientationText, cv::Point(10, 92), FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 100, 0), 2);
				


				// Also display it on the video frame so you can see it in real-time.
			 	//
			 	for(int j=0;j<testWall.rvecs.size();j++) {
					cv::Point marker_origin(markerCorners[j][0].x, markerCorners[j][0].y);
					cv::Vec3d rvec = testWall.rvecs[j];
					string transText = cv::format("Rvec: [%.2f, %.2f, %.2f]", rvec[0], rvec[1], rvec[2]);

                                	cv::putText(frame,transText,cv::Point(marker_origin.x,marker_origin.y+65),cv::FONT_HERSHEY_PLAIN,1.2,cv::Scalar(255,255,0),2,cv::LINE_AA);
				}
			}

		/*	float sum_cos=0.0f;
			float sum_sin=0.0f;

			float sum_cos_roll = 0.0f;
			float sum_sin_roll = 0.0f;

			float sum_cos_pitch =0.0f;
			float sum_sin_pitch = 0.0f;

			std::string Yawval,Rollval,Pitchval;
			for(int j=0;j<visible_smoothed_tvecs.size();j++) 
			
			{
				cv::Mat rotation_matrix;
				cv::Mat rvec_mat(visible_smoothed_rvecs[j]); //constructor rvec_mat()
            			cv::Rodrigues(rvec_mat,rotation_matrix); //3x3 rotation matrix is the orientation of the object frame in the camera frame(opencv docs)
				//The inverse of a 3x3 Rotation matrix is directly it's transpose
            			// 2. Decompose the rotation matrix to get Euler angles
         
   			float sy = std::sqrt(rotation_matrix.at<double>(0,0) * rotation_matrix.at<double>(0,0) +  rotation_matrix.at<double>(1,0) * rotation_matrix.at<double>(1,0));
            			bool singular = sy < 1e-6; // Check for singularity (gimbal lock)
            			float roll,pitch,yaw;

            			//THE BELOW R,P,Y ARE GIVEN IN CAMERA FRAM I.E CAMERA PERSPECTIVE
				//CAMERA PITCH IS SAME AS DRONE YAW
				if(!singular) {
					roll  = std::atan2(rotation_matrix.at<double>(2,1) , rotation_matrix.at<double>(2,2));
                    			pitch = std::atan2(-rotation_matrix.at<double>(2,0), sy);
                    			yaw   = std::atan2(rotation_matrix.at<double>(1,0), rotation_matrix.at<double>(0,0));
            			}

            			else {
					roll  = std::atan2(-rotation_matrix.at<double>(1,2), rotation_matrix.at<double>(1,1));
                    			pitch = std::atan2(-rotation_matrix.at<double>(2,0), sy);
                    			yaw   = 0.0f;
            			}
				float drone_yaw = -1.0*pitch;
				float drone_pitch = roll; //drone_pitch is camera roll Yd == Xc
				float drone_roll = -1.0*yaw; //Xd == Zc

				
				//OPTIONAL DEBUGGING STUFF
				cv::Point marker_origin(markerCorners[j][0].x, markerCorners[j][0].y);
			        string transText = cv::format("Xd:%.0f,Yd:%.0f,Zd:%.0f",(drone_roll*180)/PI,(drone_pitch*180)/PI,(drone_yaw*180)/PI);
                                cv::putText(frame,transText,cv::Point(marker_origin.x,marker_origin.y+65),cv::FONT_HERSHEY_PLAIN,1.2,cv::Scalar(255,255,0),2,cv::LINE_AA);
				
				
				//---------------------------
				
				
				//drone yaw
				//
				sum_sin+=sin(drone_yaw);
				sum_cos+=cos(drone_yaw);

				//drone pitch Yd
				sum_sin_pitch+=sin(drone_pitch);
				sum_cos_pitch+=cos(drone_pitch);

				//drone roll
				sum_sin_roll+=sin(drone_roll);
				sum_cos_roll+=cos(drone_roll);



				//Yawval = cv::format("YAW: %f",(pitch*180)/PI); //degrees

				if(markerIDs[j]==2) {
				cout<<markerIDs[j]<<" "<<"RPY"<<"\n";
				cout<<(pitch*180)/PI<<"\n";
				}
			}

			float psi = std::atan2(sum_sin,sum_cos);
			float theta=std::atan2(sum_sin_pitch,sum_cos_pitch);
			float phi=std::atan2(sum_sin_roll,sum_cos_roll);

			Yawval = cv::format("YAW(Zd): %f(degrees)",(psi*180)/PI); //degree
			Rollval = cv::format("ROLL(Xd): %f(degrees)",(phi*180)/PI); //degrees
                        Pitchval = cv::format("PITCH(Yd): %f(degrees)",(theta*180)/PI); //degrees

			*/
			
			//Object position determination
			//
			teraGridLocalize(&teraGrid,num_visible,&visible_marker_ids[0],&visible_marker_ranges[0]);
			std::cout<<"NUM VISIBLE"<<num_visible<<"\n";
		
			std::cout<<"MARKER IDS\n"<<visible_marker_ids[0]<<" "<<visible_marker_ids[1]<<" "<<visible_marker_ids[2]<<" "<<visible_marker_ids[3]<<"\n";
		
			std::cout<<"MARKER RANGES\n"<<visible_marker_ranges[0]<<" "<<visible_marker_ranges[1]<<" "<<visible_marker_ranges[2]<<" "<<visible_marker_ranges[3]<<"\n";

			
			//Optional Visualization for Debugging
			//
			cv::Point text_origin(20,30);
			cv::putText(frame,"STATUS: Providing data to Triangulation", text_origin,FONT_HERSHEY_PLAIN,1.3,Scalar(0,255,0),2,LINE_AA);
			string countText = cv::format("Visible Markers: %d",num_visible);
			cv::putText(frame,countText,cv::Point(text_origin.x,text_origin.y+25),FONT_HERSHEY_PLAIN,1.3,Scalar(0,255,255),2,LINE_AA);

			//string rangeText = cv::format("Range  to ID %d: %.1f", visible_marker_ids[0],visible_marker_ranges[0]);
			//cv::putText(frame,rangeText,Point(text_origin.x,text_origin.y+50),FONT_HERSHEY_PLAIN,1.3,Scalar(255,255,255),2,LINE_AA);
			//
			//string rangeText = cv::format("%f %f %f\n", teraGrid.x.pData[0], teraGrid.x.pData[1],teraGrid.x.pData[2]);
			//cv::putText(frame,rangeText,cv::Point(text_origin.x,text_origin.y+50),FONT_HERSHEY_PLAIN,1.3,Scalar(255,255,255),2,LINE_AA);
			printf("x values\n %f %f %f\n",teraGrid.x.pData[0], teraGrid.x.pData[1],teraGrid.x.pData[2]);
			//printf("Yaw value(psi):%f\n",psi);
			//string YAWval = cv::format("YAW: %f",yaw);

			//cv::putText(frame,Yawval,cv::Point(text_origin.x,text_origin.y+50),FONT_HERSHEY_PLAIN,1.3,Scalar(255,255,255),2,LINE_AA);
			//cv::putText(frame,Rollval,cv::Point(text_origin.x,text_origin.y+75),FONT_HERSHEY_PLAIN,1.3,Scalar(255,255,255),2,LINE_AA);
                        //cv::putText(frame,Pitchval,cv::Point(text_origin.x,text_origin.y+100),FONT_HERSHEY_PLAIN,1.3,Scalar(255,255,255),2,LINE_AA);

			allDetected=true;
		}
		else 
		{
			cv::Point text_origin(20,30);
			cv::putText(frame,"Status: Less than 4 markers detected",text_origin,FONT_HERSHEY_PLAIN,1.3,Scalar(0,0,255),2,LINE_AA); 
		}

	}
}

// The function signature uses 'int' for the dictionary type
void poseEstimation(cv::Mat& frame, int arucoDictType, cv::Mat& matrixCoefficients, cv::Mat& distortionCoefficients) {
    cv::Mat gray;
    cv::cvtColor(frame, gray, COLOR_BGR2GRAY);

    aruco::Dictionary markerDict = aruco::getPredefinedDictionary(arucoDictType);
    aruco::DetectorParameters paramMarkers = aruco::DetectorParameters();

    paramMarkers.cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX; //SUB PIXEL ACCURACY, WE ALSO TRY CORNER_REFINE_APRILTAG for better robustness in detected values
    paramMarkers.cornerRefinementMethod = aruco::CORNER_REFINE_APRILTAG;
    aruco::ArucoDetector detector(markerDict, paramMarkers);

    vector<vector<Point2f>> markerCorners;
    vector<int> markerIDs;
    detector.detectMarkers(gray, markerCorners, markerIDs);

    uint32_t currentTime = static_cast<uint32_t>(chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count());

    if (!markerCorners.empty()) {
	    //9.7cm
        vector<Vec3d> rvecs, tvecs;
        aruco::estimatePoseSingleMarkers(markerCorners,9.7, matrixCoefficients, distortionCoefficients, rvecs, tvecs);  //10cm, tvecs will be in the same uniitt

        const float alpha = 0.1f; // Smoothing factor

        for (size_t i = 0; i < markerIDs.size(); ++i) {
            

	    //vector<cv::Point2f> source_points = markerCorners[i];

	    /*vector<cv::Point2f> destination_points = {
            cv::Point2f(0, 0),         // Top-Left corner
            cv::Point2f(199, 0),       // Top-Right corner
            cv::Point2f(199, 199),     // Bottom-Right corner
            cv::Point2f(0, 199)        // Bottom-Left corner
            };*/
		
	    //cv::Mat homography_matrix = cv::findHomography(source_points, destination_points);
	    
	  //  cv::Mat warped_marker_image;

	/*    cv::warpPerspective(frame,                  // The original image you want to transform
                            warped_marker_image,    // The output image
                            homography_matrix,      // The transformation matrix to use
                            cv::Size(200, 200));    // The size of the output image (must match your destination)

        // Show the result in a new window. You'll see the marker as a flat square.
        cv::imshow("Top-Down view of Marker ID: " + std::to_string(markerIDs[i]), warped_marker_image);
	*/

	    int current_id = markerIDs[i];
            Vec3d current_tvec = tvecs[i];
            Vec3d current_rvec = rvecs[i];

            float avg_x=0, avg_y=0;
            for(const auto& corner : markerCorners[i]){
                avg_x += corner.x;
                avg_y += corner.y;
            }
            avg_x /= 4.0f;
            avg_y /= 4.0f;

            MarkerData data;

            if (markerDataMap.count(current_id)) {
                MarkerData& prev_data = markerDataMap.at(current_id);
                // Apply exponential moving average for smoothing
                data.float_x = alpha * current_tvec[0] + (1.0f - alpha) * prev_data.float_x;
                data.float_y = alpha * current_tvec[1] + (1.0f - alpha) * prev_data.float_y;
                data.float_z = alpha * current_tvec[2] + (1.0f - alpha) * prev_data.float_z;
                data.rvec_x = alpha * current_rvec[0] + (1.0f - alpha) * prev_data.rvec_x;
                data.rvec_y = alpha * current_rvec[1] + (1.0f - alpha) * prev_data.rvec_y;
                data.rvec_z = alpha * current_rvec[2] + (1.0f - alpha) * prev_data.rvec_z;
            } 
	    else {
                // First time seeing this marker, initialize directly
                data.float_x = current_tvec[0];
                data.float_y = current_tvec[1];
                data.float_z = current_tvec[2];
                data.rvec_x = current_rvec[0];
                data.rvec_y = current_rvec[1];
                data.rvec_z = current_rvec[2];
            }

            data.id = current_id;
            data.pixel_x = static_cast<uint16_t>(avg_x);
            data.pixel_y = static_cast<uint16_t>(avg_y);
            data.update_time = currentTime;
            data.access_time = currentTime;
            markerDataMap[current_id] = data;
            
	    //Sending Data via websockets
	    //char buffer[64];

	    //snprintf(buffer, sizeof(buffer), "{\"x\": %.2f}", data.float_z);
	    //
	    //send_joystick_data(buffer);

	    //

            Vec3d smoothed_tvec(data.float_x, data.float_y, data.float_z);
            Vec3d smoothed_rvec(data.rvec_x, data.rvec_y, data.rvec_z);

	    cv::Mat rotation_matrix;
	    cv::Rodrigues(smoothed_rvec,rotation_matrix);
	    // 2. Decompose the rotation matrix to get Euler angles
	    float sy = std::sqrt(rotation_matrix.at<double>(0,0) * rotation_matrix.at<double>(0,0) +  rotation_matrix.at<double>(1,0) * rotation_matrix.at<double>(1,0));
	    bool singular = sy < 1e-6; // Check for singularity (gimbal lock)
 	    float roll,pitch,yaw;

	    if(!singular) {
		    roll  = std::atan2(rotation_matrix.at<double>(2,1) , rotation_matrix.at<double>(2,2));
    		    pitch = std::atan2(-rotation_matrix.at<double>(2,0), sy);
    		    yaw   = std::atan2(rotation_matrix.at<double>(1,0), rotation_matrix.at<double>(0,0));
	    }

	    else {
		    roll  = std::atan2(-rotation_matrix.at<double>(1,2), rotation_matrix.at<double>(1,1));
    		    pitch = std::atan2(-rotation_matrix.at<double>(2,0), sy);
    	            yaw   = 0;
	    }

	    float roll_deg = roll * 180.0 / CV_PI;
	    float pitch_deg = pitch * 180.0 / CV_PI;
	    float yaw_deg = yaw * 180.0 / CV_PI;


	    // Convert Point2f to Point for polylines to prevent runtime crash
            vector<Point> corners_int;
            corners_int.reserve(markerCorners[i].size());
            for (const auto& corner : markerCorners[i]) {
                corners_int.push_back(Point((int)corner.x, (int)corner.y));
            }
            polylines(frame, corners_int, true, Scalar(0, 255, 255), 4, LINE_AA);

            // Draw axes and text using the smoothed values
            cv::drawFrameAxes(frame, matrixCoefficients, distortionCoefficients, smoothed_rvec, smoothed_tvec, 5);

            //Point text_origin = corners_int[0];
            Point text_origin = {20,30};
	    putText(frame, "ID: " + to_string(data.id), text_origin, FONT_HERSHEY_PLAIN, 1.3, Scalar(255, 255, 255), 2, LINE_AA);
            
            string transText = format("T: x=%.1f y=%.1f z=%.1f", data.float_x, data.float_y, data.float_z);
            putText(frame, transText, Point(text_origin.x, text_origin.y +25), FONT_HERSHEY_PLAIN, 1.3, Scalar(0, 255, 0), 2, LINE_AA);

	 
            float angle_x = data.rvec_x * 180.0 / CV_PI;
            float angle_y = data.rvec_y * 180.0 / CV_PI;
            float angle_z = data.rvec_z * 180.0 / CV_PI;
            
	    string rotText = format("R: x=%.1f y=%.1f z=%.1f", angle_x, angle_y, angle_z);
            putText(frame, rotText, Point(text_origin.x, text_origin.y +50), FONT_HERSHEY_PLAIN, 1.3, Scalar(0, 255, 255), 2, LINE_AA);

	    // Display Roll, Pitch, and Yaw
	    string rpyText = format("RPY: r=%.1f p=%.1f y=%.1f", roll_deg, pitch_deg, yaw_deg);
	    putText(frame, rpyText, Point(text_origin.x, text_origin.y +75), FONT_HERSHEY_PLAIN, 1.3, Scalar(255, 0, 255), 2, LINE_AA);
	}
	//cv::imshow("Live Camera Feed", frame);

    }
   // cv::imshow("Live Camera Feed", frame);


    // Remove markers that haven't been seen for a while
    auto it = markerDataMap.begin();
    while (it != markerDataMap.end()) {
        if (currentTime - it->second.update_time > 1000) { // 1 second timeout
            it = markerDataMap.erase(it);
        } else {
            ++it;
        }
    }
}

int main(int argc, char* argv[]) 
{

    thread ws_thread(Init_WSSocket);
    ws_thread.detach(); //detatch the thread to run independently
    cout << "Websocket server running is currently running in the background" <<"\n";    

    //Initialize teraGrid
    teraGridInit(&teraGrid);

    string calibrationFilePath, arucoTypeStr = "DICT_ARUCO_ORIGINAL";
    //int cameraId = 0; 

    /*if (argc < 3) {
        cerr << "Usage: " << argv[0] << " -c <calibration_file.yml> -i <camera_id> [-t <aruco_type>]" << endl;
        cerr << "Example: " << argv[0] << " -c calib.yml -i 0 -t DICT_4X4_50" << endl;
        return 1;
    }*/

    if(argc<2) {
	    cerr<<"Usage: " << argv[0] << "-t <aruco_type>"<<"\n";
	    return 1;
    }

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        //The commented section was made to work with webcam/kurokesu cam.
	//We do not require it in case of a zed cam as we are not passing any calibration file
	//if (arg == "-c" && i + 1 < argc) {
        //  calibrationFilePath = argv[++i];
        //} 
	//else if (arg == "-i" && i + 1 < argc) {
         //    try {
           //     cameraId = stoi(argv[++i]);
           // } catch (const std::exception& e) {
           //     cerr << "Error: Invalid camera ID provided with -i flag." << endl;
           //     return 1;
               //}
         //} 
         if (arg == "-t" && i + 1 < argc) {
            arucoTypeStr = argv[++i];
        }
    }

    /*if (calibrationFilePath.empty()) {
        cerr << "Error: Missing required argument for calibration file (-c)" << endl;
        return 1;
    }*/

    if (ARUCO_DICT.find(arucoTypeStr) == ARUCO_DICT.end()) {
        cerr << "Error: ArUco tag type '" << arucoTypeStr << "' is not supported" << endl;
        return 1;
    }

    // The dictionary type is simply 'int'
    int arucoDictType = ARUCO_DICT[arucoTypeStr];
    cout << "Using ArUco dictionary: " << arucoTypeStr << endl;

    //Mat k, d;
    /*try {
        loadCalibration(calibrationFilePath, k, d);
        cout << "Calibration files loaded successfully." << endl;
    } catch (const runtime_error& e) {
        cerr << e.what() << endl;
        return 1;
    }*/

    /*cv::Mat k = (cv::Mat_<double>(3,3)<<
		    525.01220703125,0,655.9865112304688,
		    0,525.01220703125,375.3223571777344,
		    0,0,1); */


      /*cv::Mat k = (cv::Mat_<double>(3,3)<<
                    1059.7500,0,1126.0800,
                    0,1059.5200,643.7720,
                    0,0,1);
      */

    sl::Camera zed;

    
    //cv::Mat d = cv::Mat::zeros(1,5,CV_64F);

    std::cout << "Calibration parameters loaded." << std::endl;


    /*VideoCapture video;
    video.open(cameraId);
    if (!video.isOpened()) {
        cerr << "Error: Cannot open camera with ID " << cameraId << endl;
        return 1;
    }*/

    
    sl::InitParameters init_params;
    init_params.camera_resolution =sl::RESOLUTION::HD720;
    
    
    sl::ERROR_CODE err = zed.open(init_params);
    if(err !=sl::ERROR_CODE::SUCCESS) {
	    return 1;
    }
    
    cv::Mat k;
    cv::Mat d;   
    getZedCalibration(zed,k,d);

   
    sl::Mat image_zed;
    char key=' ';
    while(key!='q') {
	    if(zed.grab() == sl::ERROR_CODE::SUCCESS) {
		    zed.retrieveImage(image_zed,sl::VIEW::LEFT); //Get the left image
		    //auto timestamp = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE); //Image Timestamp
		    cv::Mat frame = slMat2cvMat(image_zed);
		    if(!frame.empty()) {
		    	//poseEstimation(frame,arucoDictType,k,d);
			poseEstimationV2(frame,arucoDictType,k,d);
			cv::imshow("Estimated Pose",frame);
			//if(allDetected) return 0;
		    }
            }
	    key = cv::waitKey(1);
    }
    zed.close();

    /*Mat frame;
    while (video.read(frame)) {
        if (frame.empty()){
            cerr << "Error: Captured empty frame" << endl;
            break;
        }
        poseEstimation(frame, arucoDictType, k, d);
        imshow("Estimated Pose", frame);
	*/

        /*char key = (char)waitKey(1);
        if (key == 'q' || key == 27) { // Quit on 'q' or ESC
         
	    break;
        }*/
    //}

    //video.release();
    //destroyAllWindows();
    return 0;
}
