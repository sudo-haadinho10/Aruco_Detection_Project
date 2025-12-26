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
//#include "poseestimation.hpp"

#include "teraGrid.h"


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

void poseEstimationV2(cv::Mat& frame,int arucoDictType,cv::Mat& matrixCoefficients,cv::Mat& distortionCoefficients) 
{

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

	uint32_t currentTime = static_cast<uint32_t>(chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count());


	if(!markerCorners.empty()) 
	{
		std::vector<Vec3d> rvecs,tvecs;
		aruco::estimatePoseSingleMarkers(markerCorners,10, matrixCoefficients, distortionCoefficients, rvecs, tvecs);  //10cm, tvecs will be in the same uniitt
		const float alpha = 0.8f;

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

			if (markerDataMap.count(current_id)) 
			{
                		MarkerData& prev_data = markerDataMap.at(current_id);
                		// Apply exponential moving average for smoothing
				data.float_x = alpha * current_tvec[0] + (1.0f - alpha) * prev_data.float_x;
                		data.float_y = alpha * current_tvec[1] + (1.0f - alpha) * prev_data.float_y;
                		data.float_z = alpha * current_tvec[2] + (1.0f - alpha) * prev_data.float_z;
                		data.rvec_x = alpha * current_rvec[0] + (1.0f - alpha) * prev_data.rvec_x;
                		data.rvec_y = alpha * current_rvec[1] + (1.0f - alpha) * prev_data.rvec_y;
                		data.rvec_z = alpha * current_rvec[2] + (1.0f - alpha) * prev_data.rvec_z;
            	       }
            	       else 
		       {
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
	    char buffer[64];

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
    int cameraId = 0; 

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
        if (arg == "-c" && i + 1 < argc) {
            calibrationFilePath = argv[++i];
        } else if (arg == "-i" && i + 1 < argc) {
             try {
                cameraId = stoi(argv[++i]);
            } catch (const std::exception& e) {
                cerr << "Error: Invalid camera ID provided with -i flag." << endl;
                return 1;
            }
        } else if (arg == "-t" && i + 1 < argc) {
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

    cv::Mat k = (cv::Mat_<double>(3,3)<<
		    525.01220703125,0,655.9865112304688,
		    0,525.01220703125,375.3223571777344,
		    0,0,1); 

      /*cv::Mat k = (cv::Mat_<double>(3,3)<<
                    1059.7500,0,1126.0800,
                    0,1059.5200,643.7720,
                    0,0,1);
      */
   
    cv::Mat d = cv::Mat::zeros(1,5,CV_64F);

    std::cout << "Calibration parameters loaded." << std::endl;


    /*VideoCapture video;
    video.open(cameraId);
    if (!video.isOpened()) {
        cerr << "Error: Cannot open camera with ID " << cameraId << endl;
        return 1;
    }*/

    sl::Camera zed;
    sl::InitParameters init_params;
    init_params.camera_resolution =sl::RESOLUTION::HD720;
    
    
    sl::ERROR_CODE err = zed.open(init_params);
    if(err !=sl::ERROR_CODE::SUCCESS) {
	    return 1;
    }
   
    sl::Mat image_zed;
    char key=' ';
    while(key!='q') {
	    if(zed.grab() == sl::ERROR_CODE::SUCCESS) {
		    zed.retrieveImage(image_zed,sl::VIEW::LEFT); //Get the left image
		    auto timestamp = zed.getTimestamp(sl::TIME_REFERENCE::IMAGE); //Image Timestamp
		    cv::Mat frame = slMat2cvMat(image_zed);
		    if(!frame.empty()) {
		    	//poseEstimation(frame,arucoDictType,k,d);
			poseEstimationV2(frame,arucoDictType,k,d);
			cv::imshow("Estimated Pose",frame);
			if(allDetected) return 0;
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
