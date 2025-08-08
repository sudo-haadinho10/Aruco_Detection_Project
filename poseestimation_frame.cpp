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
#include <cmath> // Required for atan2, sqrt
#include <ws.h>
#include "wsServer.h"


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

struct MarkerData {
    int id;
    uint16_t pixel_x, pixel_y;
    float float_x, float_y, float_z;
    float rvec_x, rvec_y, rvec_z;
    uint32_t update_time, access_time;
};

struct MarkerData array[5];

map<int, MarkerData> markerDataMap;

// Global mutex to protect access to the markerDataMap
std::mutex markerDataMutex;

void loadCalibration(const string& filename, Mat& cameraMatrix, Mat& distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) throw runtime_error("Error: Cannot open calibration file: " + filename);
    fs["camera_matrix"] >> cameraMatrix;
    fs["dist_coeffs"] >> distCoeffs;
    fs.release();
    if (cameraMatrix.empty() || distCoeffs.empty()) throw runtime_error("Error: Loaded calibration matrices are empty");
}

// *** NEW HELPER FUNCTIONS (from Python script) ***

// Checks if a matrix is a valid rotation matrix.
bool isRotationMatrix(const Mat& R) {
    Mat Rt;
    transpose(R, Rt);
    Mat shouldBeIdentity = Rt * R;
    Mat I = Mat::eye(3, 3, shouldBeIdentity.type());
    return norm(I, shouldBeIdentity, NORM_L1) < 1e-6;
}

// Calculates rotation matrix to euler angles
Vec3f rotationMatrixToEulerAngles(const Mat& R) {
    assert(isRotationMatrix(R));
    float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;

    float x, y, z;
    if (!singular) {
        x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = atan2(-R.at<double>(2, 0), sy);
        z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return Vec3f(x, y, z);
}
// *** END OF NEW HELPER FUNCTIONS ***


void poseEstimation(Mat& frame, int arucoDictType, Mat& matrixCoefficients, Mat& distortionCoefficients) {
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    aruco::Dictionary markerDict = aruco::getPredefinedDictionary(arucoDictType);
    aruco::DetectorParameters paramMarkers = aruco::DetectorParameters();
    aruco::ArucoDetector detector(markerDict, paramMarkers);

    vector<vector<Point2f>> markerCorners;
    vector<int> markerIDs;
    detector.detectMarkers(gray, markerCorners, markerIDs);

    uint32_t currentTime = static_cast<uint32_t>(chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count());

    if (!markerCorners.empty()) {
        vector<Vec3d> rvecs, tvecs;
        // NOTE: Make sure 9.5 is your marker size in the desired unit (e.g., cm)
        aruco::estimatePoseSingleMarkers(markerCorners, 9.5, matrixCoefficients, distortionCoefficients, rvecs, tvecs);

        const float alpha = 0.1f;

        for (size_t i = 0; i < markerIDs.size(); ++i) {
            int current_id = markerIDs[i];
            Vec3d current_tvec = tvecs[i];
            Vec3d current_rvec = rvecs[i];

            float avg_x = 0, avg_y = 0;
            for(const auto& corner : markerCorners[i]) {
                avg_x += corner.x;
                avg_y += corner.y;
            }
            avg_x /= 4.0f;
            avg_y /= 4.0f;

            MarkerData data;
            if (markerDataMap.count(current_id)) {
                MarkerData& prev_data = markerDataMap.at(current_id);
                data.float_x = alpha * current_tvec[0] + (1.0f - alpha) * prev_data.float_x;
                data.float_y = alpha * current_tvec[1] + (1.0f - alpha) * prev_data.float_y;
                data.float_z = alpha * current_tvec[2] + (1.0f - alpha) * prev_data.float_z;
                data.rvec_x = alpha * current_rvec[0] + (1.0f - alpha) * prev_data.rvec_x;
                data.rvec_y = alpha * current_rvec[1] + (1.0f - alpha) * prev_data.rvec_y;
                data.rvec_z = alpha * current_rvec[2] + (1.0f - alpha) * prev_data.rvec_z;
            } else {
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

	    {//std::lock_guard<std::mutex> lock(markerDataMutex);

            markerDataMap[current_id] = data;
	    
            char buffer[64];
            snprintf(buffer, sizeof(buffer), "{\"z\": %.2f}",data.float_z);
            send_joystick_data(buffer);

	    }
            Vec3d smoothed_tvec(data.float_x, data.float_y, data.float_z);
            Vec3d smoothed_rvec(data.rvec_x, data.rvec_y, data.rvec_z);

            vector<Point> corners_int;
            for (const auto& corner : markerCorners[i]) corners_int.push_back(Point((int)corner.x, (int)corner.y));
            polylines(frame, corners_int, true, Scalar(0, 255, 255), 4, LINE_AA);
            cv::drawFrameAxes(frame, matrixCoefficients, distortionCoefficients, smoothed_rvec, smoothed_tvec, 5);

            // --- *** UPDATED AND NEW DISPLAY LOGIC *** ---

            Point text_origin = corners_int[0];
            putText(frame, "ID: " + to_string(data.id), text_origin, FONT_HERSHEY_PLAIN, 1.3, Scalar(255, 255, 255), 2, LINE_AA);
            
            // Display Marker position relative to camera
            string transText = format("MARKER Pos: x=%.1f y=%.1f z=%.1f", data.float_x, data.float_y, data.float_z);
            putText(frame, transText, Point(text_origin.x, text_origin.y + 25), FONT_HERSHEY_PLAIN, 1.3, Scalar(0, 255, 0), 2, LINE_AA);

            // --- Convert rvec to matrix and calculate all new values ---
            Mat R_ct;
            Rodrigues(smoothed_rvec, R_ct);
            Mat R_tc = R_ct.t();

            Mat R_flip = Mat::eye(3, 3, CV_64F);
            R_flip.at<double>(1, 1) = -1.0;
            R_flip.at<double>(2, 2) = -1.0;
            
            // Calculate Corrected Euler Angles
            Mat R_corrected = R_flip * R_tc; 
            Vec3f euler_angles = rotationMatrixToEulerAngles(R_corrected);
            float roll_deg = euler_angles[0] * 180.0 / CV_PI;
            float pitch_deg = euler_angles[1] * 180.0 / CV_PI;
            float yaw_deg = euler_angles[2] * 180.0 / CV_PI;
            
            // Display Corrected Marker Attitude
            string rotText = format("MARKER Att: r=%.1f p=%.1f y=%.1f", roll_deg, pitch_deg, yaw_deg);
            putText(frame, rotText, Point(text_origin.x, text_origin.y + 50), FONT_HERSHEY_PLAIN, 1.3, Scalar(0, 255, 255), 2, LINE_AA);

            // Calculate and Display Camera Position relative to Marker
            Mat tvec_mat = Mat(smoothed_tvec);
            Mat pos_camera_mat = -R_tc * tvec_mat;
            string camPosText = format("CAMERA Pos: x=%.1f y=%.1f z=%.1f",
                                       pos_camera_mat.at<double>(0),
                                       pos_camera_mat.at<double>(1),
                                       pos_camera_mat.at<double>(2));
            putText(frame, camPosText, Point(text_origin.x, text_origin.y + 75), FONT_HERSHEY_PLAIN, 1.3, Scalar(255, 165, 0), 2, LINE_AA);

        }
    }

    {
	        //std::lock_guard<std::mutex> lock(markerDataMutex);
    		auto it = markerDataMap.begin();
    		while (it != markerDataMap.end()) {
        		if (currentTime - it->second.update_time > 1000) it = markerDataMap.erase(it);
        		else ++it;
		}
    }
}

int main(int argc, char* argv[]) {
    thread ws_thread(Init_WSSocket);
    ws_thread.detach();
    cout << "Websocket server running in the background" << "\n";

    string calibrationFilePath, arucoTypeStr = "DICT_ARUCO_ORIGINAL";
    int cameraId = 0;

    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " -c <calibration_file.yml> -i <camera_id> [-t <aruco_type>]" << endl;
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-c" && i + 1 < argc) calibrationFilePath = argv[++i];
        else if (arg == "-i" && i + 1 < argc) {
            try { cameraId = stoi(argv[++i]); }
            catch (const exception& e) { cerr << "Error: Invalid camera ID." << endl; return 1; }
        } else if (arg == "-t" && i + 1 < argc) arucoTypeStr = argv[++i];
    }

    if (calibrationFilePath.empty()) { cerr << "Error: Missing -c argument." << endl; return 1; }
    if (ARUCO_DICT.find(arucoTypeStr) == ARUCO_DICT.end()) { cerr << "Error: Unsupported ArUco tag type." << endl; return 1; }

    int arucoDictType = ARUCO_DICT[arucoTypeStr];
    cout << "Using ArUco dictionary: " << arucoTypeStr << endl;

    Mat k, d;
    try {
        loadCalibration(calibrationFilePath, k, d);
        cout << "Calibration file loaded." << endl;
    } catch (const runtime_error& e) {
        cerr << e.what() << endl;
        return 1;
    }

    VideoCapture video;
    video.open(cameraId);
    if (!video.isOpened()) {
        cerr << "Error: Cannot open camera with ID " << cameraId << endl;
        return 1;
    }

    Mat frame;
    while (video.read(frame)) {
        if (frame.empty()) { cerr << "Error: Captured empty frame." << endl; break; }
        poseEstimation(frame, arucoDictType, k, d);
        imshow("Estimated Pose", frame);

        char key = (char)waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    video.release();
    destroyAllWindows();
    return 0;
}
