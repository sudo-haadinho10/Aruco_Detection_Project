import cv2 
import numpy as np

cam_matrix_path = "calib_matrix.npy"
dist_coeffs_path = "distortion_coefficients.npy"

#Output the file

output_yml_path = "calibration.yml"

try:
    camera_matrix = np.load(cam_matrix_path)
    dist_coeffs = np.load(dist_coeffs_path)
    print("Successfully loaded npy files")

except FileNotFoundError as e:
    print(f"Error: Could not find .npy file. {e}")
    exit()

#Write data to the .yml file
fs = cv2.FileStorage(output_yml_path,cv2.FILE_STORAGE_WRITE)
fs.write('camera_matrix' , camera_matrix)
fs.write('dist_coeffs' , dist_coeffs)
fs.release()


print(f"Calibration data successfully converted and saved to {output_yml_path}")


