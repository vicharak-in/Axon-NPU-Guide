# Welding defects detection using yolo11x

### This repo contains program to run welding defect detection on video stream using yolo11x
The model used for this program can be found [here](https://drive.google.com/file/d/1lb_11Mwi-oPx4SRMbpwnJGXS5d7uwQ07/view?usp=sharing).  

To use the program, we need to first compile the model  
(Prerequisites: Cmake & OpenCV)  
Run the script to compile the program: `./compile.sh`

Then the executable file will be created as `build/defect_detection`

Run the executable as `<path_to_executable>/defect_detection <path_to_defect_detection_yolo_model> <path_to_stream_for_openCV>`
