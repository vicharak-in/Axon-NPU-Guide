
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>

#include "pose_estimation.h"

int skeleton[38] ={16, 14, 14, 12, 17, 15, 15, 13, 12, 13, 6, 12, 7, 13, 6, 7, 6, 8, 
            7, 9, 8, 10, 9, 11, 2, 3, 1, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7}; 


int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <video_path>\n", argv[0]);
        return -1;
    }

    const char *modelPath = argv[1];
    const char *vidPath = argv[2];


    PoseEstimation model(modelPath);
    
    // use model.run(vidPath, true) for live stream to auto refresh the lag if occurs
    model.run(vidPath);
        
    return 0;
}
