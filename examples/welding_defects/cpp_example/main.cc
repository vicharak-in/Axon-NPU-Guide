#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "defect_detection.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("%s <model_path> <video_path>\n", argv[0]);
        return -1;
    }

    const char *modelPath = argv[1];
    const char *vidPath = argv[2];


    DefectDetection model(modelPath);
    model.run(vidPath);
    
    return 0;
}

