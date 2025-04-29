#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include "CNN_model.h"

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << argv[0] << " <model_path> <video_path>\n" << std::endl;
        return -1;
    }

    const char *modelPath = argv[1];
    const char *vidPath = argv[2];

    std::string modelPaths[] = {std::string(modelPath)};
    CNNModel model(modelPaths, 1);
    
    return 0;
}
