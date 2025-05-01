
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <set> 
#include <numeric>
#include "pose_estimation.h"


int read_data_from_file(const char *path, char **out_data)
{
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *data = (char *)malloc(file_size+1);
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if(file_size != fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if(fp) {
        fclose(fp);
    }
    *out_data = data;
    return file_size;
}

float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1){
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold) {
    for (int i = 0; i < validCount; ++i)
    {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId)
        {
            continue;
        }
        for (int j = i + 1; j < validCount; ++j)
        {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId)
            {
                continue;
            }
            float xmin0 = outputLocations[n * 5 + 0];
            float ymin0 = outputLocations[n * 5 + 1];
            float xmax0 = outputLocations[n * 5 + 0] + outputLocations[n * 5 + 2];
            float ymax0 = outputLocations[n * 5 + 1] + outputLocations[n * 5 + 3];

            float xmin1 = outputLocations[m * 5 + 0];
            float ymin1 = outputLocations[m * 5 + 1];
            float xmax1 = outputLocations[m * 5 + 0] + outputLocations[m * 5 + 2];
            float ymax1 = outputLocations[m * 5 + 1] + outputLocations[m * 5 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold)
            {
                order[j] = -1;
            }
        }
    }
    return 0;
}

static float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

static float unsigmoid(float y) {
    return -1.0 * logf((1.0 / y) - 1.0);
}

inline static int32_t __clip(float val, float min, float max) {
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

uint8_t qnt_f32_to_affine_u8(float f32, int32_t zp, float scale) {
    float dst_val = (f32 / scale) + zp;
    uint8_t res = (uint8_t)__clip(dst_val, 0, 255);
    return res;
}

float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}
float deqnt_affine_u8_to_f32(uint8_t qnt, int32_t zp, float scale) {
    return ((float)qnt - (float)zp) * scale;
}

static void softmax(float *input, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    float sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        sum_exp += expf(input[i] - max_val);
    }

    for (int i = 0; i < size; ++i) {
        input[i] = expf(input[i] - max_val) / sum_exp;
    }
}

int process_i8(int8_t *input, int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int index) {
    int input_loc_len = 64;
    int tensor_len = input_loc_len + OBJ_CLASS_NUM;
    int validCount = 0;

    int8_t thres_i8 = qnt_f32_to_affine(unsigmoid(threshold), zp, scale);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < OBJ_CLASS_NUM; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= thres_i8) { 
                    float box_conf_f32 = sigmoid(deqnt_affine_to_f32(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ],
                                                 zp, scale));
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = deqnt_affine_to_f32(input[i * grid_w * grid_h + h * grid_w + w], zp, scale);
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax(&loc[i * 16], 16);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }
                    xywh_[0]=(w+0.5)-xywh_[0];
                    xywh_[1]=(h+0.5)-xywh_[1];
                    xywh_[2]=(w+0.5)+xywh_[2];
                    xywh_[3]=(h+0.5)+xywh_[3];
                    xywh[0]=((xywh_[0]+xywh_[2])/2)*stride;
                    xywh[1]=((xywh_[1]+xywh_[3])/2)*stride;
                    xywh[2]=(xywh_[2]-xywh_[0])*stride;
                    xywh[3]=(xywh_[3]-xywh_[1])*stride;
                    xywh[0]=xywh[0]-xywh[2]/2;
                    xywh[1]=xywh[1]-xywh[3]/2;
                    boxes.push_back(xywh[0]);
                    boxes.push_back(xywh[1]);
                    boxes.push_back(xywh[2]);
                    boxes.push_back(xywh[3]);
                    boxes.push_back(float(index + (h * grid_w) + w));
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

int process_fp32(float *input, int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, std::vector<float> &boxScores, std::vector<int> &classId, float threshold,
                      int32_t zp, float scale, int index) {
    int input_loc_len = 64;
    int tensor_len = input_loc_len + OBJ_CLASS_NUM;
    int validCount = 0;
    float thres_fp = unsigmoid(threshold);
    for (int h = 0; h < grid_h; h++) {
        for (int w = 0; w < grid_w; w++) {
            for (int a = 0; a < OBJ_CLASS_NUM; a++) {
                if(input[(input_loc_len + a)*grid_w * grid_h + h * grid_w + w ] >= thres_fp) { 
                    float box_conf_f32 = sigmoid(input[(input_loc_len + a) * grid_w * grid_h + h * grid_w + w ]);
                    float loc[input_loc_len];
                    for (int i = 0; i < input_loc_len; ++i) {
                        loc[i] = input[i * grid_w * grid_h + h * grid_w + w];
                    }

                    for (int i = 0; i < input_loc_len / 16; ++i) {
                        softmax(&loc[i * 16], 16);
                    }
                    float xywh_[4] = {0, 0, 0, 0};
                    float xywh[4] = {0, 0, 0, 0};
                    for (int dfl = 0; dfl < 16; ++dfl) {
                        xywh_[0] += loc[dfl] * dfl;
                        xywh_[1] += loc[1 * 16 + dfl] * dfl;
                        xywh_[2] += loc[2 * 16 + dfl] * dfl;
                        xywh_[3] += loc[3 * 16 + dfl] * dfl;
                    }
                    xywh_[0]=(w+0.5)-xywh_[0];
                    xywh_[1]=(h+0.5)-xywh_[1];
                    xywh_[2]=(w+0.5)+xywh_[2];
                    xywh_[3]=(h+0.5)+xywh_[3];
                    xywh[0]=((xywh_[0]+xywh_[2])/2)*stride;
                    xywh[1]=((xywh_[1]+xywh_[3])/2)*stride;
                    xywh[2]=(xywh_[2]-xywh_[0])*stride;
                    xywh[3]=(xywh_[3]-xywh_[1])*stride;
                    xywh[0]=xywh[0]-xywh[2]/2;
                    xywh[1]=xywh[1]-xywh[3]/2;
                    boxes.push_back(xywh[0]);
                    boxes.push_back(xywh[1]);
                    boxes.push_back(xywh[2]);
                    boxes.push_back(xywh[3]);
                    boxes.push_back(float(index + (h * grid_w) + w));
                    boxScores.push_back(box_conf_f32);
                    classId.push_back(a);
                    validCount++;
                }
            }
        }
    }
    return validCount;
}

rknn_output* getRKNNOutput(bool isQuant) {
    rknn_output* outputs = (rknn_output*)malloc(4 * sizeof(rknn_output));
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 4; i++) {
        outputs[i].index = i;
        outputs[i].want_float = !isQuant;
        outputs[i].is_prealloc = true;
    }
    outputs[0].buf = (float*)malloc(1664000);
    outputs[0].size = 1664000;
    outputs[1].buf = (float*)malloc(416000);
    outputs[1].size = 416000;
    outputs[2].buf = (float*)malloc(104000);
    outputs[2].size = 104000;
    outputs[3].buf = (float*)malloc(1713600);
    outputs[3].size = 1713600;

    return outputs;
}

void freeRKNNOutput(rknn_output* rknnOutput) {
    for(int i=0; i<4; ++i) free(rknnOutput[i].buf);
    free(rknnOutput);
    rknnOutput = nullptr;
    return;
}

int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = input[left];
        while (low < high) {
            while (low < high && input[high] <= key) {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key) {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& background_color) {
    int width = img.cols;
    int height = img.rows;

    if (width == height) {
        return img.clone();
    }

    int size = std::max(width, height);
    cv::Mat result(size, size, img.type(), background_color);

    int x_offset = (size - width) / 2;
    int y_offset = (size - height) / 2;

    cv::Rect roi(x_offset, y_offset, width, height);
    img.copyTo(result(roi));

    return result;
}

PoseEstimation::PoseEstimation(const std::string& modelPath_):modelPath(modelPath_), 
                        modelSize(640, 640), ctx(0), inferT(3), postT(3), syncedQ(0, 50),
                        skeleton{
                            {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
                            {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
                            {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2},
                            {1, 3}, {2, 4}, {3, 5}, {4, 6}} {
    colours.push_back(cv::Scalar(255, 0, 0));
    colours.push_back(cv::Scalar(0, 255, 0));
    colours.push_back(cv::Scalar(0, 0, 255));
    colours.push_back(cv::Scalar(255, 51, 255));
    colours.push_back(cv::Scalar(255, 128, 13));
    colours.push_back(cv::Scalar(153, 255, 153));
}

cv::Mat PoseEstimation::preprocess(const cv::Mat& image) {
    cv::Mat img;
    cv::cvtColor(image, img, cv::COLOR_BGR2RGB);
    cv::Scalar background_color(56, 56, 56);
    cv::resize(img, img, rSize, 0, 0, cv::INTER_CUBIC);
    cv::Mat squareImg = expand2square(img, background_color);

    return squareImg;
}

void PoseEstimation::inference(cv::Mat& frame, cv::Mat& image) {
    int ret;
    rknn_input inputs[io_num.n_input];
    rknn_output* outputs = getRKNNOutput(isQuant);
    
    memset(inputs, 0, sizeof(inputs));

    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = modelWidth * modelHeight * modelChannel;
    inputs[0].buf   = image.data;

    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) {
        std::cerr << "rknn_input_set fail! ret=" << ret << std::endl;
        return;
    }

    ret = rknn_run(ctx, nullptr);

    if (ret < 0){
        std::cerr << "rknn_run fail! ret=" << ret << std::endl;
        return;
    }

    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    if (ret < 0) {
        std::cerr << "rknn_outputs_get fail! ret=" << ret << std::endl;
        return;
    }
    rknn_outputs_release(ctx, io_num.n_output, outputs);

    postprocess(frame, outputs);
}

void PoseEstimation::postprocess(cv::Mat& frame, rknn_output* outputs, int iter) {
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int index = 0;
    float conf_threshold = 0.4;

    for (int i = 0; i < 3; i++) {
        grid_h = output_attrs[i].dims[2];
        grid_w = output_attrs[i].dims[3];
        stride = modelHeight / grid_h;
        if (isQuant) {
            validCount += process_i8((int8_t *)outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                                     classId, conf_threshold, output_attrs[i].zp, output_attrs[i].scale,index);
        }
        else
        {
            validCount += process_fp32((float *)outputs[i].buf, grid_h, grid_w, stride, filterBoxes, objProbs,
                                     classId, conf_threshold, output_attrs[i].zp, output_attrs[i].scale, index);
        }
        index += grid_h * grid_w;
    }
    
    if (validCount > 0) {
    
        std::vector<int> indexArray(validCount);
        std::iota(indexArray.begin(), indexArray.end(), 0);

        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
        
        std::set<int> class_set(std::begin(classId), std::end(classId));
        for (auto c : class_set) {
            nms(validCount, filterBoxes, classId, indexArray, c, 0.4);
        }
        
        cv::Point kpts[17];
        for(int i=0; i<validCount; ++i) {
            if(indexArray[i]  == -1) {
                continue;
            }
            int n = indexArray[i];
            float x1 = (filterBoxes[n * 5 + 0] - xPad) / aspectRatio;
            float y1 = (filterBoxes[n * 5 + 1] - yPad) / aspectRatio;
            float w = (filterBoxes[n * 5 + 2]) / aspectRatio;
            float h = (filterBoxes[n * 5 + 3]) / aspectRatio;
            int kpIndex = (int)filterBoxes[n * 5 + 4];
            cv::rectangle(frame, cv::Rect(cvRound(x1), cvRound(y1), cvRound(w), cvRound(h)), cv::Scalar(0,255,0), 2);

            for(int j=0; j<17; ++j) {
                x1 = (((float*)outputs[3].buf)[j*3*8400 + kpIndex] - xPad) / aspectRatio;
                y1 = (((float*)outputs[3].buf)[j*3*8400 + 8400 + kpIndex] - yPad) / aspectRatio;
                kpts[j] = cv::Point(cvRound(x1), cvRound(y1));
                cv::circle(frame, kpts[j], 3, colours[j<5?0:(j<11?1:2)], -1);
            }
            for(int j=0; j<skeleton.size(); ++j) {
                cv::line(frame, kpts[skeleton[j].first], kpts[skeleton[j].second], colours[j<4?2:(j<7?3:(j<12?1:0))], 2);
            }
        }

    }

    freeRKNNOutput(outputs);

    auto item = std::make_shared<finalFrames>();
    item->frame = frame;
    item->iter = iter;
    
    syncedQ.push(*item);
    return;
}

void PoseEstimation::inferenceWorker(int workerId) {
    int ret;
    int model_len = 0;
    char* model;
    int is_crypt = 0;
    rknn_context ctxI;

    model_len = read_data_from_file(modelPath.c_str(), &model);

    if (model == NULL) {
        std::cerr << "load_model fail!" << std::endl;
        return;
    }

    ret = rknn_init(&ctxI, model, model_len, 0, NULL);

    free(model);

    if (ret < 0) {
        std::cerr << "rknn_init fail! ret=" << ret << std::endl;
        return;
    }

    ret = rknn_query(ctxI, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_query fail! ret=" << ret << std::endl;
        return;
    }

    rknn_tensor_attr input_attrs_[io_num.n_input];
    memset(input_attrs_, 0, sizeof(input_attrs_));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs_[i].index = i;
        ret = rknn_query(ctxI, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return;
        }
    }
    
    rknn_tensor_attr output_attrs_[io_num.n_output];
    memset(output_attrs_, 0, sizeof(output_attrs_));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs_[i].index = i;
        ret = rknn_query(ctxI, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return;
        }
    }

    if (output_attrs_[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs_[0].type != RKNN_TENSOR_FLOAT16) {
        isQuant = true;
    }
    else {
        isQuant = false;
    }
    // rknn_set_core_mask(ctxI, RKNN_NPU_CORE_AUTO);
    if(workerId == 0) {
        this->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
        memcpy(this->input_attrs, input_attrs_, io_num.n_input * sizeof(rknn_tensor_attr));
        this->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
        memcpy(this->output_attrs, output_attrs_, io_num.n_output * sizeof(rknn_tensor_attr));
    }

    rknn_input inputs[io_num.n_input];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].size  = modelWidth * modelHeight * modelChannel;

    std::unique_lock<std::mutex> lockInfer(inferMtx, std::defer_lock), lockPost(postMtx, std::defer_lock);
    
    while(!stopFlag || !inferQ.empty()) {
        lockInfer.lock();
        inferCv.wait(lockInfer, [this](){return !inferQ.empty() || stopFlag;});
        
        if(inferQ.empty()) {
            lockInfer.unlock();
            break;
        }
        auto item = inferQ.front();
        inferQ.pop();
        lockInfer.unlock();
        readCv.notify_one();
        
        inferT.start(workerId);
        int ret;
        
        rknn_output* outputs = item->outputs;

        inputs[0].buf = item->processedFrame.data;

        ret = rknn_inputs_set(ctxI, io_num.n_input, inputs);
        
        if (ret < 0) {
            std::cerr << "rknn_input_set fail! ret=" << ret << std::endl;
            return;
        }

        ret = rknn_run(ctxI, nullptr);

        if (ret < 0){
            std::cerr << "rknn_run fail! ret=" << ret << std::endl;
            return;
        }

        ret = rknn_outputs_get(ctxI, io_num.n_output, outputs, NULL);
        if (ret < 0) {
            std::cerr << "rknn_outputs_get fail! ret=" << ret << std::endl;
            return;
        }
        rknn_outputs_release(ctxI, io_num.n_output, outputs);
        inputs[0].buf = nullptr;

        item->processedFrame.release();
        
        inferT.stop(workerId);
        auto postItem = std::make_shared<postStruct>();
        postItem->frame = item->frame;
        postItem->outputs = outputs;
        postItem->iter = item->iter;
        item.reset();
        item = nullptr;
        
        lockPost.lock();
        postQ.push(postItem);
        lockPost.unlock();
        postCv.notify_one();
    }
    stopPost = true;
    postCv.notify_all();

    if (ctxI != 0)
    {
        rknn_destroy(ctxI);
        ctxI = 0;
    }
}

void PoseEstimation::postprocessWorker(int workerId) {
    std::unique_lock<std::mutex> postLock(postMtx, std::defer_lock);
    while(!stopPost || !postQ.empty()) {
        postLock.lock();
        postCv.wait(postLock, [this](){return !postQ.empty() || stopPost;});
        if(postQ.empty()) {
            postLock.unlock();
            break;
        }
        auto item = postQ.front();
        postQ.pop();
        postLock.unlock();
        postT.start(workerId);
        postprocess(item->frame, item->outputs, item->iter);
        postT.stop(workerId);
    }
    stopStream = true;
    streamCv.notify_all();
}

void PoseEstimation::streamer() {
    std::unique_lock<std::mutex> streamLock(streamMtx, std::defer_lock);
    int totFrames = 0;
    // open particular sized output window if needed
    // cv::namedWindow("Output Stream", cv::WINDOW_NORMAL);
    // cv::resizeWindow("Output Stream", 1280, 720);  
    std::chrono::steady_clock::time_point startTime, endTime;
    std::chrono::duration<double, std::milli> elapsed;
    startTime = std::chrono::steady_clock::now();
    
    while(!stopStream) {
        cv::Mat frame;
        if(!syncedQ.pop(frame)) {
            continue;
        }
        ++totFrames;
        if(totFrames == 20) {
            endTime = std::chrono::steady_clock::now();
            elapsed = endTime-startTime;
            startTime = endTime;
            totFrames = 0;
            std::cout << "\nFPS: " << 20000.0/elapsed.count() << std::endl;
        }
        cv::imshow("Output Stream", frame);
        cv::waitKey(1);
        frame.release();
    }
}

void PoseEstimation::run(const std::string& streamPath, bool _autoRefresh) {
    int streamPathInt = -100000;
    autoRefresh = _autoRefresh;
    try {
        streamPathInt = std::stoi(streamPath);
    } catch (const std::out_of_range& e) {
        std::cout << "Integer out of range: " << streamPath << std::endl;
    } catch (...) {
        ;;
    }
    std::vector<std::thread> inferenceThreads, postprocessThreads;
    for(int i=0; i<numInferenceWorkers; ++i) {
        inferenceThreads.emplace_back(&PoseEstimation::inferenceWorker, this, i);        
    }

    cv::VideoCapture cap;

    if(streamPathInt != -100000) {
        cap.open(streamPathInt);
    } else {
        cap.open(streamPath);
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream or file." << std::endl;
        return;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    if(width > height) {
        aspectRatio = 640 / (float)width;
        int newHeight = aspectRatio * (float)height;
        rSize.width = 640;
        rSize.height = newHeight;
        xPad = 0;
        yPad = float(640-newHeight)/2;
    } else {
        aspectRatio = 640 / (float)height;
        int newWidth = aspectRatio * (float)width;
        rSize.width = newWidth;
        rSize.height = 640;
        xPad = float(640-newWidth)/2;
        yPad = 0;
    }


    for(int i=0; i<numPostprocessWorkers; ++i) {
        postprocessThreads.emplace_back(&PoseEstimation::postprocessWorker, this, i);
    }
    std::thread streamThread(&PoseEstimation::streamer, this);

    std::unique_lock<std::mutex> inferLock(inferMtx, std::defer_lock);
    size_t mxQ = 18, curFrame = 0;

    for(int it=0; it<3000; ++it) {
        cv::Mat frame, processedFrame;
        
        cap >> frame;

        if (frame.empty()) {
            std::cout << "Stream ended or frame empty." << std::endl;
            break;
        }
        if(inferQ.size() >= mxQ) {
            if(autoRefresh) {
                inferLock.lock();
                // for(int i=0; i<numRemoval; ++i)
                inferQ.pop();
                inferLock.unlock();
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        
        processedFrame = preprocess(frame);

        auto inferData = std::make_shared<inferStruct>();
        inferData->frame = frame;
        inferData->processedFrame = processedFrame;
        inferData->outputs = getRKNNOutput(isQuant);
        inferData->iter = curFrame;
        ++curFrame;

        inferLock.lock();
        readCv.wait(inferLock, [this, mxQ](){return inferQ.size()<mxQ;});
        inferQ.push(inferData);
        
        inferLock.unlock();
        inferCv.notify_one();
    }
    cap.release();

    std::this_thread::sleep_for(std::chrono::seconds(2));
    stopFlag = true;
    inferCv.notify_all();

    for(int i=0; i<numInferenceWorkers; ++i) {
        inferenceThreads[i].join();
    }

    std::cout << "Average inference time per NPU core: " << inferT.getAverageMilliseconds() << " ms\n";

    for(int i=0; i<numPostprocessWorkers; ++i) {
        postprocessThreads[i].join();
    }

    std::cout << "Average postprocess time: " << postT.getAverageMilliseconds() << " ms\n";

    streamThread.join();
}
