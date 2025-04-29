
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <set> 
#include <numeric>
#include "object_detection.h"

static int read_data_from_file(const char *path, char **out_data)
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

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1){
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float> &outputLocations, std::vector<int> classIds, std::vector<int> &order,
               int filterId, float threshold)
{
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
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

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

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) {
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

static void compute_dfl(float* tensor, int dfl_len, float* box){
    for (int b=0; b<4; b++){
        float exp_t[dfl_len];
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

static int process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes, 
                      std::vector<float> &objProbs, 
                      std::vector<int> &classId, 
                      float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            if (max_score> score_thres_i8){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}

static int process_fp32(float *box_tensor, float *score_tensor, float *score_sum_tensor, 
                        int grid_h, int grid_w, int stride, int dfl_len,
                        std::vector<float> &boxes, 
                        std::vector<float> &objProbs, 
                        std::vector<int> &classId, 
                        float threshold)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    for (int i = 0; i < grid_h; i++)
    {
        for (int j = 0; j < grid_w; j++)
        {
            int offset = i* grid_w + j;
            int max_class_id = -1;

            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < threshold){
                    continue;
                }
            }

            float max_score = 0;
            for (int c= 0; c< OBJ_CLASS_NUM; c++){
                if ((score_tensor[offset] > threshold) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            if (max_score> threshold){
                offset = i* grid_w + j;
                float box[4];
                float before_dfl[dfl_len*4];
                for (int k=0; k< dfl_len*4; k++){
                    before_dfl[k] = box_tensor[offset];
                    offset += grid_len;
                }
                compute_dfl(before_dfl, dfl_len, box);

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5)*stride;
                y1 = (-box[1] + i + 0.5)*stride;
                x2 = (box[2] + j + 0.5)*stride;
                y2 = (box[3] + i + 0.5)*stride;
                w = x2 - x1;
                h = y2 - y1;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(w);
                boxes.push_back(h);

                objProbs.push_back(max_score);
                classId.push_back(max_class_id);
                validCount ++;
            }
        }
    }
    return validCount;
}


rknn_output* getRKNNOutput(bool isQuant, size_t n_output, rknn_tensor_attr* output_attrs) {
    rknn_output* outputs = (rknn_output*)malloc(n_output * sizeof(rknn_output));
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = !isQuant;
        outputs[i].is_prealloc = true;
        outputs[i].buf = malloc(output_attrs[i].size << (isQuant?0:1));
        outputs[i].size = output_attrs[i].size << (isQuant?0:1);
    }
    
    return outputs;
}

void freeRKNNOutput(rknn_output* rknnOutput) {
    for(int i=0; i<9; ++i) free(rknnOutput[i].buf);
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

ObjectDetection::ObjectDetection(const std::string& modelPath_):modelPath(modelPath_), 
                        modelSize(640, 640), ctx(0), inferT(3), postT(3), streamT(1), syncedQ(0, 50),
                        skeleton{
                            {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12},
                            {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
                            {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2},
                            {1, 3}, {2, 4}, {3, 5}, {4, 6}},
                        labels{
                            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
                            "scissors", "teddy bear", "hair drier", "toothbrush"
                        } {
    ;
}

cv::Mat ObjectDetection::preprocess(const cv::Mat& image) {
    cv::Mat img;
    cv::cvtColor(image, img, cv::COLOR_BGR2RGB);
    cv::Scalar background_color(56, 56, 56);
    cv::resize(img, img, rSize, 0, 0, cv::INTER_CUBIC);
    cv::Mat squareImg = expand2square(img, background_color);

    return squareImg;
}

void ObjectDetection::inference(cv::Mat& frame, cv::Mat& image) {
    int ret;
    rknn_input inputs[io_num.n_input];
    rknn_output* outputs = getRKNNOutput(isQuant, io_num.n_output, output_attrs);
    
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

void ObjectDetection::postprocess(cv::Mat& frame, rknn_output* outputs, int iter) {
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int stride = 0;
    int grid_h = 0;
    int grid_w = 0;
    int index = 0;
    float conf_threshold = 0.4;
    
    int dfl_len = output_attrs[0].dims[1] /4;
    int output_per_branch = io_num.n_output / 3;
    
    for (int i = 0; i < 3; i++) {
        void *score_sum = nullptr;
        int32_t score_sum_zp = 0;
        float score_sum_scale = 1.0;
        if (output_per_branch == 3){
            score_sum = outputs[i*output_per_branch + 2].buf;
            score_sum_zp = output_attrs[i*output_per_branch + 2].zp;
            score_sum_scale = output_attrs[i*output_per_branch + 2].scale;
        }
        int box_idx = i*output_per_branch;
        int score_idx = i*output_per_branch + 1;
        grid_h = output_attrs[box_idx].dims[2];
        grid_w = output_attrs[box_idx].dims[3];
        stride = modelHeight / grid_h;
        if (isQuant) {
            validCount += process_i8((int8_t *)outputs[box_idx].buf, output_attrs[box_idx].zp, output_attrs[box_idx].scale,
                                     (int8_t *)outputs[score_idx].buf, output_attrs[score_idx].zp, output_attrs[score_idx].scale,
                                     (int8_t *)score_sum, score_sum_zp, score_sum_scale,
                                     grid_h, grid_w, stride, dfl_len, 
                                     filterBoxes, objProbs, classId, conf_threshold);
        } else {
            validCount += process_fp32((float *)outputs[box_idx].buf, (float *)outputs[score_idx].buf, (float *)score_sum,
                                       grid_h, grid_w, stride, dfl_len, 
                                       filterBoxes, objProbs, classId, conf_threshold);
        }

        
    }
    
    if (validCount > 0) {
    
        std::vector<int> indexArray(validCount);
        std::iota(indexArray.begin(), indexArray.end(), 0);

        quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);
        
        std::set<int> class_set(std::begin(classId), std::end(classId));
        for (auto c : class_set) {
            nms(validCount, filterBoxes, classId, indexArray, c, NMS_THRESH);
        }
        
        for(int i=0; i<validCount; ++i) {
            if(indexArray[i]  == -1) {
                continue;
            }
            int n = indexArray[i];
            float x1 = (filterBoxes[n * 4 + 0] - xPad) / aspectRatio;
            float y1 = (filterBoxes[n * 4 + 1] - yPad) / aspectRatio;
            float w = (filterBoxes[n * 4 + 2]) / aspectRatio;
            float h = (filterBoxes[n * 4 + 3]) / aspectRatio;
            int id = classId[n];
            float objConf = objProbs[n];
            
            cv::rectangle(frame, cv::Rect(cvRound(x1), cvRound(y1), cvRound(w), cvRound(h)), cv::Scalar(0,255,0), 2);
            std::string label = labels[id] + " " + cv::format("%.2f", objConf * 100) + "%";

            int baseline = 0;
            
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            int top = std::max(int(y1), labelSize.height + 10);
            cv::putText(frame, label, cv::Point(x1, top - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            
        }

    }

    auto item = std::make_shared<finalFrames>();
    item->frame = frame;
    item->iter = iter;
    syncedQ.push(*item);

    freeRKNNOutput(outputs);

    return;
}

void ObjectDetection::inferenceWorker(int workerId) {
    int ret;
    int model_len = 0;
    char* model;
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

    if (output_attrs_[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs_[0].type == RKNN_TENSOR_INT8) {
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
        inputAttrSet = true;
        inputAttrCv.notify_all();
    } else {
        std::unique_lock<std::mutex> lck(inputAttrMtx);
        inputAttrCv.wait(lck, [this](){return inputAttrSet.load();});
        lck.unlock();
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
    
    /*
    if (input_attrs != NULL)
    {
        free(input_attrs);
        input_attrs = NULL;
    }
    if (output_attrs != NULL)
    {
        free(output_attrs);
        output_attrs = NULL;
    }
    */
    
    if (ctxI != 0)
    {
        rknn_destroy(ctxI);
        ctxI = 0;
    }
}

void ObjectDetection::postprocessWorker(int workerId) {
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

void ObjectDetection::streamer() {
    std::unique_lock<std::mutex> streamLock(streamMtx, std::defer_lock);
    int totFrames = 0, frameNum = 0;
    std::chrono::steady_clock::time_point startTime, endTime;
    std::chrono::duration<double, std::milli> elapsed;
    {
        std::unique_lock<std::mutex> lck(inputAttrMtx);
	inputAttrCv.wait(lck, [this](){return inputAttrSet.load();});
	lck.unlock();
    }
    streamT.start(0);
    startTime = std::chrono::steady_clock::now();
    while(!stopStream) {
        cv::Mat frame;
        if(!syncedQ.pop(frame)) {
            continue;
        }
	    ++frameNum;
        ++totFrames;
        if(frameNum == 20) {
            endTime = std::chrono::steady_clock::now();
            elapsed = endTime-startTime;
            startTime = endTime;
            frameNum = 0;
            std::cout << "\nFPS: " << 20000.0/elapsed.count() << std::endl;
        }
        cv::imshow("processed stream", frame);
        cv::waitKey(1);
        frame.release();
    }
    streamT.stop(0);
    std::cout << "Net FPS over " << totFrames << ": " << totFrames*1000.0/streamT.getTotalMilliseconds() << std::endl;
}

void ObjectDetection::run(const std::string& streamPath) {
    int streamPathInt = -100000;
    try {
        streamPathInt = std::stoi(streamPath);
    } catch (const std::out_of_range& e) {
        std::cout << "Integer out of range: " << streamPath << std::endl;
    } catch (...) {
        ;;
    }
    std::vector<std::thread> inferenceThreads, postprocessThreads;
    for(int i=0; i<numInferenceWorkers; ++i) {
        inferenceThreads.emplace_back(&ObjectDetection::inferenceWorker, this, i);        
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
        postprocessThreads.emplace_back(&ObjectDetection::postprocessWorker, this, i);
    }
    std::thread streamThread(&ObjectDetection::streamer, this);

    std::unique_lock<std::mutex> inferLock(inferMtx, std::defer_lock);
    size_t mxQ = 18, curFrame = 0;

    {
        std::unique_lock<std::mutex> lck(inputAttrMtx);
        inputAttrCv.wait(lck, [this](){return inputAttrSet.load();});
        lck.unlock();
    }
    for(int it=0; it<3000; ++it) {
        cv::Mat frame, processedFrame;
        
        cap >> frame;

        if (frame.empty()) {
            std::cout << "Stream ended or frame empty." << std::endl;
            break;
        }
        if(autoRefresh && inferQ.size() >= mxQ) {
            inferLock.lock();
            // for(int i=0; i<numRemoval; ++i)
            inferQ.pop();
            inferLock.unlock();
        }
        
        processedFrame = preprocess(frame);

        auto inferData = std::make_shared<inferStruct>();
        inferData->frame = frame;
        inferData->processedFrame = processedFrame;
        inferData->outputs = getRKNNOutput(isQuant, io_num.n_output, output_attrs);
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
    std::cout << "Total inference time: " << inferT.getTotalMilliseconds() << " ms\n";
    std::cout << "Average inference time: " << inferT.getAverageMilliseconds() << " ms\n";

    for(int i=0; i<numPostprocessWorkers; ++i) {
        postprocessThreads[i].join();
    }

    std::cout << "Total postprocess time: " << postT.getTotalMilliseconds() << " ms\n";
    std::cout << "Average postprocess time: " << postT.getAverageMilliseconds() << " ms\n";

    streamThread.join();
}
