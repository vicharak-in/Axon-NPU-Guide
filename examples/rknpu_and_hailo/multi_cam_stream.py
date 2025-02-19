import numpy as np
import cv2
from rknnlite.api import RKNNLite
import queue
import threading
import concurrent.futures
import time
import argparse
from utils import HailoAsyncInference, check_q, FinalFrames, time_it
from object_detection_utils import ObjectDetectionUtils, class_colors


class Yolo11():

    def __init__(self, rknn_model_path, hef_model_path, OBJ_THRESH=0.3, NMS_THRESH=0.45, 
            input_height=640, input_width=640, inference_workers=3, postprocess_workers=3):
        self.model_path = rknn_model_path
        self.OBJ_THRESH = OBJ_THRESH
        self.NMS_THRESH = NMS_THRESH
        self.put = 9
        self.inference_workers = inference_workers
        self.postprocess_workers = postprocess_workers
        self.put_fps = True
        self.inference_time = 0
        self.total_inf = 0
        self.input_height = input_height
        self.input_width = input_width
        self.input_size = (input_height, input_width)
        self.hef_model_path = hef_model_path
        self.utils = ObjectDetectionUtils("coco.txt")
        self.CLASSES = (
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
            "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
            "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
            "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
            "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        )
        self.indices_req = np.arange(8400)
        self.final_frames = None

    def update_classes(self, classes):
        self.CLASSES = classes
    
    
    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_height, self.input_width), 
                         interpolation=cv2.INTER_LINEAR)
        img = np.expand_dims(img, 0)
        return img

    
    def dfl(self, position):
        x = np.array(position)
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        y = np.exp(x.reshape(n, p_num, mc, h, w))
        
        y = y / y.sum(axis=2, keepdims=True)
        
        acc_metrix = np.arange(mc, dtype=float).reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(axis=2)
        
        return y

    
    def box_process_old(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        grid = np.stack((col, row), axis=0)
        stride = np.array([640 // grid_h, 640 // grid_w]).reshape(1, 2, 1, 1)

        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy
    
    
    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]

        col = np.arange(grid_w)
        row = np.arange(grid_h)
        grid = np.stack(np.meshgrid(col, row), axis=0)  # (2, grid_h, grid_w)

        stride = np.array([640 // grid_h, 640 // grid_w]).reshape(1, 2, 1, 1)

        position = self.dfl(position)

        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]

        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy
    
    
    def filter_boxes_old(self, boxes, box_confidences, box_class_probs):
        
        box_confidences = box_confidences.ravel()

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = class_max_score * box_confidences >= self.OBJ_THRESH
        
        scores = (class_max_score[_class_pos]* box_confidences[_class_pos])

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

        return boxes, classes, scores

    
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        classes = np.argmax(box_class_probs, axis=-1)
        class_max_score = box_class_probs[self.indices_req, classes]
        
        combined_scores = class_max_score * box_confidences.ravel()

        valid_pos = combined_scores >= self.OBJ_THRESH

        boxes = boxes[valid_pos]
        classes = classes[valid_pos]
        scores = combined_scores[valid_pos]

        return boxes, classes, scores

    
    def nms_boxes(self, boxes, scores):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.NMS_THRESH)[0]
            order = order[inds + 1]

        return np.array(keep)

    
    def draw(self, image, boxes, scores, classes, ind=0):
        
        for box, score, cl in zip(boxes, scores, classes):

            top, left, right, bottom = box
            top = int(top*self.scale_width)
            left = int(left*self.scale_height)
            right = int(right*self.scale_width)
            bottom = int(bottom*self.scale_height)

            cv2.rectangle(image, (top, left), (right, bottom), class_colors[cl], 1)
            label = '{0} {1:.2f}'.format(self.CLASSES[cl], score)
            cv2.putText(image, label,
                        (top, left),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 255), 1)
            

        if self.put_fps == True:
            if isinstance(self.cur_fps, np.ndarray):
                fps_label = 'FPS: {:.2f}'.format(self.cur_fps[ind%self.total_inputs])
            else:
                fps_label = 'FPS: {:.2f}'.format(self.cur_fps)
            cv2.putText(image, fps_label,
                        (self.width - 120, 30),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), 1)
            
        return image   
              
    
    def postprocess(self, inference_output, frame, ind=0):
        boxes, classes_conf, scores = [], [], []
        default_branch = 3
        pair_per_branch = len(inference_output) // default_branch
        
        for i in range(default_branch):
            boxes.append(self.box_process(inference_output[pair_per_branch * i]))
            classes_conf.append(inference_output[pair_per_branch * i + 1])
            scores.append(np.ones_like(
                inference_output[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

        def sp_flatten(_in):
            return _in.transpose(0, 2, 3, 1).reshape(-1, _in.shape[1])

        boxes = np.concatenate([sp_flatten(b) for b in boxes])
        classes_conf = np.concatenate([sp_flatten(c) for c in classes_conf])
        scores = np.concatenate([sp_flatten(s) for s in scores])

        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            s = scores[inds]
            keep = self.nms_boxes(b, s)
            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(np.full_like(keep, c))
                nscores.append(s[keep])

        if nclasses or nscores:
            boxes = np.concatenate(nboxes)
            classes = np.concatenate(nclasses)
            scores = np.concatenate(nscores)
        
        return self.draw(frame, boxes, scores, classes, ind)
        

    
    def inference(self, model, img):
        return model.inference(inputs=[img])

    def inference_worker2(self, i):
        rknn_lite = RKNNLite()
        rknn_lite.load_rknn(self.model_path)
        ret = rknn_lite.init_runtime(async_mode=True, core_mask=i)
        if(ret != 0):
            print(f'Failed to load RKNNLite with core mask = {i}')
            return

        st = time.time()

        while True:
            item = self.inference_q.get()
            if item is None:
                break

            img, frame, iter = item
            outputs = self.inference(rknn_lite, img)
            self.postprocess_q.put((outputs, frame, iter))
        
        self.inference_time += time.time()-st
        rknn_lite.release()


    def inference_worker(self, i):
        rknn_lite = RKNNLite()
        rknn_lite.load_rknn(self.model_path)
        ret = rknn_lite.init_runtime(async_mode=True, core_mask=i)
        if(ret != 0):
            print(f'Failed to load RKNNLite with core mask = {i}')
            return

        def inp_set():

            item = self.inference_q.get()
            if item is None:
                return None, None
            img, frame, iter = item
            rknn_lite.rknn_runtime.set_inputs([img], None, ['nhwc'], inputs_pass_through=None)
            return frame, iter

        frame, iter = inp_set()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            stt = time.time()
            processed_here = 0
            while True:
                self.total_inf += 1
                rknn_lite.rknn_runtime.run(False)
                future1 = executor.submit(rknn_lite.rknn_runtime.get_outputs, False)
                future2 = executor.submit(inp_set, )
                cur_frame = frame
                cur_iter = iter
                frame, iter = future2.result()
                processed_here += 1
                if(frame is None):
                    outputs = future1.result()
                    self.postprocess_q.put((outputs, cur_frame, cur_iter))
                    self.inference_q.task_done()
                    break    
                outputs = future1.result()
                self.inference_q.task_done()
                self.postprocess_q.put((outputs, cur_frame, cur_iter))

            self.inference_time += time.time()-stt

        rknn_lite.release()

    
    
    def postprocess_hailo(self, outputs, frame, iter):
        detections = self.utils.extract_detections(outputs)

        if len(outputs) == 1:
            outputs = outputs[0]

        detections = self.utils.extract_detections(outputs)
        
        image = self.utils.visualize(
            detections, frame,
            self.width, self.height
        )
        if self.put_fps == True:
            if isinstance(self.cur_fps, np.ndarray):
                fps_label = 'FPS: {:.2f}'.format(self.cur_fps[iter%self.total_inputs])
            else:
                fps_label = 'FPS: {:.2f}'.format(self.cur_fps)
            cv2.putText(image, fps_label,
                        (self.width - 120, 30),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 255, 0), 1)
        return image
    

    def postprocess_worker(self,):
        while True:
            item = self.postprocess_q.get()
            if item is None:
                break
            else:
                outputs, frame, iter = item
                if(check_q(iter)):
                    image = self.postprocess(outputs, frame, iter)
                else:
                    image = self.postprocess_hailo(outputs, frame, iter)

                if self.final_frames is not None:
                    if isinstance(self.final_frames, list):
                        self.final_frames[iter%self.total_inputs].add_frame(iter, image)
                    else:
                        self.final_frames.add_frame(iter, image)

                if hasattr(self, "result") and self.result is not None:
                    self.result.write(image)
                
            self.postprocess_q.task_done()

            if hasattr(self, "frames_processed"):
                self.frames_processed[iter%self.total_inputs] += 1


    def stream_worker(self,):
        time.sleep(0.5)
        
        for i in range(self.total_inputs):
            cv2.namedWindow(f'Axon: cam_{i}', cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f'Axon: cam_{i}', 560, 315)
        while self.do_stream:
            for i in range(self.total_inputs):
                st = time.time()
                frame = self.final_frames[i].get_frame()
                if frame is not None:
                    cv2.imshow(f'Axon: cam_{i}', frame)
                    en = time.time()
                    cv2.waitKey(10 + max(0, int(1000/(self.cur_fps[i]+1) + (st-en)*1000)-10))
                else:
                    time.sleep(0.008)

        
    def process_output(self,):
        self.process_output_time = time.time()
        while True:
            result = self.output_q.get()
            
            if result is None:
                break  
            
            frame, outputs, iter = result
            
            detections = self.utils.extract_detections(outputs)

            if len(outputs) == 1:
                outputs = outputs[0]

            detections = self.utils.extract_detections(outputs)
            
            image = self.utils.visualize(
                detections, frame, 
                self.width, self.height
            )

            if self.put_fps == True:
                if isinstance(self.cur_fps, np.ndarray):
                    fps_label = 'FPS: {:.2f}'.format(self.cur_fps[iter%self.total_inputs])
                else:
                    fps_label = 'FPS: {:.2f}'.format(self.cur_fps)
                cv2.putText(image, fps_label,
                            (self.width - 120, 30),
                            cv2.FONT_HERSHEY_PLAIN,
                            1, (0, 255, 0), 1)
            
            if self.final_frames is not None:
                if isinstance(self.final_frames, list):
                    self.final_frames[iter%self.total_inputs].add_frame(iter, image)
                else:
                    self.final_frames.add_frame(iter, image)

            if hasattr(self, "result") and self.result is not None:
                self.result.write(image)


            if hasattr(self, "frames_processed"):
                self.frames_processed[iter%self.total_inputs] += 1

        self.output_q.task_done() 
   
    def run(self, input_paths, export_type='stream', output_paths=None):
        if not isinstance(input_paths, list):
            input_paths = [input_paths]

        if not isinstance(output_paths, list):
            output_paths = [output_paths]
            
        self.total_inputs = len(input_paths)

        self.final_frames = [FinalFrames(6) for _ in range(self.total_inputs)]

        self.cur_fps = np.full(self.total_inputs, 30)

        self.inference_q = queue.Queue(maxsize=5*self.total_inputs)
        self.inference_q2 = queue.Queue(maxsize=5*self.total_inputs)
        inference_threads = []

        for i in range(self.inference_workers):
            inference_threads.append(threading.Thread(target=self.inference_worker2, args=(1<<i,)))
            inference_threads[i].start()

        
        self.postprocess_q = queue.Queue(maxsize=8*self.total_inputs)

        self.output_q = queue.Queue()

        self.hailo_inference = HailoAsyncInference(
            self.hef_model_path, self.inference_q2, self.postprocess_q, 8
        )

        postprocess_threads = []
        for i in range(self.postprocess_workers):
            postprocess_threads.append(threading.Thread(target=self.postprocess_worker))
            postprocess_threads[i].start()
        

        cap = [cv2.VideoCapture(cam_id) for cam_id in input_paths]
        self.width = int(cap[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale_width = self.width/self.input_width
        self.scale_height = self.height/self.input_height
        
        hailo_thread = threading.Thread(target=self.hailo_inference.run,)

        hailo_thread.start()

        if export_type == 'stream':
            self.do_stream = True
            stream_thread = threading.Thread(target=self.stream_worker) 
            stream_thread.start()   
        
        iter = 0
        
        self.frames_processed = np.zeros(self.total_inputs)

        self.check_fps = True
        
        self.prev_t = time.time()
        np.set_printoptions(precision=2, suppress=True)

        total_frames_processed = 0
        start_t = time.time()
        frame_list = []
        max_read_frame = 24000
        while iter<max_read_frame:
            lapse_t = time.time()
            iter += 1

            if(iter%40 == 0):
                cur_t = time.time()
                frames_processed = self.frames_processed
                self.cur_fps = frames_processed/(cur_t-self.prev_t)*0.6 + self.cur_fps*0.4
                self.frames_processed = np.zeros(self.total_inputs)
                total_frames_processed += frames_processed.sum()
                print(f"FPS = {self.cur_fps} || total FPS = {self.cur_fps.sum():.2f}")
                self.prev_t = cur_t
                
            success, frame = cap[iter%self.total_inputs].read()            
            
            if success:

                if(self.inference_q.full() and export_type=="stream"):
                    for _ in range(self.total_inputs):
                        self.inference_q.get_nowait()
                
                if(self.inference_q2.full() and export_type=="stream"):
                    for _ in range(self.total_inputs):
                        self.inference_q2.get_nowait()
                
                if(check_q(iter)):
                    img = self.preprocess(frame)
                    self.inference_q.put_nowait((img, frame, iter))
                else:
                    self.inference_q2.put_nowait((frame, iter))

            else:
                print("input ended or couldn't get frame from webcam")
                break

                for i in range(self.total_inputs):
                    cap[i].release()
                        
            c_t = time.time()-lapse_t

            time.sleep(max(0, 0.006-c_t))

        for i in range(self.postprocess_workers):
            self.postprocess_q.put(None)
        for i in range(2*self.inference_workers):
            self.inference_q.put(None)
        self.inference_q2.put(None)        
        for inference_thread in inference_threads:
            inference_thread.join()
        
        for postprocess_thread in postprocess_threads:
            postprocess_thread.join()
        
        hailo_thread.join()


        self.do_stream = False
        self.check_fps = False
        stream_thread.join()

        cv2.destroyAllWindows()


    def __del__(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To take necessary parameters to run model.")
    
    parser.add_argument(
        "-M", "--rknn_model_path",
        type = str,
        default = "/home/vicharak/nec/yolo11s.rknn",
        help = "To declare path of rknn model"
    )

    parser.add_argument(
        "-M2", "--hef_model_path",
        type = str,
        default = "/home/vicharak/yolov11s.hef",
        help = "To declare path of hef model"
    )

    parser.add_argument(
        "-I", "--input_path",
        type = str,
        nargs = '+',
        default = "/home/vicharak/edmonton_canada.mp4",
        help = "To declare path of input stream"
    )
    
    parser.add_argument(
        "-O", "--obj_thres",
        type = float,
        default = 0.3,
        help = "minimum confidence threshold"
    )

    parser.add_argument(
        "-N", "--nms_thres",
        type = float,
        default = 0.4,
        help = "nms / iou threshold"
    )

    parser.add_argument(
        "-H", "--height",
        type = int,
        default = 640,
        help = "height of input to model"
    )

    parser.add_argument(
        "-W", "--width",
        type = int,
        default = 640,
        help = "width of input to model"
    )

    args = parser.parse_args()

    model = Yolo11(
        rknn_model_path = args.rknn_model_path,
        hef_model_path = args.hef_model_path,
        OBJ_THRESH = args.obj_thres,
        NMS_THRESH = args.nms_thres,
        input_height = args.height,
        input_width = args.width
    )
    
    model.run(input_paths = args.input_path, export_type = 'stream')
    
