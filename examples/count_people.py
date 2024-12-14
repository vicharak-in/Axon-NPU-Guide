import numpy as np
import cv2
from rknnlite.api import RKNNLite
import queue
import threading
import concurrent.futures
import time
import heapq
import argparse


class FinalFrames:
    def __init__(self, capacity=30):
        self.capacity = capacity
        self.frames = []
        self.last_streamed = -1
        self.lock = threading.Lock()

    def set_capacity(self, new_capacity):
        with self.lock:
            self.capacity = new_capacity
            while len(self.frames) > self.capacity:
                heapq.heappop(self.frames)

    def add_frame(self, iter, frame):
        with self.lock:
            if(iter < self.last_streamed):
                return
            heapq.heappush(self.frames, (iter, frame))
            if len(self.frames) > self.capacity:
                heapq.heappop(self.frames)

    def get_frame(self):
        with self.lock:
            if self.frames:
                iter, frame = heapq.heappop(self.frames)
                self.last_streamed = iter
                return frame
            return None


class CountPeople():

    def __init__(self, model_path, OBJ_THRESH=0.4, NMS_THRESH=0.45, 
            input_height=640, input_width=640, inference_workers=3, postprocess_workers=2):
        self.model_path = model_path
        self.OBJ_THRESH = OBJ_THRESH
        self.NMS_THRESH = NMS_THRESH
        self.put = 9
        self.inference_workers = inference_workers
        self.postprocess_workers = postprocess_workers
        self.inference_time = 0
        self.total_inf = 0
        self.input_height = input_height
        self.input_width = input_width
        self.input_size = (input_height, input_width)
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


    def update_classes(self, classes):
        self.CLASSES = classes
    
    def preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_height, self.input_width), 
                         interpolation=cv2.INTER_CUBIC)
        img = np.expand_dims(img, 0)
        return img

    def dfl(self, position):
        x = np.array(position)
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        y = x.reshape(n, p_num, mc, h, w)
        
        y = np.exp(y) / np.exp(y).sum(axis=2, keepdims=True)
        
        acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
        y = (y * acc_metrix).sum(axis=2)
        
        return y

    def box_process(self, position):
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        grid = np.stack((col, row), axis=0)
        stride = np.array([640 // grid_h, 640 // grid_w]).reshape(1, 2, 1, 1)

        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy
    
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        
        box_confidences = box_confidences.ravel()

        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        _class_pos = class_max_score * box_confidences >= self.OBJ_THRESH
        
        scores = (class_max_score[_class_pos]* box_confidences[_class_pos])

        boxes = boxes[_class_pos]
        classes = classes[_class_pos]

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

        persons = 0
        
        for box, score, cl in zip(boxes, scores, classes):
            if cl != 0:
                continue

            persons += 1
            top, left, right, bottom = box
            top = int(top*self.width/self.input_height)
            left = int(left*self.height/self.input_width)
            right = int(right*self.width/self.input_width)
            bottom = int(bottom*self.height/self.input_height)

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            
        self.put += 1
        if(self.put==10):
            self.put = 0
            self.count_label = f'Total people: {persons}'

        cv2.putText(image, self.count_label,
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        if hasattr(self, "put_fps") and self.put_fps == True:
            if isinstance(self.cur_fps, np.ndarray):
                fps_label = 'FPS: {:.2f}'.format(self.cur_fps[ind%self.total_inputs])
            else:
                fps_label = 'FPS: {:.2f}'.format(self.cur_fps)
            cv2.putText(image, fps_label,
                        (self.width - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
            
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


    def postprocess_worker(self,):
        while True:
            item = self.postprocess_q.get()
            if item is None:
                break
            else:
                outputs, frame, iter = item
                image = self.postprocess(outputs, frame, iter)

                if hasattr(self, "final_frames") and self.final_frames is not None:
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
        while self.do_stream:
            for i in range(self.total_inputs):
                st = time.time()
                frame = self.final_frames[i].get_frame()
                if frame is not None:
                    cv2.imshow(f'Axon: cam_{i}', frame)
                    en = time.time()
                    cv2.waitKey(10 + max(0, int(1000/(self.cur_fps[i]+1) + (st-en)*1000)-10))
                else:
                    time.sleep(0.004)


    def fps_checker(self,):
        time.sleep(1)
        while self.check_fps:
            cur_t = time.time()
            frames_processed = self.frames_processed
            self.cur_fps = self.cur_fps*0.6 + frames_processed/(cur_t-self.prev_t)*0.4
            self.prev_t = cur_t
            self.frames_processed = np.zeros(self.total_inputs)
            print(f"FPS = {self.cur_fps}")
            time.sleep(1)
        
        
    def run(self, input_paths, export_type='stream', output_paths=None):
        if not isinstance(input_paths, list):
            input_paths = [input_paths]

        if not isinstance(output_paths, list):
            output_paths = [output_paths]
            
        self.total_inputs = len(input_paths)

        self.final_frames = [FinalFrames(6) for _ in range(self.total_inputs)]

        self.cur_fps = np.full(self.total_inputs, 30)
        self.put_fps = True

        self.inference_q = queue.Queue(maxsize=3*self.total_inputs)
        inference_threads = []

        for i in range(self.inference_workers):
            inference_threads.append(threading.Thread(target=self.inference_worker, args=(1<<i,)))
            inference_threads[i].start()

        
        self.postprocess_q = queue.Queue(maxsize=3*self.total_inputs)

        postprocess_threads = []
        for i in range(self.postprocess_workers):
            postprocess_threads.append(threading.Thread(target=self.postprocess_worker))
            postprocess_threads[i].start()
        

        cap = [cv2.VideoCapture(cam_id) for cam_id in input_paths]
        self.width = int(cap[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.do_stream = True
        if export_type == 'stream':
            stream_thread = threading.Thread(target=self.stream_worker) 
            stream_thread.start()   
        
        iter = 0
        
        self.frames_processed = np.zeros(self.total_inputs)

        self.check_fps = True
        fps_checker_thread = threading.Thread(target=self.fps_checker)
        fps_checker_thread.start()
        self.prev_t = time.time()

        while True:
            success, frame = cap[iter%self.total_inputs].read()
            
            iter += 1
            
            if success:
                img = self.preprocess(frame)

                if(self.inference_q.full() and export_type=="stream"):
                    for _ in range(self.total_inputs):
                        self.inference_q.get_nowait()

                self.inference_q.put((img, frame, iter))
            else:
                print("input ended or couldn't get frame from webcam")
                for i in range(2*self.inference_workers):
                    self.inference_q.put(None)
                break

                for i in range(self.total_inputs):
                    cap[i].release()

        for i in range(self.postprocess_workers):
            self.postprocess_q.put(None)
        
        for inference_thread in inference_threads:
            inference_thread.join()
        
        for postprocess_thread in postprocess_threads:
            postprocess_thread.join()
        
        self.do_stream = False
        self.check_fps = False
        stream_thread.join()
        fps_checker_thread.join()

        cv2.destroyAllWindows()


    def __del__(self):
        pass


if __name__ == "__main__":
    # Run the program to count people using command: `python3 -M rknn_model_path -I input_path`
    parser = argparse.ArgumentParser(description="To take necessary parameters to run model.")
    
    parser.add_argument(
        "-M", "--model_path",
        type = str,
        required = True,
        help = "To declare path of cnn model"
    )

    parser.add_argument(
        "-I", "--input_path",
        type = str,
        required = True,
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
        default = 0.45,
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

    model = CountPeople(
        model_path = args.model_path,
        OBJ_THRESH = args.obj_thres,
        NMS_THRESH = args.nms_thres,
        input_height = args.height,
        input_width = args.width
    )

    start = time.time()
    model.run(input_paths = args.input_path, export_type = 'stream')
    end = time.time()

    print(f"total time taken = {(end-start)} seconds")
