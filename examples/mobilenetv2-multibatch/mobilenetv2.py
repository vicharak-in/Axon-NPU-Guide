import numpy as np
import cv2
from rknnlite.api import RKNNLite
import time
import argparse
import threading
import queue

import faulthandler
faulthandler.enable()

class Mobilnet:
    def __init__(self, model_path, input_height, input_width):
        self.model_path = model_path
        self.input_height = input_height
        self.input_width = input_width
        self.frames_processed = 0
        self.inference_q = queue.Queue(maxsize=8)
        self.display_q = queue.Queue(maxsize=16)
        self.do_stream = True
        self.check_fps = True
        self.cur_fps = 0.0
        self.prev_t = time.time()
        self.batch_size = 4
        self.rknn = None

        with open("./labels.txt", "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

    def softmax(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def preprocess(self, frame):
        img = cv2.resize(frame, (self.input_width, self.input_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        return img

    def display_worker(self):
        while self.do_stream:
            try:
                frame = self.display_q.get(timeout=0.1)
                cv2.imshow("MobileNet Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.do_stream = False
                    break
            except queue.Empty:
                continue

    def fps_checker(self):
        time.sleep(1)
        while self.check_fps:
            cur_t = time.time()
            frames_processed = self.frames_processed
            elapsed = cur_t - self.prev_t
            if elapsed > 0:
                self.cur_fps = self.cur_fps*0.6 + frames_processed/elapsed*0.4
            self.prev_t = cur_t
            self.frames_processed = 0
            print(f"FPS = {self.cur_fps:.2f}")
            time.sleep(1)

    def init_rknn(self):
        self.rknn = RKNNLite()
        print("Loading RKNN model...")
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            print(f"Failed to load RKNN model, error code: {ret}")
            return False
        
        print("Initializing RKNN runtime...")
        ret = self.rknn.init_runtime()
        if ret != 0:
            print(f"Failed to initialize RKNN runtime, error code: {ret}")
            return False
        
        print("RKNN initialized successfully")
        return True

    def inference_worker(self):
        try:
            while True:
                batch_item = self.inference_q.get()
                if batch_item is None:
                    break

                batch_frames, batch_preprocessed = batch_item
                
                try:
                    batch_input = np.stack(batch_preprocessed, axis=0)
                    batch_input = batch_input.astype(np.uint8)
                    
                    outputs = self.rknn.inference(inputs=[batch_input])
                    if outputs is None or len(outputs) == 0:
                        print("Warning: Empty inference output")
                        continue
                    
                    output = outputs[0]
                    probs = self.softmax(output)
                    preds = np.argmax(probs, axis=1)
                    
                    for frame, pred_id, prob in zip(batch_frames, preds, probs):
                        confidence = prob[pred_id]
                        label_text = self.labels[pred_id].split(',')[0]
                        cv2.putText(frame, label_text, (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        try:
                            self.display_q.put(frame, timeout=0.1)
                            self.frames_processed += 1
                        except queue.Full:
                            self.frames_processed += 1
                    
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
        except Exception as e:
            print(f"Error in inference worker: {e}")
            import traceback
            traceback.print_exc()

    def run(self, input_path, export_type='stream'):
        if not self.init_rknn():
            print("Failed to initialize RKNN")
            return
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Cannot open input stream")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Input video FPS: {fps}")

        fps_thread = threading.Thread(target=self.fps_checker)
        fps_thread.start()

        display_thread = threading.Thread(target=self.display_worker)
        display_thread.start()

        inference_thread = threading.Thread(target=self.inference_worker)
        inference_thread.start()

        batch_frames, batch_preprocessed = [], []
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Input ended")
                    break

                frame_count += 1
                try:
                    preprocessed = self.preprocess(frame)
                    batch_frames.append(frame)
                    batch_preprocessed.append(preprocessed)
                except Exception as e:
                    print(f"Error preprocessing frame {frame_count}: {e}")
                    continue

                if len(batch_frames) == self.batch_size:
                    try:
                        self.inference_q.put((batch_frames, batch_preprocessed), timeout=1.0)
                        batch_frames, batch_preprocessed = [], []
                    except queue.Full:
                        print("Inference queue full, dropping batch")
                        batch_frames, batch_preprocessed = [], []
            
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up...")
            try:
                self.inference_q.put(None, timeout=1.0)
            except:
                pass

            inference_thread.join(timeout=3)
            self.do_stream = False
            display_thread.join(timeout=2)
            self.check_fps = False
            fps_thread.join(timeout=2)

            if self.rknn:
                self.rknn.release()
                print("RKNN released")

            cap.release()
            cv2.destroyAllWindows()
            print("Cleanup complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To take necessary parameters to run model.")

    parser.add_argument(
        "-m", "--model_path",
        type=str,
        required=True,
        help="To declare path of cnn model"
    )

    parser.add_argument(
        "-i", "--input_path",
        type=str,
        required=True,
        help="To declare path of input stream"
    )

    args = parser.parse_args()

    model = Mobilnet(
        model_path=args.model_path,
        input_height=224,
        input_width=224
    )

    model.run(input_path=args.input_path, export_type='stream')
