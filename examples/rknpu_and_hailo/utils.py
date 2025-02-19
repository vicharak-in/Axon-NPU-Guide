from typing import List, Optional, Tuple, Dict
from functools import partial
import queue
from loguru import logger
import numpy as np
import time
import threading
import heapq
import cv2
from hailo_platform import (HEF, VDevice,
                            FormatType, HailoSchedulingAlgorithm)


class HailoAsyncInference:
    def __init__(
        self, hef_path: str, input_queue: queue.Queue,
        output_queue: queue.Queue, batch_size: int = 1,
        input_type: Optional[str] = None, output_type: Optional[Dict[str, str]] = None,
        send_original_frame: bool = False) -> None:
        """
        Initialize the HailoAsyncInference class with the provided HEF model 
        file path and input/output queues.

        Args:
            hef_path (str): Path to the HEF model file.
            input_queue (queue.Queue): Queue from which to pull input frames 
                                       for inference.
            output_queue (queue.Queue): Queue to hold the inference results.
            batch_size (int): Batch size for inference. Defaults to 1.
            input_type (Optional[str]): Format type of the input stream. 
                                        Possible values: 'UINT8', 'UINT16'.
            output_type Optional[dict[str, str]] : Format type of the output stream. 
                                         Possible values: 'UINT8', 'UINT16', 'FLOAT32'.
        """
        self.input_queue = input_queue
        self.output_queue = output_queue
        params = VDevice.create_params()    

        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)    
        self.batch_size = batch_size  
        if input_type is not None:
            self._set_input_type(input_type)
        if output_type is not None:
            self._set_output_type(output_type)

        self.output_type = output_type
        self.send_original_frame = send_original_frame

    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        """
        Set the input type for the HEF model. If the model has multiple inputs,
        it will set the same type of all of them.

        Args:
            input_type (Optional[str]): Format type of the input stream.
        """
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))
    
    def _set_output_type(self, output_type_dict: Optional[Dict[str, str]] = None) -> None:
        """
        Set the output type for the HEF model. If the model has multiple outputs,
        it will set the same type for all of them.

        Args:
            output_type_dict (Optional[dict[str, str]]): Format type of the output stream.
        """
        for output_name, output_type in output_type_dict.items():
            self.infer_model.output(output_name).set_format_type(
                getattr(FormatType, output_type)
            )

    def callback(
        self, completion_info, bindings_list: list, input_batch: list, iter_list: list
    ) -> None:
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the 
                             inference task.
            bindings_list (list): List of binding objects containing input 
                                  and output buffers.
            processed_batch (list): The processed batch of images.
        """
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            for i in range(self.batch_size):

                result = bindings_list[i].output().get_buffer()
                self.output_queue.put((result, input_batch[i], iter_list[i]))

    def get_vstream_info(self) -> Tuple[list, list]:

        """
        Get information about input and output stream layers.

        Returns:
            Tuple[list, list]: List of input stream layer information, List of 
                               output stream layer information.
        """
        return (
            self.hef.get_input_vstream_infos(), 
            self.hef.get_output_vstream_infos()
        )

    def get_hef(self) -> HEF:
        """
        Get the object's HEF file
        
        Returns:
            HEF: A HEF (Hailo Executable File) containing the model.
        """
        return self.hef

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the model's input layer.

        Returns:
            Tuple[int, ...]: Shape of the model's input layer.
        """
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input

    def run(self) -> None:
        with self.infer_model.configure() as configured_infer_model:
            keep_running = True
            while keep_running:
                bindings_list = []
                original_list = []
                iter_list = []
                for i in range(self.batch_size):
                    batch_data = self.input_queue.get()
                    if batch_data is None:
                        keep_running = False
                        exit()
                        break  

            
                    frame, iter = batch_data
                
                    bindings = self._create_bindings(configured_infer_model)
                
                    preprocessed_batch = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    preprocessed_batch = cv2.resize(preprocessed_batch, (640, 640), 
                            interpolation=cv2.INTER_LINEAR)
                    bindings.input().set_buffer(preprocessed_batch)
                    bindings_list.append(bindings)
                    original_list.append(frame)
                    iter_list.append(iter)

                configured_infer_model.wait_for_async_ready(timeout_ms=10000, frames_count=self.batch_size)
                job = configured_infer_model.run_async(
                    bindings_list, partial(
                        self.callback,
                        input_batch=original_list,
                        bindings_list=bindings_list,
                        iter_list=iter_list
                    )
                )
                # job.wait(1000)
            # job.wait(5000)  # Wait for the last job

    def _get_output_type_str(self, output_info) -> str:
        if self.output_type is None:
            return str(output_info.format.type).split(".")[1].lower()
        else:
            self.output_type[output_info.name].lower()

    def _create_bindings(self, configured_infer_model) -> object:
        """
        Create bindings for input and output buffers.

        Args:
            configured_infer_model: The configured inference model.

        Returns:
            object: Bindings object with input and output buffers.
        """
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=(getattr(np, self._get_output_type_str(output_info)))
                )
            for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape, 
                    dtype=(getattr(np, self.output_type[name].lower()))
                )
            for name in self.output_type
            }
        return configured_infer_model.create_bindings(
            output_buffers=output_buffers
        )


def check_q(iter):
    return iter%3 < 1


def time_it(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.total_time += (end_time - start_time)
        wrapper.call_count += 1
        return result

    wrapper.total_time = 0
    wrapper.call_count = 0
    return wrapper


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
