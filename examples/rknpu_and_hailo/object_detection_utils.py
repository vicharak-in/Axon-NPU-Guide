import numpy as np
import cv2


def generate_color(class_id: int) -> tuple:
    """
    Generate a unique color for a given class ID.

    Args:
        class_id (int): The class ID to generate a color for.

    Returns:
        tuple: A tuple representing an RGB color.
    """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())

class_colors = [generate_color(i) for i in range(80)]

class ObjectDetectionUtils:
    def __init__(self, labels_path: str, padding_color: tuple = (114, 114, 114), label_font: str = "LiberationSans-Regular.ttf"):
        """
        Initialize the ObjectDetectionUtils class.

        Args:
            labels_path (str): Path to the labels file.
            padding_color (tuple): RGB color for padding. Defaults to (114, 114, 114).
            label_font (str): Path to the font used for labeling. Defaults to "LiberationSans-Regular.ttf".
        """
        self.labels = self.get_labels(labels_path)
        self.padding_color = padding_color
        self.label_font = label_font
    
    def get_labels(self, labels_path: str) -> list:
        """
        Load labels from a file.

        Args:
            labels_path (str): Path to the labels file.

        Returns:
            list: List of class names.
        """
        with open(labels_path, 'r', encoding="utf-8") as f:
            class_names = f.read().splitlines()
        return class_names


    def visualize(self, detections: dict, image: np.ndarray, width: int, height: int, min_score: float = 0.45, scale_factor: float = 1):
        """
        Visualize detections on the image.

        Args:
            detections (dict): Detection results.
            image (PIL.Image.Image): Image to draw on.
            output_path (str): Path to save the output image.
            width (int): Image width.
            height (int): Image height.
            min_score (float): Minimum score threshold. Defaults to 0.45.
            scale_factor (float): Scale factor for coordinates. Defaults to 1.
        """
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        scores = detections['detection_scores']
        

        for box, score, cl in zip(boxes, scores, classes):

            ymin, xmin, ymax, xmax = box
            ymin = int(ymin*height)
            xmin = int(xmin*width)
            ymax = int(ymax*height)
            xmax = int(xmax*width)
            

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), class_colors[cl], 1)
            cv2.putText(image, '{0} {1:.2f}'.format(self.labels[cl], score),
                        (xmin, ymin),
                        cv2.FONT_HERSHEY_PLAIN,
                        1, (0, 0, 255), 1)

        return image

        

    def extract_detections(self, input_data: list, threshold: float = 0.5) -> dict:
        """
        Extract detections from the input data.

        Args:
            input_data (list): Raw detections from the model.
            threshold (float): Score threshold for filtering detections. Defaults to 0.5.

        Returns:
            dict: Filtered detection results.
        """
        boxes, scores, classes = [], [], []
        num_detections = 0
        
        for i, detection in enumerate(input_data):
            if len(detection) == 0:
                continue

            for det in detection:
                bbox, score = det[:4], det[4]

                if score >= threshold:
                    boxes.append(bbox)
                    scores.append(score)
                    classes.append(i)
                    num_detections += 1
                    
        return {
            'detection_boxes': boxes, 
            'detection_classes': classes, 
            'detection_scores': scores,
            'num_detections': num_detections
        }
