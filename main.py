## main.py - Runs the main simulation of the bounding boxes.

# We will be reading images using OpenCV
import cv2

# Pytorch libraries
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Basic python and ML Libraries
import numpy as np

# For typing annotations
from typing import Tuple, List, Dict
from nptyping import NDArray, Int8

# For ignoring warnings
import warnings
warnings.filterwarnings('ignore')

ANIMALS = ['horse', 'dog', 'sheep', 'bird', 'cat', 'cow']
NUM_CLASSES = 6
Color = Tuple[int, int, int]

class ImageData:
    ''' Han's model API attributes. '''
    def __init__(self, x0: int, y0: int, w: int, h: int, label: str) -> None:
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.label = label

def get_object_detection_model(num_classes: int) -> torchvision.models.detection.FasterRCNN:
    ''' Returns predictor model we'll use to calculate the animals' bounding boxes. '''
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def apply_nms(orig_prediction: Dict[str, torch.Tensor], iou_thresh: int=0.3) -> Dict[str, torch.Tensor]:
    ''' Non-maximum suppression on the bounding boxes. '''
    # torchvision returns the indices of the bounding boxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    return final_prediction

def draw_bounding_boxes(frame: NDArray[Int8], img_data: List[ImageData], color: Color = (255, 0, 0),
                        label_offset: int = 10, thickness: int = 2, pause: int = 50) -> None:
    ''' Renders a video frame with bounding boxes. '''
    for img in img_data:
        x, y, width, height, label = int(img.x0), int(img.y0), int(img.w), int(img.h), img.label
        frame = cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)
        frame = cv2.putText(frame, label, (x, y - label_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    cv2.imshow('Simulation', frame)
    cv2.waitKey(pause)

def display_frames(video_path: str) -> None:
    ''' Render all frames in video with bounding boxes. '''
    vidcap = cv2.VideoCapture(video_path)
    counter = 0
    INTERVAL = 3
    MIN_SCORE = 0.7

    boxes = []
    labels = []

    while True:
        success, frame = vidcap.read()
        if not success:
            break

        counter += 1

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        orig_height, orig_width = img_rgb.shape[0], img_rgb.shape[1]
        img_rgb /= 255.0
        img_res = cv2.resize(img_rgb, (480, 480), cv2.INTER_AREA)
        scale_h, scale_w = orig_height / 480, orig_width / 480
        img = np.array(img_res)
        img = np.transpose(img, (2, 0, 1))

        if counter % INTERVAL == 0:
            d = model([torch.tensor(img).to(device)])[0]
            d = apply_nms(d, iou_thresh=0.01)

            boxes = []
            labels = []

            for j, score in enumerate((d['scores'])):
                if score > MIN_SCORE:
                    boxes.append(d['boxes'][j])
                    labels.append(d['labels'][j])

        img_data = []

        for i, curr_boxes in enumerate(boxes):
            x1, y1, x2, y2 = [box.detach().item() for box in curr_boxes]
            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1
            data = ImageData(x * scale_w, y * scale_h, w * scale_w, h * scale_h, ANIMALS[labels[i]])
            img_data.append(data)

        draw_bounding_boxes(img_rgb, img_data)

    vidcap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = get_object_detection_model(NUM_CLASSES)
    model.to(device)
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()

    # remember to put this file into this directory
    display_frames('horses_and_dogs.mp4')