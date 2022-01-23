# Basic python and ML Libraries
import numpy as np
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# xml library for parsing xml files
from xml.etree import ElementTree

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# We will be reading images using OpenCV
import cv2
from random import randint, choice

# For typing annotations
from typing import Tuple, List
from nptyping import NDArray, Int8

ANIMALS = ['horse', 'dog', 'sheep', 'bird', 'cat', 'cow']
Color = Tuple[int, int, int]
class ImageData:
    ''' Han's model API. '''
    def __init__(self, x0, y0, w, h, label):
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

def apply_nms(orig_prediction, iou_thresh=0.3):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

batch_size = 5
buffer = []

def predict_mock() -> List[ImageData]:
    ''' Mock data for CNN prediction (just to test bounding boxes) '''
    example_labels = ['label1', 'label2', 'label3']
    bboxes = []

    for _ in range(randint(1, 5)):
        x, y = randint(1, 100), randint(1, 300)
        w, h = randint(1, 100), randint(1, 300)
        label = choice(example_labels)
        bboxes.append(ImageData(x, y, w, h, label))

    return bboxes

def draw_bounding_boxes(frame: NDArray[Int8], img_data: List[ImageData], color: Color = (255, 0, 0),
                        label_offset: int = 10, thickness: int = 2, pause: int = 50) -> None:
    ''' Outputs a PNG with a bounding box. '''
    for img in img_data:
        x, y, width, height, label = int(img.x0), int(img.y0), int(img.w), int(img.h), img.label
        frame = cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)
        frame = cv2.putText(frame, label, (x, y - label_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    cv2.imshow('Simulation', frame)
    cv2.waitKey(pause)

def display_frames(video_path: str = 'cats_trimmed.mp4') -> None:
    ''' Render all frames in video with bounding boxes. '''
    vidcap = cv2.VideoCapture(video_path)

    while True:
        success, frame = vidcap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (480, 480), cv2.INTER_AREA)
        img_res /= 255.0
        img = np.array(img_res)
        img = np.transpose(img, (2, 0, 1))

        d = model([torch.tensor(img).to(device)])[0]
        d = apply_nms(d, iou_thresh=0.01)

        boxes = d['boxes']
        labels = d['labels']
        img_data = []

        for i, curr_boxes in enumerate(boxes):
            x1, y1, x2, y2 = [box.detach().item() for box in curr_boxes]
            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1
            data = ImageData(x, y, w, h, ANIMALS[labels[i]])
            img_data.append(data)

        draw_bounding_boxes(img_res, img_data)

    vidcap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 6
    model = get_object_detection_model(num_classes)
    model.to(device)
    print('Loading model!')
    model.load_state_dict(torch.load('mymodel2.pth', map_location=torch.device('cpu')))
    model.eval()
    display_frames()