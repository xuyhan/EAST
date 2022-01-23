## training.py - Contains code needed for training the models.

# Basic python and ML Libraries
import numpy as np
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans

# these are the helper libraries imported.
from utils import utils

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import os
from typing import Dict
from voc import VOCDataset

import main

def prepare_paths():
    labs = {}
    names = []
    img_pths = []
    anno_pths = []

    for f in os.listdir('VOCdevkit/VOC2012/Annotations'):
        name = f.split('.')[0]
        names.append(name)
        anno_pth = 'VOCdevkit/VOC2012/Annotations/' + f

        tree = et.parse(anno_pth)
        root = tree.getroot()

        f = False
        l = ''

        for member in root.findall('object'):
            l = member.find('name').text

            if l in ['horse', 'dog', 'sheep', 'bird', 'cat', 'cow']:
                if l not in labs:
                    labs[l] = len(labs)

                f = True

        if not f:
            continue

        img_pths.append('VOCdevkit/VOC2012/JPEGImages/' + name + '.jpg')
        anno_pths.append(anno_pth)

    idx = np.arange(len(img_pths))
    np.random.shuffle(idx)

    img_pths = np.array(img_pths)[idx]
    anno_pths = np.array(anno_pths)[idx]
    img_pths_train = img_pths[:3000]
    anno_pths_train = anno_pths[:3000]
    img_pths_test = img_pths[3000:]
    anno_pths_test = anno_pths[3000:]

    return img_pths_train, anno_pths_train, img_pths_test, anno_pths_test, labs

img_pths_train, anno_pths_train, img_pths_test, anno_pths_test, labs = prepare_paths()

# Send train=True fro training transforms and False for val/test transforms
def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(0.5),
            # ToTensorV2 converts image to pytorch tensor without div by 255
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# use our dataset and defined transformations
# dataset = FruitImagesDataset(files_dir, 480, 480, transforms= get_transform(train=True))
# dataset_test = FruitImagesDataset(files_dir, 480, 480, transforms= get_transform(train=False))

dataset = VOCDataset(img_pths_train, anno_pths_train, 480, 480, transforms= get_transform(train=True))
dataset_test = VOCDataset(img_pths_test, anno_pths_test, 480, 480, transforms= get_transform(train=False))

# split the dataset in train and test set
# torch.manual_seed(1)
# indices = torch.randperm(len(dataset)).tolist()

# train test split
# test_split = 0.2
# tsize = int(len(dataset) * test_split)
# dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=16, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

def apply_nms(orig_prediction: Dict[torch.Tensor], iou_thresh: int=0.3) -> Dict[torch.Tensor]:
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    return final_prediction

def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')

def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    for box in (target['boxes']):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    
    plt.show()

batch_size = 5
buffer = []
EPOCHS = 0 # set to positive number to train

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 6
    model = main.get_object_detection_model(num_classes)
    model.to(device)
    print('Loading model!')
    model.load_state_dict(torch.load('mymodel2.pth', map_location=torch.device('cpu')))

    for i in range(EPOCHS):
        img, target = dataset_test[i]

        if len(buffer) < batch_size:
            buffer.append(img.unsqueeze(0))
        else:
            buffer = torch.concat(buffer)
            model.eval()
            with torch.no_grad():
                prediction = model(buffer.to(device))

            for j in range(batch_size):
                nms_prediction = apply_nms(prediction[j], iou_thresh=0.1)
                plot_img_bbox(torch_to_pil(buffer[j]), nms_prediction)

            buffer = []