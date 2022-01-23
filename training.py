# Basic python and ML Libraries
import os
import random
import numpy as np
import pandas as pd
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV
import cv2

# xml library for parsing xml files
from xml.etree import ElementTree as et

# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# torchvision libraries
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# these are the helper libraries imported.
from utils.engine import train_one_epoch, evaluate
import utils.utils
import utils.transforms as T

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from tqdm import tqdm

animals = ['horse', 'dog', 'sheep', 'bird', 'cat', 'cow']


def prepare_paths():
    import os
    names = []
    img_pths = []
    anno_pths = []

    for f in tqdm(os.listdir('VOCdevkit/VOC2012/Annotations')):
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
                f = True

        if not f:
            continue

        img_pths.append('VOCdevkit/VOC2012/JPEGImages/' + name + '.jpg')
        anno_pths.append(anno_pth)

    idx = np.arange(len(img_pths))
    np.random.shuffle(idx)

    img_pths = np.array(img_pths)[idx]
    anno_pths = np.array(anno_pths)[idx]

    img_pths_train = img_pths[:4000]
    anno_pths_train = anno_pths[:4000]
    img_pths_test = img_pths[4000:]
    anno_pths_test = anno_pths[4000:]

    return img_pths_train, anno_pths_train, img_pths_test, anno_pths_test


# defining the files directory and testing directory
files_dir = 'train'
test_dir = 'test'


class VOCDataset(torch.utils.data.Dataset):

    def __init__(self, imgs, annos, width, height, transforms=None):
        self.imgs = imgs
        self.annos = annos
        self.width = width
        self.height = height
        self.transforms = transforms

        self.classes = ['horse', 'dog', 'sheep', 'bird', 'cat', 'cow']

    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        anno_path = self.annos[idx]

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        img_res /= 255.0

        boxes = []
        labels = []
        tree = et.parse(anno_path)
        root = tree.getroot()

        wt = img.shape[1]
        ht = img.shape[0]

        for member in root.findall('object'):
            t = member.find('name').text

            if t not in animals:
                continue

            labels.append(self.classes.index(t))

            # bounding box
            xmin = int(float(member.find('bndbox').find('xmin').text))
            xmax = int(float(member.find('bndbox').find('xmax').text))

            ymin = int(float(member.find('bndbox').find('ymin').text))
            ymax = int(float(member.find('bndbox').find('ymax').text))

            xmin_corr = (xmin / wt) * self.width
            xmax_corr = (xmax / wt) * self.width
            ymin_corr = (ymin / ht) * self.height
            ymax_corr = (ymax / ht) * self.height

            boxes.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=img_res,
                                     bboxes=target['boxes'],
                                     labels=labels)

            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img_res, target

    def __len__(self):
        return len(self.imgs)


def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for box in (target['boxes']):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()


def get_object_detection_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


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


if __name__ == '__main__':
    model_path = 'model_bad.pth'
    train = True
    img_pths_train, anno_pths_train, img_pths_test, anno_pths_test = prepare_paths()
    dataset = VOCDataset(img_pths_train, anno_pths_train, 480, 480, transforms= get_transform(train=False))
    dataset_test = VOCDataset(img_pths_test, anno_pths_test, 480, 480, transforms= get_transform(train=False))

    print('Train size: %s' % len(dataset))
    print('Test size: %s' % len(dataset_test))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 6
    model = get_object_detection_model(num_classes)
    model.to(device)

    if train:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, )# weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)
        for epoch in range(5):
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            lr_scheduler.step()
            torch.save(model.state_dict(), model_path)

            evaluate(model, data_loader_test, device=device)
    else:
        print('Evaluating model')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        evaluate(model, data_loader_test, device=device)