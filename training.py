## contains code needed for training

# Basic python and ML Libraries
import numpy as np
# for ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# We will be reading images using OpenCV

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
from utils import utils

# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

animals = ['horse', 'dog', 'sheep', 'bird', 'cat', 'cow']


def prepare_paths():
    import os

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


class VOCDataset(torch.utils.data.Dataset):

    def __init__(self, imgs, annos, width, height, transforms=None):
        self.imgs = imgs
        self.annos = annos
        self.width = width
        self.height = height
        self.transforms = transforms

        # classes: 0 index is reserved for background
        self.classes = ['horse', 'dog', 'sheep', 'bird', 'cat', 'cow']

    def __getitem__(self, idx):
        image_path = self.imgs[idx]
        anno_path = self.annos[idx]

        # reading the images and converting them to correct size and color
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0

        boxes = []
        labels = []
        tree = et.parse(anno_path)
        root = tree.getroot()

        # cv2 image gives size as height x width
        wt = img.shape[1]
        ht = img.shape[0]

        # box coordinates for xml files are extracted and corrected for image size given
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




# use our dataset and defined transformations

#dataset = FruitImagesDataset(files_dir, 480, 480, transforms= get_transform(train=True))
#dataset_test = FruitImagesDataset(files_dir, 480, 480, transforms= get_transform(train=False))


dataset = VOCDataset(img_pths_train, anno_pths_train, 480, 480, transforms= get_transform(train=True))
dataset_test = VOCDataset(img_pths_test, anno_pths_test, 480, 480, transforms= get_transform(train=False))


# # split the dataset in train and test set
# torch.manual_seed(1)
# indices = torch.randperm(len(dataset)).tolist()

# # train test split
# test_split = 0.2
# tsize = int(len(dataset)*test_split)
# dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
# dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=16, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 6
model = get_object_detection_model(num_classes)
model.to(device)
print('Loading model!')
model.load_state_dict(torch.load('mymodel2.pth', map_location=torch.device('cpu')))

def apply_nms(orig_prediction, iou_thresh=0.3):
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

for i in range(0):
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


import cv2
from random import randint, choice
from typing import Tuple, List
from nptyping import NDArray, Int8


class ImageData:
    ''' Han's model API. '''

    def __init__(self, x0, y0, w, h, label):
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.label = label


Color = Tuple[int, int, int]


def cnn_predict_mock() -> List[ImageData]:
    ''' Mock data for CNN prediction (just to test bounding boxes) '''
    example_labels = ['label1', 'label2', 'label3']
    bboxes = []

    for _ in range(randint(1, 5)):
        x, y = randint(1, 100), randint(1, 300)
        w, h = randint(1, 100), randint(1, 300)
        label = choice(example_labels)
        bboxes.append(ImageData(x, y, w, h, label))

    return bboxes


def cnn_predict(frame: NDArray[Int8]) -> List[ImageData]:
    ''' Call Han's model here... '''
    return cnn_predict_mock()


def draw_bounding_boxes(frame: NDArray[Int8], img_data: List[ImageData], color: Color = (255, 0, 0),
                        label_offset: int = 10, thickness: int = 2, pause: int = 50) -> None:
    ''' Outputs a PNG with a bounding box. '''
    for img in img_data:
        x, y, width, height, label = int(img.x0), int(img.y0), int(img.w), int(img.h), img.label
        frame = cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness)
        frame = cv2.putText(frame, label, (x, y - label_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    cv2.imshow('Simulation', frame)
    cv2.waitKey(pause)


model.eval()

def display_frames(video_path: str = 'cats_trimmed.mp4') -> None:
    ''' Render all frames in video with bounding boxes. '''
    vidcap = cv2.VideoCapture(video_path)

    while True:
        success, frame = vidcap.read()
        if success:
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

            for i in range(len(boxes)):
                x1, y1, x2, y2 = [box.detach().item() for box in boxes[i]]

                w = x2 - x1
                h = y2 - y1
                x = x1
                y = y1

                print('%s %s %s %s %s' % (x,y,w,h, animals[labels[i]]))
                data = ImageData(x,y,w,h, animals[labels[i]])
                img_data.append(data)

            draw_bounding_boxes(img_res, img_data)
        else:
            break

    vidcap.release()
    cv2.destroyAllWindows()


display_frames()