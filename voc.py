import torch

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