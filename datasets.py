# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
# rewrite by YU Yue, Nanyang Technological University

import os
import math
import torch
import json
import torchvision
import skimage.io
import numpy as np

from tqdm import tqdm
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def select_20k_for_coco(sel_file, all_annfile):

    new_gts = []
    new_imgs = []
    selected_results = {}

    print('Building COCO20k dataset.')

    with open(all_annfile, "r") as f:
        all_info = json.load(f)

    with open(sel_file, "r") as f:
        lines = f.readlines()
        img_paths = [x.replace("\n", "") for x in lines]
    selected_img_numbers = [int(x.split("_")[-1].split(".")[0]) for x in img_paths]

    for selected_img_number in tqdm(selected_img_numbers):
        new_imgs.extend([x for x in all_info["images"] if int(x["id"]) == selected_img_number])
        new_gts.extend([x for x in all_info["annotations"] if int(x["image_id"]) == selected_img_number])

    selected_results["images"] = new_imgs
    selected_results["annotations"] = new_gts
    selected_results["categories"] = all_info["categories"]

    with open("datasets/instances_train2014_sel20k.json", "w") as f:
        json.dump(selected_results, f)

    print('Done.')

def discard_hard_for_voc(dataloader):
    hard_imgs = []
    for img_id, inp in enumerate(tqdm(dataloader)):
        objects = inp[1]["annotation"]["object"]
        obj_number = len(objects)
        hard_flag = np.zeros(obj_number)

        for i, obj in enumerate(range(obj_number)):
            hard_flag[i] = 1 if (objects[obj]["difficult"] == "1" or objects[obj]["truncated"] == "1") else 0

        if obj_number == np.sum(hard_flag):
            hard_imgs.append(img_id)

    return hard_imgs

def extract_gt_for_VOC(targets, remove_hards=False):
    objects = targets["annotation"]["object"]
    obj_number = len(objects)
    gt_bbx_list = []
    gt_cls_list = []

    for obj in range(obj_number):
        if remove_hards and (objects[obj]["truncated"] == "1" or objects[obj]["difficult"] == "1"):
            continue

        gt_cls_list.append(objects[obj]["name"])

        x1y1x2y2 = [int(objects[obj]["bndbox"]["xmin"]) - 1,
            int(objects[obj]["bndbox"]["ymin"]) - 1,
            int(objects[obj]["bndbox"]["xmax"]),
            int(objects[obj]["bndbox"]["ymax"])]

        gt_bbx_list.append(x1y1x2y2)

    return np.asarray(gt_bbx_list), gt_cls_list

def extract_gt_for_COCO(targets, remove_iscrowd=True):
    objects = targets
    obj_number = len(objects)
    gt_bbx_list = []
    gt_cls_list = []

    for obj in range(obj_number):
        if remove_iscrowd and objects[obj]["iscrowd"] == 1:
            continue

        gt_cls_list.append(objects[obj]["category_id"])

        bbx = objects[obj]["bbox"]
        x1y1x2y2 = [bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]]
        gt_bbx_list.append([int(round(x)) for x in x1y1x2y2])

    return np.asarray(gt_bbx_list), gt_cls_list

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # https://github.com/ultralytics/yolov5/blob/develop/utils/general.py
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

class Dataset:
    def __init__(self, dataset_name, set, remove_hards=False):

        self.dataset_name = dataset_name
        self.set = set
        self.name = self.dataset_name + "_" + self.set
        self.remove_hards = remove_hards
        self.hards = []

        if "VOC" in self.dataset_name:
            if self.dataset_name == "VOC07":
                self.year = "2007"
                self.root_path = "datasets/VOC2007"
            elif self.dataset_name == "VOC12":
                self.year = "2012"
                self.root_path = "datasets/VOC2012"
            else:
                raise ValueError("Unknown dataset.")

            self.dataloader = torchvision.datasets.VOCDetection(self.root_path, year=self.year, image_set=self.set,
                                                                transform=transform, download=False)
        elif self.dataset_name == "COCO20k":
            self.year = "2014"
            self.root_path = f"datasets/COCO/images/{self.set}{self.year}"
            self.sel20k = 'datasets/coco_20k_filenames.txt'
            self.all_annfile = "datasets/COCO/annotations/instances_train2014.json"
            self.annfile = "datasets/instances_train2014_sel20k.json"
            if not os.path.exists(self.annfile):
                select_20k_for_coco(self.sel20k, self.all_annfile)

            self.dataloader = torchvision.datasets.CocoDetection(self.root_path, annFile=self.annfile, transform=transform)

        else:
            raise ValueError("Unknown dataset.")

        if self.remove_hards:
            self.name += "-nohards"
            self.hards = self.get_hard_imgs()
            print("Discarded", len(self.hards), "images containing only objects annotated as 'hard'.")


    def get_img_name(self, inp):
        if "VOC" in self.dataset_name:
            return inp["annotation"]["filename"]
        elif "COCO" in self.dataset_name:
            return f"COCO_{self.set}{self.year}_{str(inp[0]['image_id'].zfill(12))}.jpg"
        else:
            raise ValueError("Unknown dataset.")

    def get_img(self, img_name):
        if "VOC" in self.dataset_name:
            image = skimage.io.imread(f"./datasets/VOC{self.year}/VOCdevkit/VOC{self.year}/JPEGImages/{img_name}")
        elif "COCO" in self.dataset_name:
            image = skimage.io.imread(f"./datasets/COCO/images/{self.set}{self.year}/{img_name}")
        else:
            raise ValueError("Unknown dataset.")
        return image

    def get_hard_imgs(self):
        hard_imgs = []
        hard_imgs_path = "datasets/hard_%s_%s_%s.txt" % (self.dataset_name, self.set, self.year)

        if not os.path.exists(hard_imgs_path):
            print("Discover hard images that should be discarded")
            if "VOC" in self.dataset_name:
                hard_imgs = discard_hard_for_voc(self.dataloader)

            with open(hard_imgs_path, "w") as f:
                for img in hard_imgs:
                    f.write(str(img) + "\n")
            print("Done, hard images are listed in datasets/hard_%s_%s_%s.txt"%(self.dataset_name, self.set, self.year))
        else:
            with open(hard_imgs_path, "r") as f:
                for l in f:
                    hard_imgs.append(int(l.strip()))

        return hard_imgs

    def extract_gt(self, targets):
        if "VOC" in self.dataset_name:
            return extract_gt_for_VOC(targets, remove_hards=self.remove_hards)
        elif "COCO" in self.dataset_name:
            return extract_gt_for_COCO(targets, remove_iscrowd=True)
        else:
            raise ValueError("Unknown dataset.")

    def extract_classes(self):
        cls_path = f"classes_{self.dataset_name}_{self.set}_{self.year}.txt"
        classes = []
        if not os.path.exists(cls_path):
            print("Extract all classes from the dataset")
            if "VOC" in self.dataset_name:
                for img_id, inp in enumerate(tqdm(self.dataloader)):
                    objects = inp[1]["annotation"]["object"]
                    for obj in range(len(objects)):
                        if objects[obj]["name"] not in classes:
                            classes.append(objects[obj]["name"])
            elif "COCO" in self.dataset_name:
                for img_id, inp in enumerate(tqdm(self.dataloader)):
                    objects = inp[1]
                    for obj in range(len(objects)):
                        if objects[obj]["category_id"] not in classes:
                            classes.append(objects[obj]["category_id"])
            else:
                raise ValueError("Unknown dataset.")

            with open(cls_path, "w") as f:
                for cls in classes:
                    f.write(str(cls) + "\n")
        else:
            with open(cls_path, "r") as f:
                for l in f:
                    classes.append(l.strip())

        return classes

class ImageDataset:
    def __init__(self, image_path):
        self.img_path = image_path
        self.name = image_path.split("/")[-1]

        with open(self.img_path, "rb") as f:
            img = Image.open(f).convert("RGB")

        img = transform(img)
        self.dataloader = [[img, self.img_path]]

    def get_img_name(self):
        return self.name.split(".")[0]

    def get_img(self):
        return skimage.io.imread(self.img_path)

'''
if __name__ == '__main__':
    dataset = Dataset("VOC07", "train")
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):
        img = inp[0]
        print(dataset.get_img_name(inp[1]))
        break
'''