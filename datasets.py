import os
import math
import json
import torchvision
import skimage.io
import numpy as np

from torchvision import transforms
from tqdm import tqdm

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
            raise ValueError("Unkown dataset.")

    def get_image(self, img_name):
        if "VOC" in self.dataset_name:
            image = skimage.io.imread(f"./datasets/VOC{self.year}/VOCdevkit/VOC{self.year}/JPEGImages/{img_name}")
        elif "COCO" in self.dataset_name:
            image = skimage.io.imread(f"./datasets/COCO/images/{self.set}{self.year}/{img_name}")
        else:
            raise ValueError("Unkown dataset.")
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





if __name__ == '__main__':
    dataset = Dataset("VOC07", "train")
    pbar = tqdm(dataset.dataloader)
    for im_id, inp in enumerate(pbar):
        img = inp[0]
        print(dataset.get_img_name(inp[1]))
        break
