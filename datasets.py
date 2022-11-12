import os
import math
import json
import torchvision
import skimage.io

from torchvision import transforms
from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

class Dataset:
    def __init__(self, dataset_name, set, remove_hards):

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
                                                                transform=transform, download=True)
        elif self.dataset_name == "COCO2k":
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
            self.hards = self.get_hards()
            print("Discarded", len(self.hards), "images containing only objects annotated as 'hard'.")

    def load_one_image(self, img_name):
        if "VOC" in self.dataset_name:
            image = skimage.io.imread(f"./datasets/VOC{self.year}/VOCdevkit/VOC{self.year}/JPEGImages/{img_name}")
        elif "COCO" in self.dataset_name:
            image = skimage.io.imread(f"./datasets/COCO/images/{self.set}{self.year}/{img_name}")
        else:
            raise ValueError("Unkown dataset.")
        return image

    def get_image_name(self, inp):
        if "VOC" in self.dataset_name:
            return inp["annotation"]["filename"]
        elif "COCO" in self.dataset_name:
            return str(inp[0]["image_id"])
        else:
            raise ValueError("Unkown dataset.")

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

'''
if __name__ == '__main__':
    dataset = Dataset("VOC07", "trainval")
'''
