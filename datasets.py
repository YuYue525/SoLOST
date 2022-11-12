import os
import math
import json
import torchvision
import skimage.io

from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

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
                                                                transform=transform, download=True)
        elif self.dataset_name == "COCO2k":
            self.year = "2014"
            self.root_path = f"datasets/COCO/images/{self.set}{self.year}"

            self.sel20k = 'datasets/coco_20k_filenames.txt'
            self.all_annfile = "datasets/COCO/annotations/instances_train2014.json"

            self.annfile = "datasets/instances_train2014_sel20k.json"

            if not os.path.exists(self.annfile):
                select_20k_for_coco(self.sel20k, self.all_annfile)

        else:
            raise ValueError("Unknown dataset.")
        '''
        if remove_hards:
            self.name += f"-nohards"
            self.hards = self.get_hards()
            print("Discarded", len(self.hards), "images containing only objects annotated as 'hard'.")
        '''

    def load_one_image(self, img_name):

        if "VOC" in self.dataset_name:
            image = skimage.io.imread(f"./datasets/VOC{self.year}/VOCdevkit/VOC{self.year}/JPEGImages/{img_name}")
        else:
            raise ValueError("Unkown dataset.")

        return image

    def get_image_name(self, inp):

        if "VOC" in self.dataset_name:
            img_name = inp["annotation"]["filename"]

        return img_name

def select_20k_for_coco(sel_file, all_annFile):
    print('Building COCO 20k dataset.')

    # all the annotations
    with open(all_annFile, "r") as f:
        train2014 = json.load(f)

    # selected images
    with open(sel_file, "r") as f:
        sel_20k = f.readlines()
        sel_20k = [s.replace("\n", "") for s in sel_20k]

    im20k = [str(int(s.split("_")[-1].split(".")[0])) for s in sel_20k]

    new_anno = []
    new_images = []

    for i in tqdm(im20k):
        new_anno.extend(
            [a for a in train2014["annotations"] if a["image_id"] == int(i)]
        )
        new_images.extend([a for a in train2014["images"] if a["id"] == int(i)])

    train2014_20k = {}
    train2014_20k["images"] = new_images
    train2014_20k["annotations"] = new_anno
    train2014_20k["categories"] = train2014["categories"]

    with open("datasets/instances_train2014_sel20k.json", "w") as outfile:
        json.dump(train2014_20k, outfile)

    print('Done.')

'''
if __name__ == '__main__':
    dataset = Dataset("VOC07", "trainval")
'''