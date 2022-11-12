import os
import math
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
        # elif self.dataset_name == "COCO2k":

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

'''
if __name__ == '__main__':
    dataset = Dataset("VOC07", "trainval")
'''