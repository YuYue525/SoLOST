### Install DINO
This method is based on DINO [paper](https://arxiv.org/pdf/2104.14294.pdf). The framework can be installed using the following commands:
```
git clone https://github.com/facebookresearch/dino.git
cd dino; 
touch __init__.py
echo -e "import sys\nfrom os.path import dirname, join\nsys.path.insert(0, join(dirname(__file__), '.'))" >> __init__.py; cd ../;
```

The code was made using the commit ba9edd1 of DINO repo (please rebase if breakage).

## Apply LOST to one image
Following are scripts to apply LOST to an image defined via the `image_path` parameter and visualize the predictions (`pred`), the maps of the Figure 2 in the paper (`fms`) and the visulization of the seed expansion (`seed_expansion`). Box predictions are also stored in the output directory given by parameter `output_dir`.

```
python main_lost.py --image_path examples/VOC07_000236.jpg --visualize pred
python main_lost.py --image_path examples/VOC07_000236.jpg --visualize fms
python main_lost.py --image_path examples/VOC07_000236.jpg --visualize seed_expansion
```

## Launching LOST on datasets
Following are the different steps to reproduce the results of **LOST** presented in the paper. 

### PASCAL-VOC
Please download the PASCAL VOC07 and PASCAL VOC12 datasets ([link](http://host.robots.ox.ac.uk/pascal/VOC/)) and put the data in the folder `datasets`. There should be the two subfolders: `datasets/VOC2007` and `datasets/VOC2012`. In order to apply lost and compute corloc results (VOC07 61.9, VOC12 64.0), please launch:
```
python main_lost.py --dataset VOC07 --set trainval
python main_lost.py --dataset VOC12 --set trainval
```

### COCO
Please download the [COCO dataset](https://cocodataset.org/#home) and put the data in  `datasets/COCO`. Results are provided given the 2014 annotations following previous works. The following command line allows you to get results on the subset of 20k images of the COCO dataset (corloc 50.7), following previous litterature. To be noted that the 20k images are a subset of the `train` set.
```
python main_lost.py --dataset COCO20k --set train
```
