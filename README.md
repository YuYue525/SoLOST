# LOST 
Pytorch implementation of the unsupervised object discovery method **LOST**. More details can be found in the paper:

**Localizing Objects with Self-Supervised Transformers and no Labels**, BMVC 2021 [[arXiv](https://arxiv.org/abs/2109.14279)]  
by *Oriane Siméoni, Gilles Puy, Huy V. Vo, Simon Roburin, Spyros Gidaris, Andrei Bursuc, Patrick Pérez, Renaud Marlet and Jean Ponce*

We refer to the work from the following citation:


```
@inproceedings{LOST,
   title = {Localizing Objects with Self-Supervised Transformers and no Labels},
   author = {Oriane Sim\'eoni and Gilles Puy and Huy V. Vo and Simon Roburin and Spyros Gidaris and Andrei Bursuc and Patrick P\'erez and Renaud Marlet and Jean Ponce},
   journal = {Proceedings of the British Machine Vision Conference (BMVC)},
   month = {November},
   year = {2021}
}
```

## Installation of LOST

### Repo Initialization

```
git clone --recursive https://github.com/YuYue525/AI6103_project.git
```

### Dependencies

This code was implemented with python 3.7, PyTorch 1.7.1 and CUDA 10.2. Please install [PyTorch](https://pytorch.org/). In order to install the additionnal dependencies, please launch the following command:

```
pip install -r requirements.txt
```

### Install DINO
This method is based on DINO [paper](https://arxiv.org/pdf/2104.14294.pdf). The framework can be installed using the following commands:
```
cd dino; touch __init__.py
echo -e "import sys\nfrom os.path import dirname, join\nsys.path.insert(0, join(dirname(__file__), '.'))" >> __init__.py; cd ../;
```

The code was made using the commit ba9edd1 of DINO repo (please rebase if breakage).

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

### Different models
We have tested the method on different setups of the VIT model, corloc results are presented in the following table (more can be found in the paper). 

<table>
  <tr>
    <th rowspan="2">arch</th>
    <th rowspan="2">pre-training</th>
    <th colspan="3">dataset</th>
  </tr>
  <tr>
    <th>VOC07</th>
    <th>VOC12</th>
    <th>COCO20k</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>DINO</td>
    <td>61.5</td>
    <td>64.1</td>
    <td>50.7</td>
  <tr>
  <tr>
    <td>ViT-S/8</td>
    <td>DINO</td>
    <td>55.3</td>
    <td>57.0</td>
    <td>49.5</td>
  <tr>
  <tr>
    <td>ViT-B/16</td>
    <td>DINO</td>
    <td>60.0</td>
    <td>63.3</td>
    <td>50.0</td>
  <tr>
  <tr>
    <td>ResNet50</td>
    <td>DINO</td>
    <td>36.8</td>
    <td>42.7</td>
    <td>26.5</td>
  <tr>
  <tr>
    <td>ResNet50</td>
    <td>Imagenet</td>
    <td>33.8</td>
    <td>39.1</td>
    <td>25.5</td>
  <tr>
</table>

However, when measuring the distance among features obtained, the original paper directly computes the dot products of feature pairs without normalization. In our implementation, we also use other measurement like cosine similarity to measure the patch similarity. The following table shows the results:

<table>
  <tr>
    <th rowspan="3">arch</th>
    <th rowspan="3">pre-training</th>
    <th colspan="6">dataset</th>
  </tr>
  <tr>
    <th colspan="2">VOC07</th>
    <th colspan="2">VOC12</th>
    <th colspan="2">COCO20k</th>
  </tr>
   <tr>
    <th>dot product</th>
    <th>cosine sim</th>
    <th>dot product</th>
    <th>cosine sim</th>
    <th>dot product</th>
    <th>cosine sim</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>DINO</td>
    <td>61.5</td>
    <td><B>61.7</td>
    <td>64.1</td>
    <td><B>64.3</td>
    <td>50.7</td>
    <td>50.7</td>
  <tr>
  <tr>
    <td>ViT-S/8</td>
    <td>DINO</td>
    <td>55.3</td>
    <td>55.3</td>
    <td>57.0</td>
    <td><B>57.2</td>
    <td>49.5</td>
    <td>49.5</td>
  <tr>
  <tr>
    <td>ViT-B/16</td>
    <td>DINO</td>
    <td>60.0</td>
    <td><B>60.1</td>
    <td>63.3</td>
    <td><B>63.4</td>
    <td>50.0</td>
    <td>50.0</td>
  <tr>
  <tr>
    <td>ResNet50</td>
    <td>DINO</td>
    <td><B>36.8</td>
    <td>36.5</td>
    <td><B>42.7</td>
    <td>42.5</td>
    <td>26.5</td>
    <td>26.5</td>
  <tr>
  <tr>
    <td>ResNet50</td>
    <td>Imagenet</td>
    <td><B>33.8</td>
    <td>33.6</td>
    <td><B>39.1</td>
    <td>39.0</td>
    <td>25.5</td>
    <td>25.5</td>
  <tr>
</table>

In our implementation, we also tried Pearson product-moment correlation coefficient (PCCs) to measure the patch similarity. The following table shows the results:

<table>
  <tr>
    <th rowspan="3">arch</th>
    <th rowspan="3">pre-training</th>
    <th colspan="6">dataset</th>
  </tr>
  <tr>
    <th colspan="2">VOC07</th>
    <th colspan="2">VOC12</th>
    <th colspan="2">COCO20k</th>
  </tr>
   <tr>
    <th>dot product</th>
    <th>PCCs</th>
    <th>dot product</th>
    <th>PCCs</th>
    <th>dot product</th>
    <th>PCCs</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>DINO</td>
    <td>61.5</td>
    <td><B>61.6</td>
    <td>64.1</td>
    <td><B>64.3</td>
    <td>50.7</td>
    <td>50.7</td>
  <tr>
  <tr>
    <td>ViT-S/8</td>
    <td>DINO</td>
    <td>55.3</td>
    <td>55.3</td>
    <td>57.0</td>
    <td><B>57.2</td>
    <td>49.5</td>
    <td>49.5</td>
  <tr>
  <tr>
    <td>ViT-B/16</td>
    <td>DINO</td>
    <td>60.0</td>
    <td><B>60.1</td>
    <td>63.3</td>
    <td><B>63.4</td>
    <td>50.0</td>
    <td>50.0</td>
  <tr>
  <tr>
    <td>ResNet50</td>
    <td>DINO</td>
    <td><B>36.8</td>
    <td>36.5</td>
    <td><B>42.7</td>
    <td>42.5</td>
    <td>26.5</td>
    <td>26.5</td>
  <tr>
  <tr>
    <td>ResNet50</td>
    <td>Imagenet</td>
    <td><B>33.8</td>
    <td>33.6</td>
    <td><B>39.1</td>
    <td>39.0</td>
    <td>25.5</td>
    <td>25.5</td>
  <tr>
</table>

Previous results on the dataset `VOC07` can be obtained by launching: 
```bash
python main_lost.py --dataset VOC07 --set trainval #VIT-S/16
python main_lost.py --dataset VOC07 --set trainval --patch_size 8 #VIT-S/8
python main_lost.py --dataset VOC07 --set trainval --arch vit_base #VIT-B/16
python main_lost.py --dataset VOC07 --set trainval --arch resnet50 #Resnet50/DINO
python main_lost.py --dataset VOC07 --set trainval --arch resnet50_imagenet #Resnet50/imagenet
```
