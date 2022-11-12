import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16

import dino.vision_transformer as vits

class vgg16Bottom(nn.Module):
    def __init__(self, original_model):
        super(vgg16Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.features = nn.Sequential(*list(self.features[0][:-1]))

    def forward(self, x):
        x = self.features(x)
        return x


class ResNet50Bottom(nn.Module):

    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        return x


def create_model(arch, patch_size, resnet_dilate, device):
    pretrained_flag = True if "imagenet" in arch else False
    replace_stride_with_dilation = None

    if "resnet" in arch:
        if resnet_dilate == 1:
            replace_stride_with_dilation = [False, False, False]
        elif resnet_dilate == 2:
            replace_stride_with_dilation = [False, False, True]
        elif resnet_dilate == 4:
            replace_stride_with_dilation = [False, True, True]

        model = resnet50(pretrained=pretrained_flag, replace_stride_with_dilation=replace_stride_with_dilation)
    elif "vgg16" in arch:
        model = vgg16(pretrained=pretrained_flag)
    else:
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)

    for p in model.parameters():
        p.requires_grad = False

    if not pretrained_flag:
        url = None

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        elif arch == "resnet50":
            url = "dino_resnet50_pretrain/dino_resnet50_pretrain.pth"

        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")

            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            strict_loading = False if "resnet" in arch else True
            msg = model.load_state_dict(state_dict, strict=strict_loading)

            print("Pretrained weights found at {} and loaded with msg: {}".format(url, msg))
        else:
            print("There is no reference weights available for this model => We use random weights.")

    if "resnet" in arch:
        model = ResNet50Bottom(model)
    elif "vgg16" in arch:
        model = vgg16Bottom(model)

    model.eval()
    model.to(device)

    return model
