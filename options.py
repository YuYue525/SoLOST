# Copyright 2021 - Valeo Comfort and Driving Assistance - Oriane Siméoni @ valeo.ai
# rewrite by YU Yue, Nanyang Technological University

import argparse

def init_parser():
    parser = argparse.ArgumentParser("Unsupervised object discovery with LOST.")

    parser.add_argument("--image_path", type=str, default=None,
                        help="If want to apply only on one image, give file path.")
    parser.add_argument("--dataset", default="VOC07", type=str, choices=[None, "VOC07", "VOC12", "COCO20k"],
                        help="Dataset name.")
    parser.add_argument("--set", default="train", type=str, choices=["val", "train", "trainval", "test"],
                        help="Path of the image to load.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory to store predictions and visualizations.")

    parser.add_argument("--no_evaluation", action="store_true", help="Compute the evaluation.")
    parser.add_argument("--no_hard", action="store_true",
                        help="Only used in the case of the VOC_all setup (see the paper).")
    parser.add_argument("--save_predictions", default=True, type=bool, help="Save predicted bouding boxes.")
    parser.add_argument("--patch_size", default=16, type=int, help="Patch resolution of the model.")
    parser.add_argument("--visualize", type=str, choices=["fms", "seed_expansion", "pred", None], default=None,
                        help="Select the different type of visualizations.")
    parser.add_argument("--resnet_dilate", type=int, default=2, help="Dilation level of the resnet model.")

    parser.add_argument("--arch", default="vit_small", type=str,
                        choices=["vit_tiny", "vit_small", "vit_base", "resnet50", "vgg16_imagenet",
                                 "resnet50_imagenet"], help="Model architecture.")

    parser.add_argument("--which_features", type=str, default="k", choices=["k", "q", "v"],
                        help="Which features to use")
    parser.add_argument("--k_patches", type=int, default=100,
                        help="Number of patches with the lowest degree considered.")

    parser.add_argument("--dinoseg", action="store_true", help="Apply DINO-seg baseline.")
    parser.add_argument("--dinoseg_head", type=int, default=4)

    return parser
