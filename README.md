# Virtual-Contrast-CT
This repo is for paper "Virtual contrast enhancement for CT scans of abdomen and pelvis"


# UNet/FCN PyTorch

This repository contains simple PyTorch implementations of U-Net and FCN, which are deep learning segmentation methods proposed by Ronneberger et al. and Long et al.

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)



# Prepare Dataset and DataLoader


You need to prepare the dataset: non-contrast CT and real-contrast CT pairs with the same ID used as the input and ground truth


# Define training 

set the path in the training session and run:
python train.py



# test
set the path in the testing session and run:
python test.py

