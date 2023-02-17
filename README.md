# Virtual-Contrast-CT
This repo is for the paper - [Virtual contrast enhancement for CT scans of abdomen and pelvis](https://www.sciencedirect.com/science/article/abs/pii/S0895611122000672)


# About UNet/FCN PyTorch

This repository is based on PyTorch implementations of U-Net and FCN, which are deep-learning segmentation methods proposed by Ronneberger et al. and Long et al.

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [Fully Convolutional Networks for Semantic Segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)


# Prepare Dataset and DataLoader

You need to prepare the dataset: non-contrast CT and real-contrast CT pairs with the same ID as the input and ground truth.


# Define training 

Set the path in the training session and run:
- python train.py



# Define testing
set the path in the testing session and run:
- python test.py

# Model & Test sample: [GoolgeDrive](https://drive.google.com/drive/folders/16fMQX28qh5fvv_mg055DmTKWaZUvI9eY?usp=share_link)

# Pretrained model:
Load pre-train model: pretrained_intensity_model_4level_3c6.pth
# Trained model:
Try early stage: earlystage_C_32_BCE_HRl5_pix_G_3C_adddreg_s256c224_tw_40.pth
