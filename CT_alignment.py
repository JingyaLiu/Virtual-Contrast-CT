from __future__ import print_function, division
import os,cv2,numbers,random,math,imutils
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import skimage, math
from skimage import io, transform
import matplotlib.pyplot as plt
from utils import get_ids
from skimage.measure import compare_ssim as ssim
# Ignore warnings
import warnings
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import scipy
import scipy.misc
from skimage.measure import compare_ssim


class ConstrastCTDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        # self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        # img_name = os.path.join(self.image_dir, sorted(os.listdir(self.image_dir))[idx])
        # image = io.imread(img_name)
        # label_name = os.path.join(self.label_dir, sorted(os.listdir(self.image_dir))[idx])
        # label = io.imread(label_name)

        img_name = os.path.join(self.image_dir, (os.listdir(self.image_dir))[idx])
        image = io.imread(img_name)
        # print('image before', image.shape)
        # image = transform.resize(image,(512,512))
        # print('image after', image.shape)
        label_name = os.path.join(self.label_dir, (os.listdir(self.image_dir))[idx])
        label = io.imread(label_name)
        # print('label before', label.shape)
        # label = transform.resize(label,(512,512))
        # print('label after', label.shape)

        # print(img_name,label_name)
        # sample = {'image': image, 'label': label}

        # if self.transform:
        #     sample = self.transform(sample)
        # image, label = sample['image'], sample['label']

        return image, label, img_name

class ConstrastCTDataset_intensity(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, image_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        # self.label_dir = label_dir
        # self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, os.listdir(self.image_dir)[idx])
        image = io.imread(img_name)
        print(os.listdir(self.image_dir)[idx])
        # label_name = os.path.join(self.label_dir, os.listdir(self.image_dir)[idx])
        # label = io.imread(label_name)
        # sample = {'image': image, 'label': label}

        # if self.transform:
        #     sample = self.transform(sample)
        # image, label = sample['image'], sample['label']

        return image, img_name



def align_ct(image, label):
    # Read the images to be aligned
    im1 =  image;
    im2 =  label;

    # Convert images to grayscale
    # im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    # im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = im1.shape

    # Define the motion model
    # warp_mode = cv2.MOTION_TRANSLATION
    warp_mode = cv2.MOTION_AFFINE
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    # (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    (cc, warp_matrix) = cv2.findTransformECC (im1,im2,warp_matrix, warp_mode, criteria,None,1)

    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    # Show final results
    label_align = im2_aligned

    label_overlap= label_align + im1
    # label_overlap[label_overlap>0] = 255

    return label_align, label_overlap


def add_intensity(images):
    intensity_level = [0.5,1,1.5,2.0]
    # intensity_level[0]
    print('procesing the intensity')
    # print(image.max(),image.min(), (image.max()-image.min()), image.shape, np.ones(images.shape).shape,images.shape)
    # image_0 = np.ones(images.shape)*intensity_level[0] + images[images<255]
    # image_1 = np.ones(images.shape)*intensity_level[1] + images[images<255]
    # image_2 = np.ones(images.shape)*intensity_level[2] + images[images<255]
    # image_3 = np.ones(images.shape)*intensity_level[3] + images[images<255]
    image_0 = images * intensity_level[0]
    image_1 = images * intensity_level[1]
    image_2 = images * intensity_level[2]
    image_3 = images * intensity_level[3]

    return image_0,image_1,image_2,image_3

def rotate(
    img,  #image matrix
    angle #angle of rotation
    ):

    height = img.shape[0]
    width = img.shape[1]

    if angle%180 == 0:
        scale = 1
    elif angle%90 == 0:
        scale = float(max(height, width))/min(height, width)
    else:
        scale = math.sqrt(pow(height,2)+pow(width,2))/min(height, width)

    #print 'scale %f\n' %scale

    rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
    rotateImg = cv2.warpAffine(img, rotateMat, (width, height))
    return rotateImg


def add_rotation(images):
    rotation = [90,180,270]
    # intensity_level[0]
    print('procesing the intensity')
    # print(image.max(),image.min(), (image.max()-image.min()), image.shape, np.ones(images.shape).shape,images.shape)
    # image_0 = np.ones(images.shape)*intensity_level[0] + images[images<255]
    # image_1 = np.ones(images.shape)*intensity_level[1] + images[images<255]
    # image_2 = np.ones(images.shape)*intensity_level[2] + images[images<255]
    # image_3 = np.ones(images.shape)*intensity_level[3] + images[images<255]
    image_90 = rotate(images,rotation[0])
    image_180 = rotate(images,rotation[1])
    image_270 = rotate(images,rotation[2])
    return image_90,image_180,image_270


def calculate_diff(image,label):
    (score, diff) = compare_ssim(image,label, full=True)
    # diff = ssim()
    return score, diff


def get_contrast_image(image):

    pre_image = image
    temp_image = image
    intensity_level = 230
    # intensity_level[0]
    print('procesing the intensity')
    # print(image.max(),image.min(), (image.max()-image.min()), image.shape, np.ones(images.shape).shape,images.shape)
    # image_0 = np.ones(images.shape)*intensity_level[0] + images[images<255]
    # image_1 = np.ones(images.shape)*intensity_level[1] + images[images<255]
    # image_2 = np.ones(images.shape)*intensity_level[2] + images[images<255]
    # image_3 = np.ones(images.shape)*intensity_level[3] + images[images<255]
    temp_image[temp_image>intensity_level] = temp_image[temp_image>intensity_level] * 2
    temp_image[temp_image<intensity_level] = temp_image[temp_image<intensity_level] * 0.5

    # image_contrast = pre_image + temp_image*0.05
    # image_1 = images * intensity_level[1]
    # image_2 = images * intensity_level[2]
    # image_3 = images * intensity_level[3]
    # print(temp_image.max(),temp_image.min())
    return temp_image



############################# intensity difference #########################################################


# traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/preprocess/data/9_24/new_training/'
# # traindir_mask = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/preprocess/data/ltrain_label_early/'
# traindir_mask_align = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/preprocess/data/9_24/new_training_label_align/'
# # traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/processing/'
# # save_img_path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/4levelintensity/'
# # print('load train data')
# # save_img_path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/7level/'
# # os.mkdir(traindir_mask_align)
# train_set = ConstrastCTDataset(traindir_img,traindir_mask_align)
# #
# #
# # # print(train_set)
# for inputs, labels, img_name in train_set:
# #     # print(inputs.size(), labels.size())
#     image, label = inputs, labels
# #     #  get ssim difference
# #     # (score, diff_image) = calculate_diff(image,label)
# #     diff_image = label - image
# #     # print(score,diff_image.max(),diff_image.min(),image.max(),label.max())
# #     print('save file: '+ img_name.split('/')[-1])
# #     diff_image = (diff_image).astype("uint8")
# #     diff_image[diff_image > 250] = 0
# #     diff_image[diff_image < 50] = 0
# #
# #     # print(score,diff_image.max(),diff_image.min(),image.max(),label.max())
# #     # ret,thresh = cv2.threshold(diff_image,110,130,cv2.THRESH_BINARY)
# #     # ret2,thresh = cv2.threshold(diff_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# #
# #     # thresh = cv2.adaptiveThreshold(diff_image, 255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
# #     # thresh = cv2.threshold(diff_image, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# #     # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# #     # 	cv2.CHAIN_APPROX_SIMPLE)
# #     # cnts = imutils.grab_contours(cnts)
# #
# #
# #     # f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(16,4))
#     f,(ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
# #
#     ax1.imshow(image, cmap = 'gray')
#     ax2.imshow(label, cmap = 'gray')
#     # ax3.imshow(diff_image, cmap = 'gray')
#     # ax4.imshow(thresh, cmap = 'gray')
#     # plt.show()
#     plt.savefig('check/check_'+ img_name.split('/')[-1],bbox_inches='tight')





############################# alignment ###############################################

# traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/EARLY_TRAIN'
# traindir_mask = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/EARLY_TRAIN_LABEL'
# traindir_mask_align = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/EARLY_TRAIN_LABEL_NEW/'
# traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TRAIN'
# traindir_mask = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TRAIN_LABEL/'
# traindir_mask_align = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TRAIN_LABEL_NEW/'
# traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/LATE_TRAIN'
# traindir_mask = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/LATE_TRAIN_LABEL'
# traindir_mask_align = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/LATE_TRAIN_LABEL_NEW/'

# traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/EARLY_TEST'
# traindir_mask = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/EARLY_TEST_LABEL'
# traindir_mask_align = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/EARLY_TEST_LABEL_NEW/'
traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TEST'
traindir_mask_align = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TEST_LABEL/'
traindir_mask = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/PORTAL_TEST_LABEL_NEW/'
# traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/LATE_TEST'
# traindir_mask = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/LATE_TEST_LABEL'
# traindir_mask_align = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/FOUR_STAGES/LATE_TEST_LABEL_NEW/'



# traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/processing/'
# save_img_path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/4levelintensity/'
# print('load train data')
# save_img_path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/7level/'
if not os.path.isdir(traindir_mask_align):
    os.mkdir(traindir_mask_align)
train_set = ConstrastCTDataset(traindir_img,traindir_mask)
# print(train_set.type)
for inputs, labels, img_name in train_set:
    # print(inputs.size(), labels.size())
    image, label = inputs,labels
    # Check the  Alignment
    # selected the bed region
    if not os.path.exists(traindir_mask_align + img_name.split('/')[-1]):
        print('save file: '+ img_name.split('/')[-1])
        label_align,label_overlap = align_ct(image, label)
        scipy.misc.imsave(traindir_mask_align + img_name.split('/')[-1],label_align)

    # print(sum(sum(label - image)),sum(sum(label - label_align)))
    # print(image.shape, label.shape)
    # f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(8,8))
    # ax1.imshow(image)
    # ax2.imshow(label)
    # ax3.imshow(label_align)
    # ax4.imshow(label_overlap)
    # plt.savefig('check/check_'+ img_name.split('/')[-1],bbox_inches='tight')



# ####################################Intensity ##################################################
# #
# traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/processing/'
# save_img_path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/Test_network/contrast/'
# save_original_path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/Test_network/precontrast/'
# # print('load train data')
# # save_img_path = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/7level/'
# if not os.path.exists(save_img_path):
#     # print('here')
#     os.makedirs(save_img_path)
#     os.makedirs(save_original_path)
#
# train_set = ConstrastCTDataset_intensity(image_dir = traindir_img)
# i = 0
# # print(train_set)
# for inputs, img_name in train_set:
#     print(inputs.shape)
#     image = inputs
#     # Check the  Alignment
#     # selected the bed region
#     # os.makedirs('data/Intensity0/')
#     # os.makedirs('data/Intensity1/')
#     # os.makedirs('data/Intensity2/')
#     # os.makedirs('data/Intensity3/')
#
#     ################Test the newtwork performance ##################
#     cv2.imwrite(save_original_path+'pre_'+ img_name.split('/')[-1],inputs)
#     image_contrast = get_contrast_image(image)
#
#     cv2.imwrite(save_img_path+'contrast_'+ img_name.split('/')[-1],image_contrast)
#     i = i+1
#     assert i != 100201

#     # image_0,image_1,image_2,image_3 = add_intensity(inputs)
#     # # image_90,image_180,image_270 = add_rotation(inputs)
#     # # plt.imshow(image_0, cmap='gray',vmin = 0,vmax = 255,)
#     # # plt.savefig('data/Intensity0/'+ img_name.split('/')[-1],bbox_inches=0)
#     # # plt.imshow(image_0, cmap='gray',vmin = 0,vmax = 255)
#     # # plt.savefig('data/Intensity1/'+ img_name.split('/')[-1],bbox_inches='tight')
#     # # plt.imshow(image_0, cmap='gray',vmin = 0,vmax = 255)
#     # # plt.savefig('data/Intensity2/'+ img_name.split('/')[-1],bbox_inches='tight')
#     # # plt.imshow(image_0, cmap='gray',vmin = 0,vmax = 255)
#     # # plt.savefig('data/Intensity3/'+ img_name.split('/')[-1],bbox_inches='tight')
#     #
#     # # plt.show()
#     # # scipy.misc.imsave('data/Intensity0/'+ img_name.split('/')[-1],image_0)
#     # # scipy.misc.imsave('data/Intensity1/'+ img_name.split('/')[-1],image_1)
#     # # scipy.misc.imsave('data/Intensity2/'+ img_name.split('/')[-1],image_2)
#     # # scipy.misc.imsave('data/Intensity3/'+ img_name.split('/')[-1],image_3)
#     # cv2.imwrite(save_img_path+'Int0_'+ img_name.split('/')[-1],image_0)
#     # cv2.imwrite(save_img_path+'Int1_'+ img_name.split('/')[-1],image_1)
#     # cv2.imwrite(save_img_path+'Int2_'+ img_name.split('/')[-1],image_2)
#     # cv2.imwrite(save_img_path+'Int3_'+ img_name.split('/')[-1],image_3)
#     # # cv2.imwrite(save_img_path+'Rot4_'+ img_name.split('/')[-1],image_90)
#     # # cv2.imwrite(save_img_path+'Rot5_'+ img_name.split('/')[-1],image_180)
#     # # cv2.imwrite(save_img_path+'Rot6_'+ img_name.split('/')[-1],image_270)
#     #
#     # # scipy.toimage(image_0, cmin=0, cmax=255).save('data/Intensity0/'+ img_name.split('/')[-1])
#     # #
#     # # # scipy.misc.imsave(traindir_mask_align + img_name.split('/')[-1],label_align)
#     # # # print('save file: '+ img_name.split('/')[-1])
#     # # print(image_0.max(),image_1.max(),image_2.max(),image_3.max(),image_0.min(),image_1.min(),image_2.min(),image_3.min())
#     # # # print(sum(sum(label - image)),sum(sum(label - label_align)))
#     # # # print(image.shape, label.shape)
#     # # f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(8,8))
#     # # ax1.imshow(image_0, cmap='gray',vmin = 0,vmax = 255)
#     # # ax2.imshow(image_1, cmap='gray',vmin = 0,vmax = 255)
#     # # ax3.imshow(image_2, cmap='gray',vmin = 0,vmax = 255)
#     # # ax4.imshow(image_3, cmap='gray',vmin = 0,vmax = 255)
#     # # # ax1.imshow(image_0)
#     # # # ax2.imshow(image_1)
#     # # # ax3.imshow(image_2)
#     # # # ax4.imshow(image_3)
#     # # plt.show()
#     # # plt.savefig('check/check_'+ img_name.split('/')[-1],bbox_inches='tight')
