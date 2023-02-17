from __future__ import print_function, division
import os,cv2,numbers,random,math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import skimage, math
from skimage import io, transform
import matplotlib.pyplot as plt
from utils import get_ids
# Ignore warnings
import warnings
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import scipy
import scipy.misc
import math
from sklearn.metrics import mean_squared_error


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
        img_name = os.path.join(self.image_dir, os.listdir(self.image_dir)[idx])
        image = scipy.misc.imread(img_name, flatten=True).astype(numpy.float32)
        label_name = os.path.join(self.label_dir, os.listdir(self.image_dir)[idx])
        label = scipy.misc.imread(label_name, flatten=True).astype(numpy.float32)
        # sample = {'image': image, 'label': label}

        # if self.transform:
        #     sample = self.transform(sample)
        # image, label = sample['image'], sample['label']

        return image, label, img_name
import numpy
import scipy.signal
import scipy.ndimage

def vifp_mscale(ref, dist):
    sigma_nsq=2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2**(4-scale+1) + 1
        sd = N/5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq<eps] = 0
        sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
        sigma1_sq[sigma1_sq<eps] = 0

        g[sigma2_sq<eps] = 0
        sv_sq[sigma2_sq<eps] = 0

        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=eps] = eps

        num += numpy.sum(numpy.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += numpy.sum(numpy.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num/den

    return vifp

from scipy.ndimage import gaussian_filter

from numpy.lib.stride_tricks import as_strided as ast


def block_view(A, block=(3, 3)):
    """Provide a 2D block view to 2D array. No error checking made.
    Therefore meaningful (as implemented) only for blocks strictly
    compatible with the shape of A."""
    # simple shape and strides computations may seem at first strange
    # unless one is able to recognize the 'tuple additions' involved ;-)
    shape = (A.shape[0]/ block[0], A.shape[1]/ block[1])+ block
    strides = (block[0]* A.strides[0], block[1]* A.strides[1])+ A.strides
    return ast(A, shape= shape, strides= strides)


def ssim(img1, img2, C1=0.01**2, C2=0.03**2):

    bimg1 = block_view(img1, (4,4))
    bimg2 = block_view(img2, (4,4))
    s1  = numpy.sum(bimg1, (-1, -2))
    s2  = numpy.sum(bimg2, (-1, -2))
    ss  = numpy.sum(bimg1*bimg1, (-1, -2)) + numpy.sum(bimg2*bimg2, (-1, -2))
    s12 = numpy.sum(bimg1*bimg2, (-1, -2))

    vari = ss - s1*s1 - s2*s2
    covar = s12 - s1*s2

    ssim_map =  (2*s1*s2 + C1) * (2*covar + C2) / ((s1*s1 + s2*s2 + C1) * (vari + C2))
    return numpy.mean(ssim_map)

# FIXME there seems to be a problem with this code
def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    # print(sigma1_sq[1:10,:])
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    ssim_map = ssim_num / ssim_den
    C3 = C2/2
    l = (2 * mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)
    c = (2 * (sigma1_sq)*(sigma2_sq) + C2)/(sigma1_sq + sigma2_sq + C2)
    s = (sigma12+ C3)/((sigma1_sq)*(sigma2_sq) + C3)

    return numpy.mean(ssim_map), numpy.mean(l), numpy.mean(c), numpy.mean(s)


import numpy
import math

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)), mse

def mse(img1, img2):
    mse = mean_squared_error(img1, img2)/(img1.shape[0]*img1.shape[1])
    return mse
# 
def snu(img1, img2):
    snu_val = np.abs((np.max(img1)-np.min(img1))/np.mean(img1) - (np.max(img2)- np.min(img2))/np.mean(img2))
    # print(snu_val)
    return snu_val
    
quality_values = []
size_values = []
vifp_values = []
ssim_values = []
l_values = []
c_values = []
s_values = []
psnr_values = []
mse_values = []
niqe_values = []
snu_values = []
reco_values = []

# testdir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/preprocess/data/test/'
# testdir_img = '/home/tensor-server/Documents/jingya/Unet/pytorch-unet/result/result/UNet-results_woinitial/CP_16_withouttinitssssssssssssssssssssial50/'
# testdir_img = '/home/tensor-server/Documents/jingya/Unet/pytorch-unet/result/result/Unet_GAN_C_32_C128_BCE_10timesalign_unet4_pix_150G150/'
# testdir_img = '/home/tensor-server/Documents/jingya/Unet/pytorch-unet/result/result/Hr-C_32_C128_BCE_HR_pix_150G150/CT_HR/'
# testdir_img = '/home/tensor-server/Documents/jingya/Unet/pytorch-unet/result/result/Hr-C_32_C128_BCE_HR_pix_150G150/CT_finetune96/'
# testdir_img = '/home/tensor-server/Documents/jingya/Unet/pytorch-unet/result/result/Hr-C_32_C128_BCE_HR_pix_150G150/CT_pretrain/'
# testdir_img = '/home/tensor-server/Documents/jingya/Unet/pytorch-unet/result/C_32_BCE_HR_pix_G_3C_adddreg_256G120'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_portal_stage_C_32_BCE_HRl5_pix_G_3C_adddreg_s256c224_tw_norm40/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_late_stage_C_32_BCE_HRl5_pix_G_3C_adddreg_s256c224_tw_norm40/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/earlystage_C_32_BCE_HRl5_pix_G_3C_adddreg_s256c224_tw_40/psnr_evaluation/'
# testdir_mask = '/media/tensor-server/JYLDisk1_6T/dataset/AutoDye/FOUR_STAGES/EARLY_TEST_LABEL_NEW/'
# testdir_mask = '/media/tensor-server/JYLDisk1_6T/dataset/AutoDye/FOUR_STAGES/PORTAL_TEST_LABEL_NEW/'
testdir_mask = '/media/tensor-server/JYLDisk1_6T/dataset/AutoDye/FOUR_STAGES/LATE_TEST_LABEL_NEW/'

# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_late_stage_G_wt_Gonly_150/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_portal_stage_G_wt_Gonly_150/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_early_stage_G_wt_Gonly_150/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_early_stage_G_wo_perceptual_50/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_portal_stage_G_wo_perceptual_50/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_late_stage_G_wo_perceptual_50/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_early_stage_G_wo_two_path_50/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_portal_stage_G_wo_two_path_50/psnr_evaluation'
testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_late_stage_G_wo_two_path_50/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk1_6T/code/contrastive-unpaired-translation/results/portal_CUT_train_lr_step/test_latest/images/fake_B/'
# testdir_mask = '/media/tensor-server/JYLDisk1_6T/code/contrastive-unpaired-translation/results/portal_CUT_train_lr_step/test_latest/images/real_B/'



# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_early_stage_G_wt_Gonly_150/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_portal_stage_G_wt_Gonly_150/psnr_evaluation'
# testdir_img = '/media/tensor-server/JYLDisk3_4T/jingya/Unet/pytorch-unet/result/model/final_late_stage_G_wt_Gonly_150/psnr_evaluation'



# save_img_path = 'check2/'
# os.mkdir(save_img_path)
train_set = ConstrastCTDataset(image_dir = testdir_img, label_dir=testdir_mask)


# print(train_set)
for inputs, labels, img_name in train_set:
    # print(inputs.size(), labels.size())
    ref, dist = inputs, labels
    vifp_values.append( vifp_mscale(ref, dist) )
    ssim, l, c, s = ssim_exact(ref/255, dist/255)
    ssim_values.append(ssim)
    l_values.append(l)
    c_values.append(c)
    s_values.append(s)
    psnr_values.append( psnr(ref, dist)[0] )
    mse_values.append(mse(ref, dist))
    snu_values.append(snu(ref,dist))
    print(len(vifp_values))

    # niqe_values.append( niqe.niqe(dist/255) )
    # reco_values.append( reco.reco(ref/255, dist/255) )

def meanv(list1):
    return np.mean(list1), np.std(list1)

print("psnr,ssim, c, s,  mse, snu", meanv(psnr_values),meanv(ssim_values),meanv(c_values),meanv(s_values), meanv(mse_values), meanv(snu_values))
