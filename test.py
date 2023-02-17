import os,cv2,random
import torch
import torch.nn as nn
import numpy as np
import skimage
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
# Ignore warnings
import warnings
from PIL import Image
from unet import UNet,UNet_4,UNet_HR, UNet_HRPXP,discriminator,UNet_HRPXP_up,discriminator_3d,discriminator_3c,UNet_HRPXP_up_SR, UNet_HRPXP_up_5l,UNet_HRPXP_up_pretrain, discriminator_light
import functionalCV as F
from tools import *
warnings.filterwarnings("ignore")

############################################################################ load data   ############################################################################

if __name__ == "__main__":

    #set the path  the images and labels for each stage
    stage = 'early'
    testdir_img = '/media/tensor-server/JYLDisk3_4T/Dataset/AutoDye/FOUR_STAGES/EARLY_TEST/'
    testdir_mask = '/media/tensor-server/JYLDisk3_4T/Dataset/AutoDye/FOUR_STAGES/EARLY_TEST_LABEL_NEW/'
    batch_size = 1
    print('load test data')
    test_set = ConstrastCTDataset_3c(image_dir = testdir_img,
                            label_dir = testdir_mask,
                            transform = transforms.Compose([
                                                # TestPatchCrop(256),
                                                # RandomHorizontalFlip(),
                                                # Rotation([0,5]),
                                                # Reshape(),
                                                # Rescale(256),
                                                Normalize(),
                                                ToTensor(),
                            ]))

    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False, num_workers=0)



    # ##################################################        testing        ##################################################################

    path = 'model/final_early_stage_G_final_50'
    # path = 'model/final_portal_stage_G_final_50'
    # path = 'model/final_late_stage_G_final_50'

    # original saved file with DataParallel
    device = torch.device("cuda")
    n_channels=3
    model = UNet_HRPXP_up_5l(n_channels, n_init=32)

    # IF USE PARALLEL TRAINING
    state_dict = torch.load(path + '.pth', map_location="cuda:0")

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

    # IF NO PARALLEL TRAINING
    # model.load_state_dict(torch.load(path + '.pth', map_location="cuda:0"))

    model.eval()
    model.cuda()

    def normalize_back(img):
        img = (img + 1) / 2 * 255
        return img

    for images,labels,img_name in testloader:
        print('processing ', img_name)
        inputs = images.cuda()
        outputs,x_up4,x_up3,x_up2,x4= model(inputs.float())
        psize = 16
        img_folder, index_id = img_name[0].split('_')[0], img_name[0].split('.')[0].split('_')[-1]
        evaluate_path_test_compare = 'result/'+ path+'/test_compare/'
        if not os.path.exists(evaluate_path_test_compare):
            os.makedirs(evaluate_path_test_compare)
        img1 = images.cpu()[0,0,:,:]
        img2 = labels[0,0,:,:]
        img3 = outputs.cpu().detach().numpy()[0,0,:,:]#+5*x_up4.cpu().detach().numpy()[0,0,:,:]#o
        # print(img1.max(),img1.min(),img2.max(),img2.min(),img3.max(),img3.min())
        img1 = normalize_back(img1)
        img2 = normalize_back(img2)
        img3 = normalize_back(img3)
        # img1 = transform.resize(np.array(img1),(512,512))
        # img2 = transform.resize(np.array(img2),(512,512))
        # img3 = transform.resize(np.array(img3),(512,512))

        # print(img1.shape,img2.shape,img3.shape)
        # vis_orig = np.concatenate((img1, img2), axis=1)
        # vis_virtual = np.concatenate((img1, img3), axis=1)
        # cv2.imwrite(evaluate_path_orig + index_id + '.png', vis_orig)
        # cv2.imwrite(evaluate_path_virtual + index_id + '.png', vis_virtual)

        vis = np.concatenate((img1, img2, img3), axis=1)
        cv2.imwrite(evaluate_path_test_compare+img_name[0],vis)
        cv2.imwrite(evaluate_path_test_compare+img_name[0],vis)

        # if not os.path.exists('result/'+ path + '/psnr_evaluation/'):
        #     os.makedirs('result/'+ path + '/psnr_evaluation/')
        # cv2.imwrite('result/'+ path + '/psnr_evaluation/' + img_name[0],img3)