from __future__ import print_function, division
import os,cv2,numbers,random,math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import skimage, math
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
# Ignore warnings
import warnings
from PIL import Image
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from unet import UNet,UNet_4,UNet_HR, UNet_HRPXP,discriminator,UNet_HRPXP_up,discriminator_3d,discriminator_3c,UNet_HRPXP_up_SR, UNet_HRPXP_up_5l,UNet_HRPXP_up_pretrain, discriminator_light
import functionalCV as F
import pytorch_ssim
from sklearn.preprocessing import minmax_scale
from torch.autograd import Variable
from tools import *
warnings.filterwarnings("ignore")

############################################################################ train   ############################################################################
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def train_model(model, netD, netPerc, stage, criterion_GAN,criterion_pixelwise,optimizerG, optimizerD, exp_lr_schedulerG,exp_lr_schedulerD,
                       num_epochs=60, gpu = True, patch_on = True, perc = False):
    LOSS,val_loss = [],[]


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs -  1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                exp_lr_schedulerG.step()
                exp_lr_schedulerD.step()
                model.train()  # Set model to training mode
                netD.train()
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            lambda_pixel = 0.1
            lambda_percep = 1
            beta_real = 1
            i = 0

            # Iterate over data.
            for inputs, labels,img_name in dataloaders[phase]:
                # patch = ()
                i += 1
                optimizerG.zero_grad()
                optimizerD.zero_grad()
                image, label = inputs[0],labels[0]
                inputs_local = inputs
                if gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                with torch.set_grad_enabled(phase == 'train'):
                    fake_B,x_up4,x_up3,x_up2,x4 = model(inputs.float())
                    # print(fake_B.size())
                    if patch_on is True:
                        size_fake_B = fake_B.size()[-1]
                        # print(size_fake_B//2)
                        fake_LB_p1,x_LB_up4,x_LB_up3,x_LB_up2,x4_LB = model(inputs_local[:,:,0:size_fake_B//2,0:size_fake_B//2].cuda().float())
                        fake_LB_p2,x_LB_up4,x_LB_up3,x_LB_up2,x4_LB = model(inputs_local[:,:,size_fake_B//2:size_fake_B,0:size_fake_B//2].cuda().float())
                        fake_LB_p3,x_LB_up4,x_LB_up3,x_LB_up2,x4_LB = model(inputs_local[:,:,0:size_fake_B//2,size_fake_B//2:size_fake_B].cuda().float())
                        fake_LB_p4,x_LB_up4,x_LB_up3,x_LB_up2,x4_LB = model(inputs_local[:,:,size_fake_B//2:size_fake_B,size_fake_B//2:size_fake_B].cuda().float())
                        # print('check local patch size:', fake_LB_p1.size(),fake_LB_p2.size(),fake_LB_p3.size(),fake_LB_p4.size())
                        fake_LB = np.zeros(fake_B.size())
                        # print(fake_LB,fake_LB_p1.size())
                        fake_LB[:,:,0:size_fake_B//2,0:size_fake_B//2] = fake_LB_p1.cpu().detach().numpy()
                        fake_LB[:,:,size_fake_B//2:size_fake_B,0:size_fake_B//2] = fake_LB_p2.cpu().detach().numpy()
                        fake_LB[:,:,0:size_fake_B//2,size_fake_B//2:size_fake_B] = fake_LB_p3.cpu().detach().numpy()
                        fake_LB[:,:,size_fake_B//2:size_fake_B,size_fake_B//2:size_fake_B] = fake_LB_p4.cpu().detach().numpy()
                        fake_LB = torch.from_numpy(fake_LB).cuda()

                    # add perceptual_loss
                    pred_fake = netD(fake_B.float(), inputs.float()).view(-1,1).squeeze()
                    real_B,real_B_up4,real_B_up3,real_B_up2,real_B4 = netPerc(labels.float())
                    perc_B,perc_B_up4,perc_B_up3,perc_B_up2,perc_B4 = netPerc(fake_B.float())
                    loss_GAN = criterion_GAN(pred_fake, Variable(torch.ones(pred_fake.size()).cuda()))+ beta_real*criterion_pixelwise(pred_fake, Variable(torch.ones(pred_fake.size()).cuda()))
                    loss_pixel = criterion_pixelwise(fake_B.float(), labels.float())
                    if perc is True:
                        perceptual_loss = 0.3*(criterion_pixelwise(perc_B,real_B) +criterion_pixelwise(perc_B_up4,real_B_up4)+criterion_pixelwise(real_B_up2,perc_B_up2))+0.7*(criterion_pixelwise(perc_B_up3,real_B_up3)+criterion_pixelwise(real_B4,perc_B4))
                        featuremap1, featuremap2, featuremap3, featuremap4 = get_featuremap(perc_B_up4,real_B_up4), get_featuremap(perc_B_up3,real_B_up3), get_featuremap(real_B_up2,perc_B_up2), get_featuremap(real_B4,perc_B4)
                        # plot the middle result -> perceptual map                    
                        # f,(ax1,ax2,ax3,ax4, ax5, ax6) = plt.subplots(1,6,figsize=(40,10))
                        # #
                        # ax1.imshow(featuremap1)
                        # plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
                        # ax2.imshow(featuremap2)
                        # plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
                        # ax3.imshow(featuremap3) #
                        # plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
                        # ax4.imshow(featuremap4)
                        # plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
                        # ax5.imshow(labels[0,0].detach().cpu().numpy())
                        # plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)

                        # ax6.imshow(fake_B[0,0].detach().cpu().numpy())
                        # plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
                        # # ax1.text(0, 0, 'Pre-contrast CT', fontsize=12 ,color = 'red')
                        # # ax2.text(0, 0, 'Real Contrast CT', fontsize=12,color = 'red')
                        # # ax3.text(0, 0, 'Virtual Contrast CT', fontsize=12,color = 'red')
                        # # ax4.text(0, 0, 'Virtual Contrast CT+lastlayer', fontsize=12,color = 'red')
                        
                        # # plt.savefig('result/'+ path + '/'+img_name[0])
                        
                        # plt.show()
                        loss_G_main = loss_GAN/2 + lambda_pixel * loss_pixel + lambda_percep*perceptual_loss
                    else:
                        loss_G_main = loss_GAN/2 + lambda_pixel * loss_pixel





                    if patch_on is True:
                        real_LB,real_LB_up4,real_LB_up3,real_LB_up2,real_LB4 = netPerc(labels.float())
                        perc_LB,perc_LB_up4,perc_LB_up3,perc_LB_up2,perc_LB4 = netPerc(fake_LB.float())
                        pred_fake_L = netD(fake_LB.float(), inputs.float()).view(-1,1).squeeze()
                        loss_GAN_L = criterion_GAN(pred_fake_L, Variable(torch.ones(pred_fake_L.size()).cuda()))+ beta_real*criterion_pixelwise(pred_fake_L, Variable(torch.ones(pred_fake_L.size()).cuda()))
                        loss_pixel_L = criterion_pixelwise(fake_LB.float(), labels.float())
                    if perc is True:
                        perceptual_loss_L = 0.3*(criterion_pixelwise(perc_LB,real_LB) +criterion_pixelwise(perc_LB_up4,real_LB_up4)+criterion_pixelwise(real_LB_up2,perc_LB_up2))+0.7*(criterion_pixelwise(perc_LB_up3,real_LB_up3)+criterion_pixelwise(real_LB4,perc_LB4))
                        featuremap1, featuremap2, featuremap3, featuremap4 = get_featuremap(perc_LB_up4,real_LB_up4), get_featuremap(perc_LB_up3,real_LB_up3), get_featuremap(real_LB_up2,perc_LB_up2), get_featuremap(real_B4,perc_B4)
                    
                        # plot the middle result -> perceptual map                    
                        # f,(ax1,ax2,ax3,ax4) = plt.subplots(1,4,figsize=(40,10))
                        # #
                        # ax1.imshow(featuremap1)
                        # plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
                        # ax2.imshow(featuremap2)
                        # plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
                        # ax3.imshow(featuremap3) #
                        # plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
                        # ax4.imshow(featuremap4)
                        # plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)

                        # # ax1.text(0, 0, 'Pre-contrast CT', fontsize=12 ,color = 'red')
                        # # ax2.text(0, 0, 'Real Contrast CT', fontsize=12,color = 'red')
                        # # ax3.text(0, 0, 'Virtual Contrast CT', fontsize=12,color = 'red')
                        # # ax4.text(0, 0, 'Virtual Contrast CT+lastlayer', fontsize=12,color = 'red')
                        
                        # # plt.savefig('result/'+ path + '/'+img_name[0])
                        
                        # plt.show()
                        
                        loss_G_main_L = loss_GAN_L/2 + lambda_pixel * loss_pixel_L + lambda_percep*perceptual_loss_L
                    else:
                        loss_G_main_L = loss_GAN_L/2 + lambda_pixel * loss_pixel_L

                    if patch_on is True:
                        loss_G =loss_G_main*0.7+loss_G_main_L*0.3#+loss_G_patch_1)
                    else:
                        loss_G = loss_G_main

                    if phase == 'train':
                        loss_G.backward(retain_graph=True)
                        optimizerG.step()

                    # Real loss
                    pred_real = netD(labels.float(), inputs.float()).view(-1,1).squeeze()
                    loss_real = criterion_GAN(pred_real, Variable(torch.ones(pred_real.size()).cuda()))+ beta_real*criterion_pixelwise(pred_real, Variable(torch.ones(pred_real.size()).cuda()))

                    # Fake loss
                    pred_fake = netD(fake_B.float(), inputs.float()).view(-1,1).squeeze()
                    loss_fake = criterion_GAN(pred_fake, Variable(torch.zeros(pred_fake.size()).cuda()))+ beta_real*criterion_pixelwise(pred_fake, Variable(torch.zeros(pred_fake.size()).cuda()))

                    # Total loss
                    loss_D = loss_real/2 + loss_fake/2
                    # print(loss_real, loss_fake, loss_D)
                    if phase == 'train':
                        loss_D.backward(retain_graph=True)
                        optimizerD.step()
                    print('[%d/%d][%d/%d] late phase Loss_D: %.4f Loss_G: %.4f loss_pixel: %.4f loss_GAN: %.4f'
                          % (epoch, num_epochs, i, dataset_sizes[phase]/batch_size,
                             loss_D.item(),
                             loss_G.item(),
                             loss_pixel.item(),
                             loss_GAN.item()/2))
                running_loss += loss_G.item() * inputs.size(0)
                running_corrects += torch.sum(fake_B == labels.data.float())

            # epoch_acc = running_corrects.double() / (dataset_sizes[phase]*inputs.shape[2]*inputs.shape[2])
            epoch_loss = running_loss / (dataset_sizes[phase])
            print(epoch_loss)
            if phase == 'train':
                LOSS.append(epoch_loss)

            plot_loss(LOSS,val_loss, stage)

        torch.save(model.state_dict(), 'model/final_{}_stage_G_final_{}.pth'.format(stage,epoch + 1))
        torch.save(netD.state_dict(),  'model/final_{}_stage_D_final_{}.pth'.format(stage,epoch + 1))
    return model,netD



#############################              plot loss    ##############################################
import matplotlib.pyplot as plt
def plot_loss(train_list, val_list, stage):

    fig, ax1 = plt.subplots()
    ax1.plot(range(len(train_list)),train_list,'k')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Train loss', color='g')
    plt.savefig('loss_figs/final_{}_stage_final.png'.format(stage),bbox_inches='tight')


############################################################################ load data   ############################################################################

if __name__ == "__main__":

    #set the path  the images and labels for each stage
    stage = 'early'
    traindir_img = '/media/tensor-server/JYLDisk3_4T/Dataset/AutoDye/FOUR_STAGES/EARLY_TRAIN/'
    traindir_mask = '/media/tensor-server/JYLDisk3_4T/Dataset/AutoDye/FOUR_STAGES/EARLY_TRAIN_LABEL_NEW/'
    valdir_img = '/media/tensor-server/JYLDisk3_4T/Dataset/AutoDye/FOUR_STAGES/EARLY_TEST/'
    valdir_mask = '/media/tensor-server/JYLDisk3_4T/Dataset/AutoDye/FOUR_STAGES/EARLY_TEST_LABEL_NEW/'
    testdir_img = '/media/tensor-server/JYLDisk3_4T/Dataset/AutoDye/FOUR_STAGES/EARLY_TEST/'
    testdir_mask = '/media/tensor-server/JYLDisk3_4T/Dataset/AutoDye/FOUR_STAGES/EARLY_TEST_LABEL_NEW/'
    batch_size = 1


    print('load train data')
    train_set = ConstrastCTDataset_3c(image_dir = traindir_img,
                            label_dir = traindir_mask,
                            transform = transforms.Compose([
                                                Rescale(224),
                                                # RandomCrop(96),
                                                #    RandomCrop(224),
                                                #    RandomHorizontalFlip(),
                                                #    Rotation([0,5]),
                                                Normalize(),
                                                # Reshape(),
                                                ToTensor()

                            ]))



    print('load val data')
    val_set = ConstrastCTDataset_3c(image_dir = valdir_img,
                            label_dir = valdir_mask,
                            transform = transforms.Compose([
                                                Rescale(224),
                                                # RandomCrop(96),
                                                #    RandomCrop(224),
                                                #    RandomHorizontalFlip(),
                                                #    Rotation([0,5]),
                                                Normalize(),
                                                # Reshape(),
                                                ToTensor()
                            ]))

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1)
    }

    image_datasets = {
        'train': train_set, 'val': val_set
    }
    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }


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

    testloader = torch.utils.data.DataLoader(test_set, batch_size=1,shuffle=False, num_workers=0)



    # # # # ##################################################        training         #####################################################################

    import torch.backends.cudnn as cudnn
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    
    # load the pretrained model
    path_G_perc = 'model/pretrained_intensity_model_4level_3c6.pth'
    model = UNet_HRPXP_up_5l(n_channels=3, n_init=32).to(device)
    model = model.cuda()
    model.load_state_dict(torch.load(path_G_perc),strict= False)
    model= torch.nn.DataParallel(model)

    netD = discriminator_light(ndf = 64).to(device)
    netD = netD.cuda()

    # print(device)
    netPerc = UNet_HRPXP_up_pretrain(n_channels=3, n_init=32).to(device)
    netPerc = netPerc.cuda()
    netPerc.load_state_dict(torch.load(path_G_perc),strict= False)

    criterion_pixelwise = torch.nn.MSELoss()
    criterion_GAN = torch.nn.BCEWithLogitsLoss()

    lr=5e-5

    for param in model.parameters():
        param.requires_grad = True

    optimizerG = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    exp_lr_schedulerG = lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.1)
    exp_lr_schedulerD = lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.1)

    model,netD = train_model(model, netD, netPerc,stage, criterion_GAN, criterion_pixelwise, optimizerG, optimizerD, exp_lr_schedulerG, exp_lr_schedulerD,
                        num_epochs=50, gpu = True, patch_on = True, perc = True)

