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
from utils import get_ids
# Ignore warnings
import warnings
from PIL import Image
from torchsummary import summary
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from unet import UNet,UNet_4,UNet_HR, UNet_HRPXP,UNet_4_pretrain,UNet_HRPXP_up_pretrain, UNet_HRPXP_up_5l_pretrain
import functionalCV as F
# import pytorch_ssim
from sklearn.preprocessing import minmax_scale

warnings.filterwarnings("ignore")

############################################################################ data processing   ############################################################################

# class Rescale(object):
#     """Rescale the image in a sample to a given size.
#
#     Args:
#         output_size (tuple or int): Desired output size. If tuple, output is
#             matched to output_size. If int, smaller of image edges is matched
#             to output_size keeping aspect ratio the same.
#     """
#
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#
#         h, w = image.shape[:2]
#         if isinstance(self.output_size, int):
#             if h > w:
#                 new_h, new_w = self.output_size * h / w, self.output_size
#             else:
#                 new_h, new_w = self.output_size, self.output_size * w / h
#         else:
#             new_h, new_w = self.output_size
#
#         new_h, new_w = int(new_h), int(new_w)
#         img = F.resize(image, (new_h, new_w))
#         # print('Rescale', image.max, image.min)
#         return {'image': img, 'label': label}

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        c, h, w = image.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        # print(c, new_h, new_w,c,h,w)
        # img = F.resize(image, (c, new_h, new_w))
        # label = F.resize(label, (c. new_h, new_w))

        frames=np.zeros((c,new_h, new_w))
        for c_i in range(c):
            new_image=transform.resize(image[c_i,:,:],(new_h, new_w))
            frames[c_i,:,:]=new_image

        # frames_l=np.zeros((c,new_h, new_w))
        # for c_i in range(c):
        #     new_image_l=transform.resize(label[c_i,:,:],(new_h, new_w))
        #     frames_l [c_i,:,:]=new_image_l
        # print('Rescale', frames.max(), frames.min())
        return {'image': frames, 'label': label}




class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly
    with a probability 0.5
    """

    def __call__(self, sample):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        image, label = sample['image'], sample['label']

        if random.random() < 0.5:
            image  = np.fliplr(image)
        # print('RandomHorizontalFlip', image.max(), image.min())

        return {'image': image, 'label': label}

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


class Rotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # print('Rotation before', image.max(), image.min())

        angle = random.uniform(self.degrees[0], self.degrees[1])
        # print(angle)
        image = rotate(image, angle)
        # rotated = [img.rotate(angle) for img in clip]
        # print('Rotation after', image.max(), image.min())

        return {'image': image, 'label': label}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, meani, stdi, meanl, stdl):
        self.meani = meani
        self.stdi = stdi
        self.meanl = meanl
        self.stdl = stdl

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        image, label = sample['image'], sample['label']
        # print('Normalize', image.max(), image.min())

        return {'image': F.normalize(image, self.meani, self.stdi),
                'label': label}
    # def __repr__(self):
    #     return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)




class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(60, h - new_h-60)
        left = np.random.randint(60, w - new_w-60)

        image = image[top: top + new_h,
                      left: left + new_w]

        # print('crop', image.max(), image.min())
        return {'image': image, 'label': label}


class TestPatchCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        patches = int(h/new_h * w/new_w)
        imagenew = np.zeros((patches, new_h, new_w))
        top, left, i = 0, 0, 0
        # print(image[top: top + new_h, left: left + new_w].shape, imagenew[0,:,:].shape)
        for j in range(int(h/new_h)):
            for k in range(int(w/new_w)):
                imagenew[i,:,:] = image[top:top + new_h, left:left + new_w]
                top =  top + new_h
                i = i +1
            top, left = 0 ,left + new_w
        # print('crop', image.max(), image.min())
        return {'image': imagenew, 'label': label}

class Reshape(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        return {'image':np.reshape(sample['image'],sample['image'].shape+(1,)),
        'label':sample['label']}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # if len(image.shape) > 3:
        #     image = image.transpose((0, 3, 1, 2))
        # else:
        #     # swap color axis because
        #     # numpy image: H x W x C
        #     # torch image: C X H X W
        #     image = image.transpose((2, 0, 1))
        # print('ToTensor', image.max, image.min)
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(np.array(label))}


class ConstrastCTDataset(Dataset):
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
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        # label_dir = ['Intensity0','Intensity1','Intensity2','Intensity3']
        # for dirs in label_dir:
        # image_dir = os.path.join(self.image_dir,dirs)
        # print(os.listdir)
        img_name = os.path.join(self.image_dir, os.listdir(self.image_dir)[idx])
        image = io.imread(img_name)/255
        # print(img_name)
        label = int(img_name.split('/')[-1][9]) # intensity 4
        # label = int(img_name.split('/')[-1][3]) #intensity 7

        image_new = np.zeros((3,512,512))
        for i in range(3):
            image_new[i,:,:] = image[:512,:512]

        sample = {'image': image_new, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        image, label = sample['image'], sample['label']
        # print(img_name,image.shape,label.shape)
        return image, label

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram
############################################################################ load data   ############################################################################

#load the images and labels
traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/4levelintensity/'
# traindir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/NLST_2/7level/'

# traindir_mask = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/preprocess/data/ltrain_label_early_align/'
# traindir_img = 'check1/'
print('load train data')
train_set = ConstrastCTDataset(image_dir = traindir_img,
                          # label_dir = traindir_mask,
                          transform = transforms.Compose([
                                               # RandomCrop(128),
                                               Rescale(256),
                                               # RandomHorizontalFlip(),
                                               # Rotation([0,5]),
                                               # Reshape(),
                                               ToTensor(),
                                               # Normalize([im_meanstd[0]], [im_meanstd[1]],[lb_meanstd[0]],[lb_meanstd[1]])

                          ]))

# testdir_img = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/preprocess/data/test/'
# testdir_mask = '/media/tensor-server/feff45c0-e805-4f15-9cff-6b81c2dd1460/AutoDye/preprocess/data/ltest_label_early_align/'
# testdir_img = 'check_test/'
# print('load test data')
# test_set = ConstrastCTDataset(image_dir = testdir_img,
#                           # label_dir = testdir_mask,
#                           transform = transforms.Compose([
#                                                # TestPatchCrop(256),
#                                                # RandomHorizontalFlip(),
#                                                # Rotation([0,5]),
#                                                # Reshape(),
#                                                ToTensor(),
#                                                # Normalize([im_meanstd[0]], [im_meanstd[1]],[lb_meanstd[0]],[lb_meanstd[1]])
#                             ]))

# testloader = torch.utils.data.DataLoader(test_set, batch_size=1,shuffle=False, num_workers=0)




image_datasets = {
    'train': train_set
}

batch_size = 4

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1),
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}


def train_model(model, criterion, optimizer, scheduler, num_epochs, gpu = True):
    LOSS,val_loss = [],[]
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()

                # forward
                if gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                with torch.set_grad_enabled(phase == 'train'):

                    ############################## unet HRnet  ########################################


                    outputs = model(inputs.float())

                    ################################        loss     ###############################
                    # print(outputs.shape,labels.shape)
                    loss = criterion(outputs, labels)


                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()
                    print('loss: %.4f' %(loss.item()))
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(outputs == labels.data.float())

            epoch_loss = running_loss / (dataset_sizes[phase])#*inputs.shape[2]*inputs.shape[2])
            # epoch_acc = running_corrects.double() / (dataset_sizes[phase]*inputs.shape[2]*inputs.shape[2])

            print('{} Loss: {:.4f}'.format( phase, epoch_loss))

            if phase == 'train':
                LOSS.append(epoch_loss)
            # deep copy the model
            if phase == 'val': #and epoch_acc >= best_acc:
                # best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
                val_loss.append(epoch_loss)
            plot_loss(LOSS,val_loss)
            if epoch%1 == 0:
            # load best model weights
                # model.load_state_dict(best_model_wts)
                torch.save(model.state_dict(), 'model/pretrained_intensity_model_4levelplus_3c{}.pth'.format(epoch + 1))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    # model.load_state_dict(best_model_wts)
    # torch.save(model.state_dict(), 'pretrained_intensity_model_4level_3d{}.pth'.format(epoch + 1))

    return model

#############################              plot loss    ##############################################
import matplotlib.pyplot as plt
def plot_loss(train_list, val_list):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(range(len(train_list)),train_list,'k')
    ax2.plot(range(len(val_list)),val_list, 'b-')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Train loss', color='g')
    ax2.set_ylabel('Val loss', color='b')
    plt.savefig('loss_figs/pretrained_intensity_model_4levelplus3c.png',bbox_inches='tight')

##################################################        training         #####################################################################
# # # #
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# print(Variable(torch.FloatTensor([0,0,0,1])).view(1, -1))
# model = UNet(n_channels=1, n_init=16).to(device)

# path_D = 'pretrained_intensity_model156'
model = UNet_HRPXP_up_5l_pretrain(n_channels=3, n_init=32).to(device)
# model = UNet_HRPXP_up_5l_pretrain(n_channels=3, n_init=32).to(device)
# model.load_state_dict(torch.load(path_D + '.pth', map_location="cuda:0"))

# model = UNet_HRPXP(n_channels=1, n_init=32).to(device)
criterion = nn.CrossEntropyLoss()

# criterion = nn.MSELoss(reduction = 'mean')
# criterion = contextual_loss()
# criterion = nn.L1Loss(reduction = 'mean')
# criterion = nn.SmoothL1Loss(reduction = 'mean')

for param in model.parameters():
    param.requires_grad = True
optimizer_ft =optim.Adam(model.parameters(), lr=1e-6, betas=(0.5, 0.999))
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=95, gamma=0.1)

model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=40, gpu = True)

##################################################        testing  hrnet       #####################################################################

# path = 'model/pretrained_intensity_model1511'
# # os.makedirs('result/'+path+'/')
# device = torch.device("cuda")
# model = UNet_4_pretrain(n_channels=1, n_init=32)
# model.load_state_dict(torch.load(path + '.pth', map_location="cuda:0"))
# model.eval()
# model.cuda()
# #  load test data
# i = 0
# for images,labels in testloader:
#     inputs = images.cuda()
#     # outputs,outputs_add = model(inputs.float())
#     outputs= model(inputs.float())
#     print(outputs.cpu().detach().numpy())
#     # # plot the result
#     # f,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(1,8,figsize=(32,4))
#     # ax1.imshow(images.cpu()[0,0,:,:], cmap='gray')
#     # ax2.imshow(labels[0,0,:,:], cmap='gray')
#     # ax3.imshow(outputs.cpu().detach().numpy()[0,0,:,:], cmap = 'gray')
#     # ax5.imshow(x4.cpu().detach().numpy()[0,0,:,:], cmap = 'gray')
#     # ax6.imshow(x3.cpu().detach().numpy()[0,0,:,:], cmap = 'gray')
#     # ax7.imshow(x2.cpu().detach().numpy()[0,0,:,:], cmap = 'gray')
#     # ax8.imshow(x1.cpu().detach().numpy()[0,0,:,:], cmap = 'gray')
#     # ax4.imshow(x_g2.cpu().detach().numpy()[0,0,:,:], cmap = 'gray')
#
#     # plt.savefig('result/'+ path + '/results' + str(i)+'.png')
#     # plt.show()
#     i = i+1
