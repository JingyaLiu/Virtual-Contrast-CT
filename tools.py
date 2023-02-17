import os,cv2,numbers,random,math
import torch
import torch.nn as nn
import numpy as np
import skimage, math
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, datasets, models
# Ignore warnings
import warnings
from PIL import Image
import functionalCV as F
warnings.filterwarnings("ignore")

############################################################################ data processing   ############################################################################
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

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

        frames_l=np.zeros((c,new_h, new_w))
        for c_i in range(c):
            new_image_l=transform.resize(label[c_i,:,:],(new_h, new_w))
            frames_l [c_i,:,:]=new_image_l
        # print('Rescale', frames.max(), frames.min())
        return {'image': frames, 'label': frames_l}


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
            label  = np.fliplr(label)
        # print('RandomHorizontalFlip', image.max(), image.min())

        return {'image': image, 'label': label}


def get_featuremap(A, B):
    diff = abs(A-B)
    featuremap = diff.mean(1)
    return featuremap[0].detach().cpu().numpy()



def rotate(
    img,  #image matrix
    angle #angle of rotation
    ):
    channel = img.shape[0]
    height = img.shape[1]
    width = img.shape[2]

    if angle%180 == 0:
        scale = 1
    elif angle%90 == 0:
        scale = float(max(height, width))/min(height, width)
    else:
        scale = math.sqrt(pow(height,2)+pow(width,2))/min(height, width)
    rotateImg = np.zeros(img.shape)
    #print 'scale %f\n' %scale
    for c_i in range(channel):
        rotateMat = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)
        rotate_temp = cv2.warpAffine(img[c_i,:,:], rotateMat, (width, height))
        rotateImg[c_i,:,:] = rotate_temp
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
        label = rotate(label, angle)
        # rotated = [img.rotate(angle) for img in clip]
        # print('Rotation after', image.max(), image.min())

        return {'image': image, 'label': label}



class Normalize():
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    # def __init__(self, meani, stdi, meanl, stdl):
    #     self.meani = meani
    #     self.stdi = stdi
    #     self.meanl = meanl
    #     self.stdl = stdl

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        image, label = sample['image'], sample['label']
        # print('Before Normalize', image.max(), image.min())
        image = (image / 255.0) * 2 - 1
        label = (label / 255.0) * 2 - 1
        # print('After Normalize', image.max(), image.min())

        return {'image': image, 'label': label}
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

        c, h, w = image.shape
        new_h, new_w = self.output_size

        # top = np.random.randint(25, h - new_h-25)
        # left = np.random.randint(25, w - new_w-25)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # print(top,left)
        image = image[:,top: top + new_h,
                      left: left + new_w]
        label = label[:,top: top + new_h,
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
        'label':np.reshape(sample['label'],sample['label'].shape+(1,))}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # if len(image.shape) > 3:
        #     image = image.transpose((0, 3, 1, 2))
        #     label = label.transpose((2, 0, 1))
        # else:
        #     # swap color axis because
        #     # numpy image: H x W x C
        #     # torch image: C X H X W
        #     image = image.transpose((2, 0, 1))
        #     label = label.transpose((2, 0, 1))
        # print('ToTensor', image.max, image.min)
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}


class ConstrastCTDataset_3c(Dataset):
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
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        image = np.zeros((3,512,512))
        label = np.zeros((3,512,512))
        length =  len(os.listdir(self.image_dir))
        img_name = os.path.join(self.image_dir, os.listdir(self.image_dir)[idx])

        # if idx < len(os.listdir(self.image_dir))-3:
            # print(os.listdir(self.image_dir))
            # print(sorted(os.listdir(self.image_dir)))
        # start
        img_id = os.listdir(self.image_dir)[idx]
        img_name = os.path.join(self.image_dir, img_id)
        image[0,:,:] = io.imread(img_name)[:512,:512]
        label_name = os.path.join(self.label_dir, img_id)
        label[0,:,:] = io.imread(label_name)[:512,:512]

        # img_id = os.listdir(self.image_dir)[idx].split('_')[1] # for test
        # # print(img_id)
        # img_name = os.path.join(self.image_dir, 'pre_'+ img_id)
        # image[0,:,:] = io.imread(img_name)[:512,:512]
        # label_name = os.path.join(self.label_dir, 'contrast_'+img_id)
        # label[0,:,:] = io.imread(label_name)[:512,:512]
        # second & third channel
        for i in range(2):
            index_img = img_id.split('.')[0][9:]
            index_id = int(img_id.split('.')[0].split('_')[0][2:])
            img_name_second = os.path.join(self.image_dir, 'CT'+str(index_id+i+1).zfill(6)+ '_'+index_img+'.png')
            # print(index_img,index_id)
            label_name_second = os.path.join(self.label_dir, 'CT'+str(index_id+i+1).zfill(6)+ '_'+ index_img+'.png')
            # print(img_name_second,label_name_second)
            if os.path.exists(img_name_second):
                # print('here')
                # image[1+i,:,:] = io.imread(img_name_second)[:512,:512]
                # label[1+i,:,:] = io.imread(label_name_second)[:512,:512]
                image[1+i,:,:] = image[i,:,:]
                label[1+i,:,:] = label[i,:,:]
            else:
                # if not exist repeat the previous channel
                image[1+i,:,:] = image[i,:,:]
                label[1+i,:,:] = label[i,:,:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        image, label = sample['image'], sample['label']
        return image, label, img_id
