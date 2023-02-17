# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_3d(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv_3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x



class inconv_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_3d, self).__init__()
        self.conv = double_conv_3d(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv_k1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_k1, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        x = self.conv(x)
        return x

class inconv_k1_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv_k1_3d, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
    def forward(self, x):
        x = self.conv(x)
        return x



class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class down_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_2, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class down_2_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_2_3d, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv_3d(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class top_1_3(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(top_1_3, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_up(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class top_1_3_3d(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(top_1_3_3d, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_up_3d(in_ch, out_ch)

    def forward(self, x1, x2):
        # print(x1.shape,x2.shape)
        x1 = self.up(x1)


        # # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1[:,:,:1,:,:]], 1)
        # print(x.shape)

        x = self.conv(x)
        # print(x.shape)

        return x


class down_2_3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_2_3, self).__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = double_conv_up(in_ch,out_ch)
    def forward(self, x1, x2):
        x1 = self.mp(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class down_2_3_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_2_3_3d, self).__init__()
        self.mp = nn.MaxPool3d(2)
        self.conv = double_conv_up_3d(in_ch,out_ch)
    def forward(self, x1, x2):
        x1 = self.mp(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class down_3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_3, self).__init__()
        self.mp2 = nn.MaxPool2d(2)
        self.mp4 = nn.MaxPool2d(4)

        self.conv = double_conv_up(in_ch,out_ch)
    def forward(self, x1, x2):
        x1 = self.mp2(x1)
        x2 = self.mp4(x2)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class down_3_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_3_3d, self).__init__()
        self.mp2 = nn.MaxPool3d(2)
        self.mp4 = nn.MaxPool3d(4)

        self.conv = double_conv_up_3d(in_ch,out_ch)
    def forward(self, x1, x2):
        x1 = self.mp2(x1)
        x2 = self.mp4(x2)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class top_1_4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(top_1_4, self).__init__()

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = double_conv_up(in_ch,out_ch)
    def forward(self, x1, x2, x3):
        x1 = self.up4(x1)
        x2 = self.up2(x2)

        # input is CHW
        diffY = x3.size()[2] - x1.size()[2]
        diffX = x3.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # input is CHW
        diffY = x3.size()[2] - x2.size()[2]
        diffX = x3.size()[3] - x2.size()[3]

        x2 = F.pad(x2, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))



        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x


class top_1_4_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(top_1_4_3d, self).__init__()

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.conv = double_conv_up_3d(in_ch,out_ch)
    def forward(self, x1, x2, x3):
        # print(x1.shape, x2.shape, x3.shape)
        x1 = self.up4(x1)
        x2 = self.up2(x2)
        # print(x1.shape, x2.shape, x3.shape)

        # input is CHW
        # diffY = x3.size()[2] - x1.size()[2]
        # diffX = x3.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))
        #
        # # input is CHW
        # diffY = x3.size()[2] - x2.size()[2]
        # diffX = x3.size()[3] - x2.size()[3]
        #
        # x2 = F.pad(x2, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))

        print(x1.shape, x2.shape, x3.shape)

        x = torch.cat([x3, x2[:,:,:1,:,:], x1[:,:,:1,:,:]], dim=1)
        x = self.conv(x)
        return x

class down_2_4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_2_4, self).__init__()
        self.mp2 = nn.MaxPool2d(2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = double_conv_up(in_ch,out_ch)
    def forward(self, x1, x2, x3):
        x1 = self.mp2(x1)
        x2 = self.up2(x2)

        # input is CHW
        diffY = x3.size()[2] - x1.size()[2]
        diffX = x3.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # input is CHW
        diffY = x3.size()[2] - x2.size()[2]
        diffX = x3.size()[3] - x2.size()[3]

        x2 = F.pad(x2, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))



        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x

class down_2_4_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_2_4_3d, self).__init__()
        self.mp2 = nn.MaxPool3d(2)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        # self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = double_conv_up_3d(in_ch,out_ch)
    def forward(self, x1, x2, x3):
        x1 = self.mp2(x1)
        x2 = self.up2(x2)

        # input is CHW
        # diffY = x3.size()[2] - x1.size()[2]
        # diffX = x3.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))
        #
        # # input is CHW
        # diffY = x3.size()[2] - x2.size()[2]
        # diffX = x3.size()[3] - x2.size()[3]
        #
        # x2 = F.pad(x2, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))



        x = torch.cat([x3, x2[:,:,:1,:,:], x1[:,:,:1,:,:]], dim=1)
        x = self.conv(x)
        return x

class down_3_4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_3_4, self).__init__()
        self.mp2 = nn.MaxPool2d(2)
        self.mp4 = nn.MaxPool2d(4)
        # self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = double_conv_up(in_ch,out_ch)
    def forward(self, x1, x2, x3):
        x1 = self.mp4(x1)
        x2 = self.mp2(x2)

        # input is CHW
        diffY = x3.size()[2] - x1.size()[2]
        diffX = x3.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # input is CHW
        diffY = x3.size()[2] - x2.size()[2]
        diffX = x3.size()[3] - x2.size()[3]

        x2 = F.pad(x2, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x

class down_3_4_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_3_4_3d, self).__init__()
        self.mp2 = nn.MaxPool3d(2)
        self.mp4 = nn.MaxPool3d(4)
        # self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = double_conv_up_3d(in_ch,out_ch)
    def forward(self, x1, x2, x3):
        x1 = self.mp4(x1)
        x2 = self.mp2(x2)

        # input is CHW
        # diffY = x3.size()[2] - x1.size()[2]
        # diffX = x3.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))
        #
        # # input is CHW
        # diffY = x3.size()[2] - x2.size()[2]
        # diffX = x3.size()[3] - x2.size()[3]
        #
        # x2 = F.pad(x2, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))

        x = torch.cat([x3, x2[:,:,:1,:,:], x1[:,:,:1,:,:]], dim=1)
        x = self.conv(x)
        return x


class down_4(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_4, self).__init__()
        self.mp2 = nn.MaxPool2d(2)
        self.mp4 = nn.MaxPool2d(4)
        self.mp8 = nn.MaxPool2d(8)
        # self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = double_conv_up(in_ch,out_ch)
    def forward(self, x1, x2, x3):
        x1 = self.mp8(x1)
        x2 = self.mp4(x2)
        x3 = self.mp2(x3)

        # input is CHW
        diffY = x3.size()[2] - x1.size()[2]
        diffX = x3.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # input is CHW
        diffY = x3.size()[2] - x2.size()[2]
        diffX = x3.size()[3] - x2.size()[3]

        x2 = F.pad(x2, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x

class down_4_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_4_3d, self).__init__()
        self.mp2 = nn.MaxPool3d(2)
        self.mp4 = nn.MaxPool3d(4)
        self.mp8 = nn.MaxPool3d(8)
        # self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = double_conv_up_3d(in_ch,out_ch)
    def forward(self, x1, x2, x3):
        x1 = self.mp8(x1)
        x2 = self.mp4(x2)
        x3 = self.mp2(x3)

        # input is CHW
        # diffY = x3.size()[2] - x1.size()[2]
        # diffX = x3.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))
        #
        # # input is CHW
        # diffY = x3.size()[2] - x2.size()[2]
        # diffX = x3.size()[3] - x2.size()[3]
        #
        # x2 = F.pad(x2, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))

        x = torch.cat([x3, x2[:,:,:1,:,:], x1[:,:,:1,:,:]], dim=1)
        x = self.conv(x)
        return x

class top1_5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(top1_5, self).__init__()
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.conv = double_conv_up(in_ch,out_ch)
    def forward(self, x1, x2, x3, x4):
        x1 = self.up8(x1)
        x2 = self.up4(x2)
        x3 = self.up2(x3)

        # input is CHW
        diffY = x4.size()[2] - x1.size()[2]
        diffX = x4.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # input is CHW
        diffY = x4.size()[2] - x2.size()[2]
        diffX = x4.size()[3] - x2.size()[3]

        x2 = F.pad(x2, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # input is CHW
        diffY = x4.size()[2] - x3.size()[2]
        diffX = x4.size()[3] - x3.size()[3]

        x3 = F.pad(x3, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x4, x3, x2, x1], dim=1)
        x = self.conv(x)
        return x



class up_final(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_final, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        # self.conv = double_conv_up(in_ch, out_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x1, x2):
        x = x1 + x2
        x = self.conv(x)
        x = self.up(x)
        return x


class up_final_no(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_final_no, self).__init__()

         # would be a nice idea if the upsampling could be learned too,
         # but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_up(in_ch, out_ch)
        # self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = x1 + x2
        x = self.conv(x)
        # x = x + x3
        # x = self.up(x)
        return x


class double_conv_up(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv_up, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 5, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_up_3d(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv_up_3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x



class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_up(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class up_3d(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True):
        super(up_3d, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv_up_3d(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
        #                 diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1[:,:,:1,:,:]], dim=1)
        x = self.conv(x)
        return x



class final(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(final, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class final_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(final_up, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x



class final_up_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(final_up_3d, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        # self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    def forward(self, x1, x2):
        # print(x1.shape, x2.shape)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x



class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
