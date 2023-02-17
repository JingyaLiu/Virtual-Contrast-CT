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
        x = torch.cat([x2, x1 ], 1)
        # print(x.shape)

        x = self.conv(x)
        print(x.shape)

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
    def forward(self, x1, x2, x4):
        x1 = self.up4(x1)
        x2 = self.up2(x2)

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

        x = torch.cat([x4, x2, x1], dim=1)
        x = self.conv(x)
        return x

class top_1_4_3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(top_1_4_3d, self).__init__()

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.conv = double_conv_up_3d(in_ch,out_ch)
    def forward(self, x1, x2, x3):
        print(x1.shape, x2.shape, x3.shape)
        x1 = self.up4(x1)
        x2 = self.up2(x2)
        print(x1.shape, x2.shape, x3.shape)

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

        x = torch.cat([x3, x2 , x1 ], dim=1)
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



        x = torch.cat([x3, x2 , x1 ], dim=1)
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

        x = torch.cat([x3, x2, x1], dim=1)
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

        x = torch.cat([x3, x2, x1], dim=1)
        x = self.conv(x)
        return x

class top_1_5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(top_1_5, self).__init__()

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

class down_2_5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_2_5, self).__init__()
        self.mp2 = nn.MaxPool2d(2)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = double_conv_up(in_ch,out_ch)
    def forward(self, x1, x2, x3, x4):
        x1 = self.mp2(x1)
        x2 = self.up2(x2)
        x3 = self.up4(x3)


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

class down_3_5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_3_5, self).__init__()
        self.mp2 = nn.MaxPool2d(2)
        self.mp4 = nn.MaxPool2d(4)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = double_conv_up(in_ch,out_ch)
    def forward(self, x1, x2, x3, x4):
        x1 = self.mp4(x1)
        x2 = self.mp2(x2)
        x3 = self.up4(x3)

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

        diffY = x4.size()[2] - x3.size()[2]
        diffX = x4.size()[3] - x3.size()[3]
        # input is CHW
        x3 = F.pad(x3, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x4, x3, x2, x1], dim=1)
        x = self.conv(x)
        return x

class down_4_5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_4_5, self).__init__()
        self.mp2 = nn.MaxPool2d(2)
        self.mp4 = nn.MaxPool2d(4)
        self.mp8 = nn.MaxPool2d(8)
        # self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = double_conv_up(in_ch,out_ch)
    def forward(self, x1, x2, x3, x4):
        x1 = self.mp8(x1)
        x2 = self.mp4(x2)
        x3 = self.mp2(x3)

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

class down_5(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_5, self).__init__()
        self.mp2 = nn.MaxPool2d(2)
        self.mp4 = nn.MaxPool2d(4)
        self.mp8 = nn.MaxPool2d(8)
        self.mp16 = nn.MaxPool2d(16)
        # self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv = double_conv_up(in_ch,out_ch)
    def forward(self, x1, x2, x3, x4):
        x1 = self.mp16(x1)
        x2 = self.mp8(x2)
        x3 = self.mp4(x3)
        x4 = self.mp2(x4)

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

# Use residual convolution layers
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(ResidualBlock,self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels,out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

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

        x = torch.cat([x2, x1 ], dim=1)
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
        # self.conv1 = nn.Conv2d(in_ch, 10, 3)
        # self.conv2 = nn.Conv2d(20, 10, 1)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 1)
        # self.conv = nn.Conv2d(in_ch, out_ch, 1)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        # x = self.conv1(x)
        # x = self.conv2(x)
        x = self.conv3(x)
        return x

class final_up_SR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(final_up_SR, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 10, 1)
        # self.conv2 = nn.Conv2d(20, 10, 1)
        self.conv3 = nn.Conv2d(10, out_ch, 1)
        # self.up = nn.PixelShuffle(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        # x = self.conv(x)
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.conv3(x)
        x = self.up(x)

        # print(x.size())

        # print(x.size())
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
