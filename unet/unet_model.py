# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F


from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels,n_init):
        super(UNet, self).__init__()
        # self.inc = inconv(n_channels, 64)
        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)
        # self.up2 = up(512, 128)
        # self.up3 = up(256, 64)
        # self.up4 = up(128, 64)
        # self.outc = outconv(64, 1)
        self.inc = inconv(n_channels, n_init)
        self.down1 = down(n_init, n_init*2)
        self.down2 = down(n_init*2, n_init*4)
        self.down3 = down(n_init*4, n_init*8)
        # self.down4 = down(n_init*8, n_init*16)
        # self.up1 = up(n_init*16+n_init*8, n_init*8)
        self.up2 = up(n_init*8+n_init*4, n_init*4)
        self.up3 = up(n_init*4+n_init*2, n_init*2)
        self.up4 = up(n_init*2+n_init, n_init)
        self.outc = final(n_init+1, 1)
        # self.out = outconv(n_init, 1)

    def forward(self, x_init):
        x1 = self.inc(x_init)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        x_up2 = self.up2(x4, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)
        # print('out', x.size())
        # x = F.relu(self.out(x_up4))
        xx = F.relu(self.outc(x_up4,x_init))

        # print('out', x.size())
        return xx,x,x_up2,x_up3,x_up4


class UNet_4(nn.Module):
    def __init__(self, n_channels,n_init):
        super(UNet_4, self).__init__()
        self.inc = inconv(n_channels, n_init)
        self.down1 = down(n_init, n_init*2)
        self.down2 = down(n_init*2, n_init*4)
        self.down3 = down(n_init*4, n_init*8)
        self.down4 = down(n_init*8, n_init*16)
        self.up1 = up(n_init*16+n_init*8, n_init*8)
        self.up2 = up(n_init*8+n_init*4, n_init*4)
        self.up3 = up(n_init*4+n_init*2, n_init*2)
        self.up4 = up(n_init*2+n_init, n_init)
        self.outc = final(n_init+1, 1)
        self.out = outconv(n_init, 1)

    def forward(self, x_init):
        x1 = self.inc(x_init)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)
        # print('out', x.size())
        xx = F.relu(self.outc(x_up4,x_init))
        x = F.relu(self.out(x_up4))
        # x = x + x_init
        # print('out', x.size())
        return xx,x,x_up1,x_up2,x_up3,x_up4

class UNet_4_pretrain(nn.Module):
    def __init__(self, n_channels,n_init):
        super(UNet_4_pretrain, self).__init__()
        self.inc = inconv(n_channels, n_init)
        self.down1 = down(n_init, n_init*2)
        self.down2 = down(n_init*2, n_init*4)
        self.down3 = down(n_init*4, n_init*8)
        self.down4 = down(n_init*8, n_init*16)
        self.up1 = up(n_init*16+n_init*8, n_init*8)
        self.up2 = up(n_init*8+n_init*4, n_init*4)
        self.up3 = up(n_init*4+n_init*2, n_init*2)
        self.up4 = up(n_init*2+n_init, n_init)
        self.outc = final(n_init+1, 1)
        self.out = outconv(n_init, 1)
        self.fc1 = nn.Linear(512*512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x_init):
        x1 = self.inc(x_init)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)
        # print('out', x.size())
        xx = F.relu(self.outc(x_up4,x_init))
        x = F.relu(self.out(x_up4))
        x = xx.view(-1,512*512)
        x = F.relu(self.fc1(x))#.view(-1,1).squeeze()))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = x + x_init
        # print('out', x.size())
        return x#.squeeze()


class UNet_HR(nn.Module):
    def __init__(self, n_channels,n_init):
        super(UNet_HR, self).__init__()
        self.inc = inconv(n_channels, n_init)
        self.top1_2 = inconv(n_init, n_init)
        self.down2 = down_2(n_init, n_init*2)
        self.top1_3 = top_1_3(n_init*2+n_init, n_init)
        self.down2_3 = down_2_3(n_init*2+n_init, n_init*2)
        self.down3 = down_3(n_init*2+n_init, n_init*4)
        #layer 3
        self.top1_4 = top_1_4(n_init*4+n_init*2+n_init,n_init)
        # self.outc = top_1_4(n_init*4+n_init*2+n_init, n_init)
        # layer 4
        self.down2_4 = down_2_4(n_init*4+n_init*2+n_init,n_init*2)
        self.down3_4 = down_3_4(n_init*4+n_init*2+n_init,n_init*4)
        self.down4 = down_4(n_init*4+n_init*2+n_init,n_init*8)
        self.outc = top1_5(n_init*8+n_init*4+n_init*2+n_init, n_init)

        self.out = outconv(n_init, 1)

    def forward(self, x_init):
        # layer top 16
        x1 = self.inc(x_init)
        # layer 2
        x1_2 = self.top1_2(x1) #16
        x2 = self.down2(x1) #32
        # layer 3
        x1_3 = self.top1_3(x2, x1_2) #16
        x2_3 = self.down2_3(x1_2,x2) #32
        x3 = self.down3(x2,x1_2) #64
        # layer 4
        x1_4 = self.top1_4(x3,x2_3,x1_3)
        x2_4 = self.down2_4(x1_3,x3,x2_3)
        x3_4 = self.down3_4(x1_3,x2_3,x3)
        x4 = self.down4(x1_3,x2_3,x3)

        # x = F.relu(self.outc(x3,x2_3,x1_3))
        x_g2 = F.relu(self.outc(x4,x3_4,x2_4,x1_4))

        x = F.relu(self.out(x_g2))

        # print('out', x.size())
        return x,x_g2,x4,x3_4,x2_4,x1_4

class UNet_HRPXP(nn.Module):
    def __init__(self, n_channels,n_init):
        super(UNet_HRPXP, self).__init__()
        #generator 1
        # self.inc_g1 = inconv(n_channels, n_init/2)
        # self.down_g1 = down(n_init/2, n_init)
        # self.inc_g1 = inconv(n_channels, n_init)
        self.inc_g1 = inconv_k1(n_channels, n_init)

        #generator 2
        self.inc = inconv(n_channels, n_init)
        self.top1_2 = inconv(n_init, n_init)
        self.down2 = down_2(n_init, n_init*2)
        self.top1_3 = top_1_3(n_init*2+n_init, n_init)
        self.down2_3 = down_2_3(n_init*2+n_init, n_init*2)
        self.down3 = down_3(n_init*2+n_init, n_init*4)
        #layer 3
        self.top1_4 = top_1_4(n_init*4+n_init*2+n_init,n_init)
        # self.outc = top_1_4(n_init*4+n_init*2+n_init, n_init)
        # layer 4
        self.down2_4 = down_2_4(n_init*4+n_init*2+n_init,n_init*2)
        self.down3_4 = down_3_4(n_init*4+n_init*2+n_init,n_init*4)
        self.down4 = down_4(n_init*4+n_init*2+n_init,n_init*8)
        self.outc = top1_5(n_init*8+n_init*4+n_init*2+n_init, n_init)

        # self.upfinal = up_final(n_init,1)
        self.upfinal = up_final_no(n_init,1)
        # self.out = outconv(n_init, 1)

    def forward(self, x_init):
        # layer top 16
        # generator 1
        x1_g1 = self.inc_g1(x_init)
        # x2_g1 = self.down_g1(x1_g1)

        # generator 2
        x_init_d2 = F.interpolate(x_init,scale_factor = 0.5, mode='bilinear')
        x1 = self.inc(x_init_d2)
        # layer 2
        x1_2 = self.top1_2(x1) #16
        x2 = self.down2(x1) #32
        # layer 3
        x1_3 = self.top1_3(x2, x1_2) #16
        x2_3 = self.down2_3(x1_2,x2) #32
        x3 = self.down3(x2,x1_2) #64
        # layer 4
        x1_4 = self.top1_4(x3,x2_3,x1_3)
        x2_4 = self.down2_4(x1_3,x3,x2_3)
        x3_4 = self.down3_4(x1_3,x2_3,x3)
        x4 = self.down4(x1_3,x2_3,x3)
        x_g2 = F.relu(self.outc(x4,x3_4,x2_4,x1_4))
        # final
        # x = self.upfinal(x_g2, x2_g1)
        # x = self.upfinal(x_g2, x1_g1)
        x = self.upfinal(x_g2, x1_g1)

        # x = (x+x_init)

        # print('out', x.size())
        return x,x_g2,x4,x3_4,x2_4,x1_4


class UNet_HRPXP_up(nn.Module):
    def __init__(self, n_channels, n_init):
        super(UNet_HRPXP_up, self).__init__()
        #layer 1
        self.inc_g1 = inconv_k1(n_channels, n_init)
        #layer 2
        self.top1_2 = inconv(n_init, n_init)
        self.down2 = down_2(n_init, n_init*2)
        self.top1_3 = top_1_3(n_init*2+n_init, n_init)
        self.down2_3 = down_2_3(n_init*2+n_init, n_init*2)
        self.down3 = down_3(n_init*2+n_init, n_init*4)
        #layer 3
        self.top1_4 = top_1_4(n_init*4+n_init*2+n_init,n_init)
        # layer 4
        self.down2_4 = down_2_4(n_init*4+n_init*2+n_init,n_init*2)
        self.down3_4 = down_3_4(n_init*4+n_init*2+n_init,n_init*4)
        self.down4 = down_4(n_init*4+n_init*2+n_init,n_init*8)
        self.up2 = up(n_init*8+n_init*4, n_init*4)
        self.up3 = up(n_init*4+n_init*2, n_init*2)
        self.up4 = up(n_init*2+n_init, n_init)
        self.upfinal = up_final(n_init,1)
        self.outc = final_up(n_init+n_channels, n_channels)# n_channels)

    def forward(self, x_init):
        # layer top 16
        # generator 1
        x1_g1 = self.inc_g1(x_init)
        # x2_g1 = self.down_g1(x1_g1)
        # generator 2
        # x_init_d2 = F.interpolate(x_init,scale_factor = 0.5, mode='bilinear')
        # x1 = self.inc(x1_g1)
        # layer 2
        x1_2 = self.top1_2(x1_g1) #16
        x2 = self.down2(x1_g1) #32
        # layer 3
        x1_3 = self.top1_3(x2, x1_2) #16
        x2_3 = self.down2_3(x1_2,x2) #32
        x3 = self.down3(x2,x1_2) #64
        # layer 4
        x1_4 = self.top1_4(x3,x2_3,x1_3)
        x2_4 = self.down2_4(x1_3,x3,x2_3)
        x3_4 = self.down3_4(x1_3,x2_3,x3)
        x4 = self.down4(x1_3,x2_3,x3)
        # x_g2 = F.relu(self.outc(x4,x3_4,x2_4,x1_4))
        x_up2 = self.up2(x4, x3_4)
        x_up3 = self.up3(x_up2, x2_4)
        x_up4 = self.up4(x_up3, x1_4)
        # print('out', x.size())
        xx = self.outc(x_up4,x_init)

        return xx,x_up4,x_up3,x_up2,x4

class UNet_HRPXP_up_5l(nn.Module):
    def __init__(self, n_channels, n_init):
        super(UNet_HRPXP_up_5l, self).__init__()
        #layer 1
        self.inc_g1 = inconv_k1(n_channels, n_init)
        #layer 2
        self.top1_2 = inconv(n_init, n_init)
        self.down2 = down_2(n_init, n_init*2)
        #layer 3
        self.top1_3 = top_1_3(n_init*2+n_init, n_init)
        self.down2_3 = down_2_3(n_init*2+n_init, n_init*2)
        self.down3 = down_3(n_init*2+n_init, n_init*4)
        # layer 4
        self.top1_4 = top_1_4(n_init*4+n_init*2+n_init,n_init)
        self.down2_4 = down_2_4(n_init*4+n_init*2+n_init,n_init*2)
        self.down3_4 = down_3_4(n_init*4+n_init*2+n_init,n_init*4)
        self.down4 = down_4(n_init*4+n_init*2+n_init,n_init*8)
        # layer 5
        self.top1_5 = top_1_5(n_init*8+n_init*4+n_init*2+n_init,n_init)
        self.down2_5 = down_2_5(n_init*8+n_init*4+n_init*2+n_init,n_init*2)
        self.down3_5 = down_3_5(n_init*8+n_init*4+n_init*2+n_init,n_init*4)
        self.down4_5 = down_4_5(n_init*8+n_init*4+n_init*2+n_init,n_init*8)
        self.down5 = down_5(n_init*8+n_init*4+n_init*2+n_init,n_init*16)

        self.up1 = up(n_init*16+n_init*8, n_init*8)
        self.up2 = up(n_init*8+n_init*4, n_init*4)
        self.up3 = up(n_init*4+n_init*2, n_init*2)
        self.up4 = up(n_init*2+n_init, n_init)
        self.upfinal = up_final(n_init,1)
        self.outc = final_up(n_init+n_channels, n_channels)# n_channels)

    def forward(self, x_init):
        # layer top 16
        # generator 1
        x1_g1 = self.inc_g1(x_init)
        # layer 2
        x1_2 = self.top1_2(x1_g1) #16
        x2 = self.down2(x1_g1) #32
        # layer 3
        x1_3 = self.top1_3(x2, x1_2) #16
        x2_3 = self.down2_3(x1_2,x2) #32
        x3 = self.down3(x2,x1_2) #64
        # layer 4
        x1_4 = self.top1_4(x3,x2_3,x1_3)
        x2_4 = self.down2_4(x1_3,x3,x2_3)
        x3_4 = self.down3_4(x1_3,x2_3,x3)
        x4 = self.down4(x1_3,x2_3,x3)
        # layer 5
        x1_5 = self.top1_5(x2_4,x3_4,x4,x1_4)
        x2_5 = self.down2_5(x1_4,x3_4,x4,x2_4)
        x3_5 = self.down3_5(x1_4,x2_4,x4,x3_4)
        x4_5 = self.down4_5(x1_4,x2_4,x3_4,x4)
        x5 = self.down5(x1_5,x2_5,x3_5,x4_5)

        x_up1 = self.up1(x5, x4_5)
        x_up2 = self.up2(x_up1, x3_5)
        x_up3 = self.up3(x_up2, x2_5)
        x_up4 = self.up4(x_up3, x1_5)
        xx = self.outc(x_up4,x_init)

        return xx,x_up4,x_up3,x_up2,x_up1




class UNet_HRPXP_up_SR(nn.Module):
    def __init__(self, n_channels, n_init):
        super(UNet_HRPXP_up_SR, self).__init__()
        #generator 1
        # self.inc_g1 = inconv(n_channels, n_init/2)
        # self.down_g1 = down(n_init/2, n_init)
        # self.inc_g1 = inconv(n_channels, n_init)
        self.inc_g1 = inconv_k1(n_channels, n_init)

        #generator 2
        # self.inc = inconv(n_channels, n_init)
        self.top1_2 = inconv(n_init, n_init)
        self.down2 = down_2(n_init, n_init*2)
        self.top1_3 = top_1_3(n_init*2+n_init, n_init)
        self.down2_3 = down_2_3(n_init*2+n_init, n_init*2)
        self.down3 = down_3(n_init*2+n_init, n_init*4)
        #layer 3
        self.top1_4 = top_1_4(n_init*4+n_init*2+n_init,n_init)
        # self.outc = top_1_4(n_init*4+n_init*2+n_init, n_init)
        # layer 4
        self.down2_4 = down_2_4(n_init*4+n_init*2+n_init,n_init*2)
        self.down3_4 = down_3_4(n_init*4+n_init*2+n_init,n_init*4)
        self.down4 = down_4(n_init*4+n_init*2+n_init,n_init*8)
        # self.outc = top1_5(n_init*8+n_init*4+n_init*2+n_init, n_init)
        self.up2 = up(n_init*8+n_init*4, n_init*4)
        self.up3 = up(n_init*4+n_init*2, n_init*2)
        self.up4 = up(n_init*2+n_init, n_init)
        # self.upfinal = up_final(n_init,1)
        self.upfinal = up_final(n_init,1)
        self.outc = final_up_SR(n_init+n_channels, n_channels)# n_channels)
        # self.out = outconv(n_init, 1)
        # self.out = outconv(n_init, 1)

    def forward(self, x_init):
        # layer top 16
        # generator 1
        x1_g1 = self.inc_g1(x_init)
        # x2_g1 = self.down_g1(x1_g1)
        # generator 2
        # x_init_d2 = F.interpolate(x_init,scale_factor = 0.5, mode='bilinear')
        # x1 = self.inc(x1_g1)
        # layer 2
        x1_2 = self.top1_2(x1_g1) #16
        x2 = self.down2(x1_g1) #32
        # layer 3
        x1_3 = self.top1_3(x2, x1_2) #16
        x2_3 = self.down2_3(x1_2,x2) #32
        x3 = self.down3(x2,x1_2) #64
        # layer 4
        x1_4 = self.top1_4(x3,x2_3,x1_3)
        x2_4 = self.down2_4(x1_3,x3,x2_3)
        x3_4 = self.down3_4(x1_3,x2_3,x3)
        x4 = self.down4(x1_3,x2_3,x3)
        # x_g2 = F.relu(self.outc(x4,x3_4,x2_4,x1_4))
        x_up2 = self.up2(x4, x3_4)
        x_up3 = self.up3(x_up2, x2_4)
        x_up4 = self.up4(x_up3, x1_4)
        # print('out', x.size())
        xx = self.outc(x_up4,x_init)

        # out = x_init + x_up4
        # out = torch.clamp(out, min=-1, max=1)
        # final
        # x = self.upfinal(x_g2, x2_g1)
        # x = self.upfinal(x_g2, x1_g1)
        # x = F.relu(self.upfinal(x_g2, x1_g1))

        # x = (x+x_init)

        # print('out', x.size())
        return xx,x_up4,x_up3,x_up2,x4



class UNet_HRPXP_up_3d(nn.Module):
    def __init__(self, n_channels, n_init):
        super(UNet_HRPXP_up_3d, self).__init__()
        #generator 1
        self.inc_g1 = inconv_k1_3d(n_channels, n_init)

        #generator 2
        # self.inc = inconv(n_channels, n_init)
        self.top1_2 = inconv_3d(n_init, n_init)
        self.down2 = down_2_3d(n_init, n_init*2)
        self.top1_3 = top_1_3_3d(n_init*2+n_init, n_init)
        self.down2_3 = down_2_3_3d(n_init*2+n_init, n_init*2)
        self.down3 = down_3_3d(n_init*2+n_init, n_init*4)
        #layer 3
        self.top1_4 = top_1_4_3d(n_init*4+n_init*2+n_init,n_init)
        # self.outc = top_1_4(n_init*4+n_init*2+n_init, n_init)
        # layer 4
        self.down2_4 = down_2_4_3d(n_init*4+n_init*2+n_init,n_init*2)
        self.down3_4 = down_3_4_3d(n_init*4+n_init*2+n_init,n_init*4)
        self.down4 = down_4_3d(n_init*4+n_init*2+n_init,n_init*8)
        # self.outc = top1_5(n_init*8+n_init*4+n_init*2+n_init, n_init)
        self.up2 = up_3d(n_init*8+n_init*4, n_init*4)
        self.up3 = up_3d(n_init*4+n_init*2, n_init*2)
        self.up4 = up_3d(n_init*2+n_init, n_init)

        self.outc = final_up_3d(n_init+n_channels, n_channels)

    def forward(self, x_init):
        # layer top 16
        # generator 1
        # print(x_init.shape)

        # x_init = x_init.view(-1, 1, n_channels,input.shape[3],input.shape[4])

        x1_g1 = self.inc_g1(x_init)
        # layer 2
        x1_2 = self.top1_2(x1_g1) #16
        x2 = self.down2(x1_g1) #32
        # layer 3
        x1_3 = self.top1_3(x2, x1_2) #16
        x2_3 = self.down2_3(x1_2,x2) #32
        x3 = self.down3(x2,x1_2) #64
        # layer 4
        x1_4 = self.top1_4(x3,x2_3,x1_3)
        x2_4 = self.down2_4(x1_3,x3,x2_3)
        x3_4 = self.down3_4(x1_3,x2_3,x3)
        x4 = self.down4(x1_3,x2_3,x3)
        # x_g2 = F.relu(self.outc(x4,x3_4,x2_4,x1_4))
        x_up2 = self.up2(x4, x3_4)
        x_up3 = self.up3(x_up2, x2_4)
        x_up4 = self.up4(x_up3, x1_4)
        #  add input
        # xx = F.relu(self.outc(x_up4,x_init))
        xx = (self.outc(x_up4,x_init))

        # print('out', x.size())
        return xx,x_up4,x_up3,x_up2,x4

class UNet_4_intial(nn.Module):
    def __init__(self, n_channels,n_init):
        super(UNet_4_intial, self).__init__()
        self.inc = inconv(n_channels, n_init)
        self.down1 = down(n_init, n_init*2)
        self.down2 = down(n_init*2, n_init*4)
        self.down3 = down(n_init*4, n_init*8)
        self.down4 = down(n_init*8, n_init*16)
        self.up1 = up(n_init*16+n_init*8, n_init*8)
        self.up2 = up(n_init*8+n_init*4, n_init*4)
        self.up3 = up(n_init*4+n_init*2, n_init*2)
        self.up4 = up(n_init*2+n_init, n_init)
        # self.outc = final(n_init+1, 1)
        self.outc = outconv(n_init, 1)

    def forward(self, x_init):
        x1 = self.inc(x_init)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)
        # print('out', x.size())
        # xx = F.relu(self.outc(x_up4,x_init))
        x = F.relu(self.outc(x_up4))
        # x = x + x_init
        # print('out', x.size())
        return x#,x_up1,x_up2,x_up3,x_up4

class UNet_HRPXP_up_pretrain(nn.Module):
    def __init__(self, n_channels,n_init):
        super(UNet_HRPXP_up_pretrain, self).__init__()
        #generator 1
        # self.inc_g1 = inconv(n_channels, n_init/2)
        # self.down_g1 = down(n_init/2, n_init)
        # self.inc_g1 = inconv(n_channels, n_init)
        self.inc_g1 = inconv_k1(n_channels, n_init)

        #generator 2
        # self.inc = inconv(n_channels, n_init)
        self.top1_2 = inconv(n_init, n_init)
        self.down2 = down_2(n_init, n_init*2)
        self.top1_3 = top_1_3(n_init*2+n_init, n_init)
        self.down2_3 = down_2_3(n_init*2+n_init, n_init*2)
        self.down3 = down_3(n_init*2+n_init, n_init*4)
        #layer 3
        self.top1_4 = top_1_4(n_init*4+n_init*2+n_init,n_init)
        # self.outc = top_1_4(n_init*4+n_init*2+n_init, n_init)
        # layer 4
        self.down2_4 = down_2_4(n_init*4+n_init*2+n_init,n_init*2)
        self.down3_4 = down_3_4(n_init*4+n_init*2+n_init,n_init*4)
        self.down4 = down_4(n_init*4+n_init*2+n_init,n_init*8)
        # self.outc = top1_5(n_init*8+n_init*4+n_init*2+n_init, n_init)
        self.up2 = up(n_init*8+n_init*4, n_init*4)
        self.up3 = up(n_init*4+n_init*2, n_init*2)
        self.up4 = up(n_init*2+n_init, n_init)
        # self.upfinal = up_final(n_init,1)
        self.upfinal = up_final(n_init,1)
        self.outc = final_up(n_init+n_channels, n_channels)# n_channels)
        # self.out = outconv(n_init, 1)


        # self.out = outconv(n_init, 1)
        # self.out = outconv(n_init, 1)
        self.fc1 = nn.Linear(3*512*512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4) # 4 classes

    def forward(self, x_init):
        # layer top 16
        # generator 1
        x1_g1 = self.inc_g1(x_init)
        # x2_g1 = self.down_g1(x1_g1)
        # generator 2
        # x_init_d2 = F.interpolate(x_init,scale_factor = 0.5, mode='bilinear')
        # x1 = self.inc(x1_g1)
        # layer 2
        x1_2 = self.top1_2(x1_g1) #16
        x2 = self.down2(x1_g1) #32
        # layer 3
        x1_3 = self.top1_3(x2, x1_2) #16
        x2_3 = self.down2_3(x1_2,x2) #32
        x3 = self.down3(x2,x1_2) #64
        # layer 4
        x1_4 = self.top1_4(x3,x2_3,x1_3)
        x2_4 = self.down2_4(x1_3,x3,x2_3)
        x3_4 = self.down3_4(x1_3,x2_3,x3)
        x4 = self.down4(x1_3,x2_3,x3)
        # x_g2 = F.relu(self.outc(x4,x3_4,x2_4,x1_4))
        x_up2 = self.up2(x4, x3_4)
        x_up3 = self.up3(x_up2, x2_4)
        x_up4 = self.up4(x_up3, x1_4)
        # print('out', x.size())
        xx = self.outc(x_up4,x_init)


        # x = xx.view(-1,3*512*512)
        # x = F.relu(self.fc1(x))#.view(-1,1).squeeze()))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)

        return xx,x_up4,x_up3,x_up2,x4#x,xx,x_up4,x_up3,x_up2,x4

class UNet_HRPXP_up_5l_pretrain(nn.Module):
    def __init__(self, n_channels,n_init):
        super(UNet_HRPXP_up_5l_pretrain, self).__init__()
        #layer 1
        self.inc_g1 = inconv_k1(n_channels, n_init)
        #layer 2
        self.top1_2 = inconv(n_init, n_init)
        self.down2 = down_2(n_init, n_init*2)
        #layer 3
        self.top1_3 = top_1_3(n_init*2+n_init, n_init)
        self.down2_3 = down_2_3(n_init*2+n_init, n_init*2)
        self.down3 = down_3(n_init*2+n_init, n_init*4)
        # layer 4
        self.top1_4 = top_1_4(n_init*4+n_init*2+n_init,n_init)
        self.down2_4 = down_2_4(n_init*4+n_init*2+n_init,n_init*2)
        self.down3_4 = down_3_4(n_init*4+n_init*2+n_init,n_init*4)
        self.down4 = down_4(n_init*4+n_init*2+n_init,n_init*8)
        # layer 5
        self.top1_5 = top_1_5(n_init*8+n_init*4+n_init*2+n_init,n_init)
        self.down2_5 = down_2_5(n_init*8+n_init*4+n_init*2+n_init,n_init*2)
        self.down3_5 = down_3_5(n_init*8+n_init*4+n_init*2+n_init,n_init*4)
        self.down4_5 = down_4_5(n_init*8+n_init*4+n_init*2+n_init,n_init*8)
        self.down5 = down_5(n_init*8+n_init*4+n_init*2+n_init,n_init*16)

        self.up1 = up(n_init*16+n_init*8, n_init*8)
        self.up2 = up(n_init*8+n_init*4, n_init*4)
        self.up3 = up(n_init*4+n_init*2, n_init*2)
        self.up4 = up(n_init*2+n_init, n_init)
        self.upfinal = up_final(n_init,1)
        self.outc = final_up(n_init+n_channels, n_channels)# n_channels)
        self.fc1 = nn.Linear(3*256*256, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4) # 4 classes

    def forward(self, x_init):
        # layer top 16
        # generator 1
        x1_g1 = self.inc_g1(x_init)
        # layer 2
        x1_2 = self.top1_2(x1_g1) #16
        x2 = self.down2(x1_g1) #32
        # layer 3
        x1_3 = self.top1_3(x2, x1_2) #16
        x2_3 = self.down2_3(x1_2,x2) #32
        x3 = self.down3(x2,x1_2) #64
        # layer 4
        x1_4 = self.top1_4(x3,x2_3,x1_3)
        x2_4 = self.down2_4(x1_3,x3,x2_3)
        x3_4 = self.down3_4(x1_3,x2_3,x3)
        x4 = self.down4(x1_3,x2_3,x3)
        # layer 5
        x1_5 = self.top1_5(x2_4,x3_4,x4,x1_4)
        x2_5 = self.down2_5(x1_4,x3_4,x4,x2_4)
        x3_5 = self.down3_5(x1_4,x2_4,x4,x3_4)
        x4_5 = self.down4_5(x1_4,x2_4,x3_4,x4)
        x5 = self.down5(x1_5,x2_5,x3_5,x4_5)

        x_up1 = self.up1(x5, x4_5)
        x_up2 = self.up2(x_up1, x3_5)
        x_up3 = self.up3(x_up2, x2_5)
        x_up4 = self.up4(x_up3, x1_5)
        xx = F.relu(self.outc(x_up4,x_init))

        size = x_init.size()[-1]
        x = xx.view(-1,3*size*size)
        x = F.relu(self.fc1(x))#.view(-1,1).squeeze()))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x #,xx,x_up4,x_up3,x_up2,x_up1

class discriminator_3c(nn.Module):
    def __init__(self, ndf = 64):
        super(discriminator_3c, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(6, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False),
            # nn.Linear()
        )

    def forward(self, input, label):
        # print(input.shape, label.shape)

        img_input = torch.cat((input, label), 1)
        output = self.main(img_input)
        # print(output.shape)
        return output

class discriminator_light(nn.Module):
    def __init__(self, ndf = 64):
        super(discriminator_light, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(6, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 1, bias=False),
            # nn.Linear()
        )

    def forward(self, input, label):
        # print(input.shape, label.shape)

        img_input = torch.cat((input, label), 1)
        output = self.main(img_input)
        # print(output.shape)
        return output

class discriminator_3c_SR(nn.Module):
    def __init__(self, ndf = 64):
        super(discriminator_3c_SR, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(6, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 1, bias=False),
            # nn.Linear()
        )
        self.input_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, input, label):
        # print(input.shape, label.shape)

        img_input = torch.cat((self.input_up(label), input), 1)
        output = self.main(img_input)
        # print(output.shape)
        return output
# class discriminator_3d(nn.Module):
#     def __init__(self, ndf = 64):
#         super(discriminator_3d, self).__init__()
#         # self.ngpu = ngpu
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv3d(2, ndf,  kernel_size=4, stride=2,  bias=False, padding=(1, 1, 1)),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2,  bias=False, padding=(1, 1, 1)),
#             nn.BatchNorm3d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2,  bias=False, padding=(1, 1, 1)),
#             nn.BatchNorm3d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2,  bias=False, padding=(1, 1, 1)),
#             nn.BatchNorm3d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv3d(ndf * 8, 1, kernel_size=4, stride=1,  bias=False, padding=(1, 1, 1)),
#
#         )
#
#     def forward(self, input, label):
#         img_input = torch.cat((input, label), 1)
#         print(img_input.size())
#         # print(out.size()) # torch.Size([100, 1, 64, 64, 64])
#         out = self.layer1(img_input)
#         print(out.size())  # torch.Size([100, 64, 32, 32, 32])
#         out = self.layer2(out)
#         print(out.size())  # torch.Size([100, 128, 16, 16, 16])
#         out = self.layer3(out)
#         print(out.size())  # torch.Size([100, 256, 8, 8, 8])
#         out = self.layer4(out)
#         print(out.size())  # torch.Size([100, 512, 4, 4, 4])
#         out = self.layer5(out)
#         print(out.size())  # torch.Size([100, 200, 1, 1, 1])
#         # out = img_input.view(-1, 1, self.args.cube_len, self.args.cube_len, self.args.cube_len)
#         output = self.main(img_input)
#         return output

class discriminator_3d(torch.nn.Module):
    def __init__(self, ndf = 64):
        super(discriminator_3d, self).__init__()

        self.layer1 = torch.nn.Sequential(
            nn.Conv3d(1, ndf,  kernel_size=4, stride=2,  bias=False, padding=(1, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = torch.nn.Sequential(
            nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2,  bias=False, padding=(1, 1, 1)),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = torch.nn.Sequential(
            nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2,  bias=False, padding=(1, 1, 1)),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer4 = torch.nn.Sequential(
            nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=2,  bias=False, padding=(1, 1, 1)),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer5 = torch.nn.Sequential(
            nn.Conv3d(ndf * 8, 1, kernel_size=4, stride=1,  bias=False, padding=(1, 1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, input, label):
        img_input = torch.cat((input, label), 1)
        # print('img_input',img_input.shape, input.shape,label.shape)
        out = img_input.view(-1, 1, 32,input.shape[3],input.shape[4])
        # print(out.shape)
        # print(img_input.size())
        out = self.layer1(out)
        # print(out.size())  #
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())  # torch.Size([100, 256, 8, 8, 8])
        out = self.layer4(out)
        # print(out.size())  # torch.Size([100, 512, 4, 4, 4])
        out = self.layer5(out)
        # print(out.size())  # torch.Size([100, 200, 1, 1, 1])
        # output = self.main(img_input)
        return out


class discriminator(nn.Module):
    def __init__(self, ndf = 64):
        super(discriminator, self).__init__()
        # self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(2, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input, label):
        img_input = torch.cat((input, label), 1)
        # print(img_input.size())
        output = self.main(img_input)
        return output



class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs


# class discriminator(nn.Module):
#     # initializers
#     def __init__(self, d=128):
#         super(discriminator, self).__init__()
#         # self.conv1_1 = nn.Conv2d(1, d/2, 4, 2, 1)
#         # self.conv1_2 = nn.Conv2d(1, d/2, 4, 2, 1)
#         self.conv1_1 = nn.Conv2d(1, d, 4, 2, 1)
#         self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
#         self.conv2_bn = nn.BatchNorm2d(d*2)
#         self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
#         self.conv3_bn = nn.BatchNorm2d(d*4)
#         # self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0)
#         self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
#         self.conv4_bn = nn.BatchNorm2d(d*8)
#         self.conv5_1 = nn.Conv2d(d*8, d*4, 8, 2, 1)
#         self.conv5_bn = nn.BatchNorm2d(d*4)
#         self.conv5 = nn.Conv2d(d*4, 1,  6, 3, 1)
#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)
#
#     # forward method
#     # def forward(self, input):
#     def forward(self, input, label):
#         x = F.leaky_relu(self.conv1_1(input), 0.2)
#         # y = F.leaky_relu(self.conv1_2(label), 0.2)
#         # x = torch.cat([x, y], 1)
#         x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
#         x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
#         # x = F.sigmoid(self.conv4(x))
#         x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
#         x = F.leaky_relu(self.conv5_bn(self.conv5_1(x)), 0.2)
#
#         x = F.sigmoid(self.conv5(x))
#
#         return x

# class discriminator(nn.Module):
#     def __init__(self, ngpu):
#         super(discriminator, self).__init__()
#         self.ngpu = ngpu
#         self.ndf = 64
#         self.main = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(1, self.ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(self.ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (self.ndf*2) x 16 x 16
#             nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(self.ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (self.ndf*4) x 8 x 8
#             nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(self.ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (self.ndf*8) x 4 x 4
#             nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         if input.is_cuda and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#         else:
#             output = self.main(input)
#
#         return output.view(-1, 1).squeeze(1)


# class discriminator(nn.Module):
#     def __init__(self, in_channels=1):
#         super(discriminator, self).__init__()
#
#         def discriminator_block(in_filters, out_filters, normalization=True):
#             """Returns downsampling layers of each discriminator block"""
#             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#             if normalization:
#                 layers.append(nn.InstanceNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             discriminator_block(in_channels * 2, 64, normalization=False),
#             discriminator_block(64, 128),
#             discriminator_block(128, 256),
#             discriminator_block(256, 512),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(512, 1, 4, padding=1, bias=False)
#         )
#
#     def forward(self, img_A, img_B):
#         # Concatenate image and condition image by channels to produce input
#         img_input = torch.cat((img_A, img_B), 1)
#         return self.model(img_input)
