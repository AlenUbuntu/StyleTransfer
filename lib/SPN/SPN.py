from torchvision.models import vgg16 
import torch 
import torch.nn as nn 
import sys 
from .pytorch_spn.modules.gaterecurrent2dnoind import GateRecurrent2dnoind


class SpnBlock(nn.Module):
    def __init__(self, horizontal, reverse):
        super(SpnBlock, self).__init__()
        self.propagator = GateRecurrent2dnoind(horizontal, reverse)
    
    def forward(self, x, G1, G2, G3):
        sum_abs = G1.abs() + G2.abs() + G3.abs()
        sum_abs.data[sum_abs.data == 0] = 1e-6
        mask_need_norm = sum_abs.ge(1).float()

        G1_norm = torch.div(G1, sum_abs)
        G2_norm = torch.div(G2, sum_abs)
        G3_norm = torch.div(G3, sum_abs)

        G1 = torch.add(-mask_need_norm, 1) * G1 + mask_need_norm * G1_norm
        G2 = torch.add(-mask_need_norm, 1) * G2 + mask_need_norm * G2_norm
        G3 = torch.add(-mask_need_norm, 1) * G3 + mask_need_norm * G3_norm

        return self.propagator(x, G1, G2, G3)


class Encoder(nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        # (N, 3, 256, 256)
        self.conv1 = nn.Conv2d(3, channels, 3, 1, 1)  # (N, C, 256, 256)
        self.pool1 = nn.MaxPool2d(3, 2, 1)  # (N, C, 128, 128)

        self.conv2 = nn.Conv2d(channels, channels*2, 3, 1, 1)  # (N, 2C, 128, 128)
        self.pool2 = nn.MaxPool2d(3, 2, 1)  # (N, 2C, 64, 64)

        self.conv3 = nn.Conv2d(channels*2, channels*4, 3, 1, 1)  # (N, 4C, 64, 64)
        self.pool3 = nn.MaxPool2d(3, 2, 1)  # (N, 4C, 32, 32)

        self.conv4 = nn.Conv2d(channels*4, channels*8, 3, 1, 1)  # (N, 8C, 32, 32)
    
    def forward(self, x):
        output = {}
        output['conv1'] = self.conv1(x)
        
        out = torch.relu(output['conv1'])
        out = self.pool1(out)

        output['conv2'] = self.conv2(out)
        out = torch.relu(output['conv2'])
        out = self.pool2(out)

        output['conv3'] = self.conv3(out)
        out = torch.relu(output['conv3'])
        out = self.pool3(out)

        output['conv4'] = self.conv4(out)

        return output 

    
class Decoder(nn.Module):
    def __init__(self, channels=32, spn=1):
        super(Decoder, self).__init__()

        # 32 x 32 
        self.layer0 = nn.Conv2d(channels*8, channels*4, 1, 1, 0)  # (N, 4C, 32, 32), edge conv5
        self.layer1 = nn.Upsample(scale_factor=2, mode='bilinear')  # (N, 4C, 64, 64)

        # 64 x 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels*4, channels*4, 3, 1, 1),  # (N, 4C, 64, 64)
            nn.ELU(inplace=True)
        )
        self.layer3 = nn.Upsample(scale_factor=2, mode='bilinear')  # (N, 4C, 128, 128)

        # 128 x 128 
        self.layer4 = nn.Sequential(
            nn.Conv2d(channels*4, channels*2, 3, 1, 1),   # (N, 2C, 128, 128)
            nn.ELU(inplace=True)
        )

        self.layer5 = nn.Upsample(scale_factor=2, mode='bilinear')  # (N, 2C, 256, 256)

        # 256 x 256
        self.layer6 = nn.Sequential(
            nn.Conv2d(channels*2, channels, 3, 1, 1),  # (N, C, 256, 256)
            nn.ELU(inplace=True)
        )

        # output weights for 4 directions * 3 neighbors
        if spn==1:
            self.layer7 = nn.Conv2d(channels, channels*12, 3, 1, 1)  # (N, 12C, 256, 256)
        else:
            self.layer7 = nn.Conv2d(channels, channels*24, 3, 1, 1)  # (N, 24C, 256, 256)
        
    def forward(self, x):
        out = self.layer0(x['conv4'])
        out = self.layer1(out)

        out = self.layer2(out)
        out = out + x['conv3']

        out = self.layer3(out)
        out = self.layer4(out)
        out = out + x['conv2']

        out = self.layer5(out)
        out = self.layer6(out)
        out = out + x['conv1']

        out = self.layer7(out)

        return out

class SPN(nn.Module):
    def __init__(self, channels=32, spn=1):
        super(SPN, self).__init__()

        # conv for mask
        self.mask_conv = nn.Conv2d(3, channels, 3, 1, 1)

        # guidence network (U-NET Encoder-Decoder Structure)
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels, spn)

        # Propogation Blocks 
        self.left_to_right = SpnBlock(True, False)
        self.right_to_left = SpnBlock(True, True)
        self.top_to_bottom = SpnBlock(False, False)
        self.bottom_to_top = SpnBlock(False, True)

        # Post Processing - convert back to RGB channels
        self.post = nn.Conv2d(channels, 3, 3, 1, 1)

        self.channels = channels
    
    def forward(self, mask, rgb):
        # mask feature
        x = self.mask_conv(mask)

        # esitimate affine transformation matrix 
        guide = self.decoder(self.encoder(rgb))

        # Propogation 
        G = torch.split(guide, self.channels, dim=1)  # split at C channel 

        out1 = self.left_to_right(x, G[0], G[1], G[2])
        out2 = self.right_to_left(x, G[3], G[4], G[5])
        out3 = self.top_to_bottom(x, G[6], G[7], G[8])
        out4 = self.bottom_to_top(x, G[9], G[10], G[11])

        # combine the result of 4 directions by pooling 
        out = torch.max(out1, out2)
        out = torch.max(out, out3)
        out = torch.max(out, out4)

        # post processing
        out = self.post(out)

        return out 
