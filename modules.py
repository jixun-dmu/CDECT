import torch
import torch.nn as nn
from utils import *
from nlsa import *

class PFEG(nn.Module):   
    def __init__(self,scale):
        super(PFEG, self).__init__()
        
        self.conv1=conv_layer(64, 48, 1)
        self.conv1_1=conv_layer(48, 48, 1)
        self.conv1_2=conv_layer(48, 48, 1)
        self.conv1_3=conv_layer(48, 48, 1)
        self.conv1_4=conv_layer(96, 48, 1)
        self.conv1_5=conv_layer(48, 48, 1)
        self.conv1_6=conv_layer(48, 48, 1)
        self.conv3=conv_layer(192, 48, 3)
        self.deconv = DEConv(48)
        
        scale=scale
        
        self.hreb = HREB()
        
        self.udrb=UDRB(in_channels=48, nr=48, scale=scale, up=True, bottleneck=True)
        self.nlsa= NonLocalSparseAttention(
              channels=48, chunk_size=25, n_hashes=4, reduction=4, res_scale=0.1)
        self.contrast_h = stdv_channels_h
        self.contrast_w = stdv_channels_w
        
        self.act = activation('relu', neg_slope=0.05)
        
    def forward(self, x):

        #CFEM
        x1 =  self.hreb(x)
        x2 =   self.act(self.conv1_1(x))
        
        x3 = self.hreb(x1)
        x4 =  self.act(self.conv1_2(x1))
        
        x5 = self.hreb(x3)
        x6 =  self.act(self.conv1_3(x3))
        x7 = self.deconv(x5)
        
        x8 = torch.cat([x2, x4, x6, x7], dim=1)
        x9 =  self.act(self.conv3(x8))
        
        x10 = self.udrb(x9)
        #PFEG
        x11 = x10+x

        #NLSA
        x12 = self.nlsa(x11)

        #SFCM
        xh = self.contrast_h(x)
        xw = self.contrast_w(x)

        x13 = xh*xw

        xh = torch.sigmoid(self.act(self.conv1_5(self.contrast_h(x))))
        xw = torch.sigmoid(self.act(self.conv1_6(self.contrast_w(x))))
        
        #PFEG
        x13 = self.act(self.conv1_4(torch.cat([x12, xh*xw*x11], dim=1)))

        return x13


class HREB(nn.Module):
    def __init__(self):
        super(HREB, self).__init__()
        
        self.conv1_2=conv_layer(48, 48, 1)
        self.conv1_3=conv_layer(48, 48, 1)
        self.conv3=conv_layer(48, 48, 3)

    
        self.gconv7 = nn.Conv2d(48, 48, 7, padding=3, groups=48)
        self.gdconv7 = nn.Conv2d(48, 48, 7, stride=1, padding=9, groups=48, dilation=3)
        self.gconv9 = nn.Conv2d(48, 48, 9, padding=4, groups=48)
        self.gdconv9 = nn.Conv2d(48, 48, 9, stride=1, padding=12, groups=48, dilation=3)
        
        self.act = activation('relu', neg_slope=0.05)
        
    def forward(self, x):
        
        
        
        x1 = self.conv1_2(x)
        x2 = self.gconv7(x1)
        x3 = self.act(self.gdconv7(x2))
        
        x4 = self.conv1_3(x)
        x5 = self.gconv9(x4)
        x6 = self.act(self.gdconv9(x5))
        
        x7 = torch.sigmoid(x3)
        x8 = torch.sigmoid(x6)
        
        x9 = x*x7
        x10 = x*x8
        
        x6 = self.act(self.conv3(x9+x10))+x
        
        
        return x6
       
class UDRB(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(UDRB, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])
        self.conv_2 = nn.Sequential(*[
            projection_conv(nr, inter_channels, scale, not up),
            nn.PReLU(inter_channels)
        ])
        self.conv_3 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])
        self.conv_4 = nn.Sequential(*[
            projection_conv(nr, inter_channels, scale, not up),
            nn.PReLU(inter_channels)
        ])
        
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        x1 = self.conv_1(x)
        x2 = self.conv_2(x1)
        x3 = x2.sub(x)
        x4 = self.conv_3(x3)
        x5 = self.conv_4(x4)
        x6 = x5+x2+x

        return x6   

class CAEM(nn.Module):   
    def __init__(self):
        super(CAEM, self).__init__()
        
        
        self.conv3=conv_layer(144, 48, 3)
        
        self.act = activation('relu', neg_slope=0.05)
        
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, x,xr,xg,xb):
        
        xr = self.sigmoid(xr)
        xg = self.sigmoid(xg)
        xb = self.sigmoid(xb)
        x1 = xr*x+x
        x2 = xg*x+x
        x3 = xb*x+x
        
        x4 =  self.act(self.conv3(torch.cat([x1, x2, x3], dim=1)))

        return x4     

