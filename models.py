import torch.nn as nn
import torch.nn.functional as F  
from utils import *
from modules import *

from stb import STB
from threepositionembedding import *

class CDECT(nn.Module):
     
    def __init__(self, args):
        super(CDECT, self).__init__()
        
        self.scale = args.scale
        
        self.conv1=conv_layer(3, 48, 1)
        self.conv1_1=conv_layer(50, 48, 1)
        self.conv1_2=conv_layer(50, 48, 1)
        self.conv1_3=conv_layer(50, 48, 1)
        
        
        self.threepositionalencoding = ThreePositionalEncoding(
                                    patch_size=2,
                                    in_chans=3,
                                    embed_dim=1,                                                                             
                                    drop_rate=0.,            
                                    patch_norm=True,
                                    )
        
        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        upsample_block = pixelshuffle_block
        if self.scale == 2:  
            self.upsampler = upsample_block(48, 3, 2)
            self.upsampler_res = upsample_block(3, 3, 2)
        elif self.scale == 3:
            self.upsampler = upsample_block(48, 3, 3)
            self.upsampler_res = upsample_block(3, 3, 3)
        elif self.scale == 4:
            self.upsampler = upsample_block(48, 3, 4)
            self.upsampler_res = upsample_block(3, 3, 4)
        
        self.pfeg=PFEG(scale=self.scale)
        self.pfeg_1=PFEG(scale=self.scale)
        self.pfeg_2=PFEG(scale=self.scale)
        self.pfeg_3=PFEG(scale=self.scale)

        self.caem=CAEM()
        
        self.stbr = STB()
        self.stbg = STB()
        self.stbb = STB()

        self.act = activation('relu', neg_slope=0.05)
        
    def forward(self, x):
        
        x = self.sub_mean(x)
        res = x
        
        #CDEM
        xtr,xtg,xtb =  self.threepositionalencoding(x)
        xtr1 =  self.conv1_1(self.stbr(xtr))
        xtg1 =  self.conv1_2(self.stbg(xtg))
        xtb1 =  self.conv1_3(self.stbb(xtb))
       
             
        xp1 =  self.act(self.conv1(x))
        xp2 = self.pfeg(xp1)
        xp3 = self.pfeg_1(xp2)
        xp4 = self.pfeg_2(xp3)
        xp5 = self.pfeg_3(xp4)
        xp6 = xp1+xp5
        
        xc = self.caem(xp6,xtr1,xtg1,xtb1)
        xc1 = xc+xp6
        
 
        x1 = self.upsampler(xc1)
        out = x1+self.upsampler_res(res)

        out = self.add_mean(out)
        
        return out
        