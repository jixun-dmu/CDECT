import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import  to_2tuple
import math
from utils import *

class ThreePositionalEncoding(nn.Module):
    

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=1,
                 norm_layer=nn.LayerNorm,
                 drop_rate=0.,
                 patch_norm=True,
                ):
        super().__init__()
       
        self.conv1=conv_layer(96, 50, 1)
        self.conv1_1=conv_layer(96, 50, 1)
        self.conv1_2=conv_layer(96, 50, 1)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
    
        self.PositionalEncodingr = PositionalEncoding(num_pos_feats_x=32, num_pos_feats_y=32, num_pos_feats_z=32)
        self.PositionalEncodingg = PositionalEncoding(num_pos_feats_x=32, num_pos_feats_y=32, num_pos_feats_z=32)
        self.PositionalEncodingb = PositionalEncoding(num_pos_feats_x=32, num_pos_feats_y=32, num_pos_feats_z=32)
        self.patch_embedr = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embedg = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.patch_embedb = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
    def forward(self, x):
        
        h, w = x.size(2), x.size(3)
        xr,xg,xb = split_channels_by_third(x)
        
        
        x1 = self.patch_embedr(x)
        Wh, Ww = x1.size(2), x1.size(3)
        depth_pool =  F.interpolate(xr, size=(Wh, Ww), mode='bicubic')    
        absolute_pos_embed = self.PositionalEncodingr(x1 , depth_pool)
        x1 = (x1 + absolute_pos_embed)   
        x1 = self.conv1(self.pos_drop(x1))
        outr= F.interpolate(x1, size=(h, w), mode='bicubic')
        
        
        x2 = self.patch_embedg(x)
        Wh, Ww = x2.size(2), x2.size(3)
        depth_pool1 =  F.interpolate(xg, size=(Wh, Ww), mode='bicubic')   
        absolute_pos_embed1 = self.PositionalEncodingg(x2 , depth_pool1)
        x2 = (x2 + absolute_pos_embed1)      
        x2 = self.conv1_1(self.pos_drop(x2))
        outg= F.interpolate(x2, size=(h, w), mode='bicubic')
        
        
        x3 = self.patch_embedb(x)
        Wh, Ww = x3.size(2), x3.size(3)
        depth_pool2 =  F.interpolate(xb, size=(Wh, Ww), mode='bicubic')  
        absolute_pos_embed2 = self.PositionalEncodingb(x3 , depth_pool2)
        x3 = (x3 + absolute_pos_embed2)      
        x3 = self.conv1_2(self.pos_drop(x3))
        outb= F.interpolate(x3, size=(h, w), mode='bicubic')
        
        
        return outr,outg,outb



class PositionalEncoding(nn.Module):
    def __init__(self, num_pos_feats_x=64, num_pos_feats_y=64, num_pos_feats_z=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats_x = num_pos_feats_x
        self.num_pos_feats_y = num_pos_feats_y
        self.num_pos_feats_z = num_pos_feats_z
        self.num_pos_feats = max(num_pos_feats_x, num_pos_feats_y, num_pos_feats_z)
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, depth):
        b, c, h, w = x.size()
        b_d, c_d, h_d, w_d = depth.size()
        assert b == b_d and c_d == 1 and h == h_d and w == w_d
        
        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            y_embed = torch.arange(h, dtype=torch.float32, device=x.device).unsqueeze(1).repeat(b, 1, w)
            x_embed = torch.arange(w, dtype=torch.float32, device=x.device).repeat(b, h, 1)
        z_embed = depth.squeeze().to(dtype=torch.float32, device=x.device)

        if self.normalize:
            eps = 1e-6
            if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
                y_embed = y_embed / (y_embed.max() + eps) * self.scale
                x_embed = x_embed / (x_embed.max() + eps) * self.scale
            z_embed_max, _ = z_embed.reshape(b, -1).max(1)
            z_embed = z_embed / (z_embed_max[:, None, None] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            pos_x = x_embed[:, :, :, None] / dim_t[:self.num_pos_feats_x]
            pos_y = y_embed[:, :, :, None] / dim_t[:self.num_pos_feats_y]
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos_z = z_embed[:, :, :, None] / dim_t[:self.num_pos_feats_z]
        pos_z = torch.stack((pos_z[:, :, :, 0::2].sin(), pos_z[:, :, :, 1::2].cos()), dim=4).flatten(3)

        if self.num_pos_feats_x != 0 and self.num_pos_feats_y != 0:
            pos = torch.cat((pos_x, pos_y, pos_z), dim=3).permute(0, 3, 1, 2)
        else:
            pos = pos_z.permute(0, 3, 1, 2)
        return pos
class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
       
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x
