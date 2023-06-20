import torch
import torch.nn as nn
from einops import rearrange


class half_PolyPhase_resUnet_Adain(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(half_PolyPhase_resUnet_Adain, self).__init__()
        self.down1 = down_half_polyphase(in_ch=in_ch, out_ch=32)
        self.down2 = down_half_polyphase(in_ch=32, out_ch=128)
        self.down3 = down_half_polyphase(in_ch=128, out_ch=256)
        self.down4 = down_half_polyphase(in_ch=256, out_ch=512)
        self.bottle_neck = down_half_polyphase_end(in_ch=512, out_ch=512)
        self.up4 = up_half_polyphase(in_ch=512,
                                     out_ch=256)
        self.up3 = up_half_polyphase(in_ch=256,
                                     out_ch=128)
        self.up2 = up_half_polyphase(in_ch=128,
                                     out_ch=32)
        self.up1 = nn.Sequential(
            *[nn.ConvTranspose3d(in_channels=32, out_channels=4, kernel_size=4, padding=1, stride=2)])
        self.outconv = outconv(in_ch=4, out_ch=out_ch)
        self.adain_shared = adain_code_generator_shared()

    def forward(self, input, alpha):
        batch_size = input.shape[0]
        shared_code = self.adain_shared(batch_size)
        x1 = self.down1(input, shared_code, alpha)
        x2 = self.down2(x1, shared_code, alpha)
        x3 = self.down3(x2, shared_code, alpha)
        x4 = self.down4(x3, shared_code, alpha)
        bl_x = self.bottle_neck(x4, shared_code, alpha)
        x = self.up4(bl_x, x3, shared_code, alpha)
        x = self.up3(x, x2, shared_code, alpha)
        x = self.up2(x, x1, shared_code, alpha)
        x = self.up1(x)
        x = self.outconv(x)
        return x


class adain_code_generator_shared(nn.Module):
    def __init__(self):
        super(adain_code_generator_shared, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Sequential(*[nn.Linear(512, 512), nn.LeakyReLU(0.2, inplace=True)])
        self.fc3 = nn.Sequential(*[nn.Linear(512, 512), nn.LeakyReLU(0.2, inplace=True)])
        self.fc4 = nn.Sequential(*[nn.Linear(512, 512), nn.LeakyReLU(0.2, inplace=True)])

    def forward(self, batch_size):
        self.ones_vec = torch.ones((batch_size, 1024)).to(self.fc1.weight.device)
        x = self.fc1(self.ones_vec)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


class adain_code_generator_seperate(nn.Module):
    def __init__(self, ch):
        super(adain_code_generator_seperate, self).__init__()
        self.basic_fc = nn.Sequential(*[nn.Linear(512, 512), nn.LeakyReLU(0.2, inplace=True)])
        self.fc_mean = nn.Linear(512, ch)
        self.fc_var = nn.Linear(512, ch)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, input, shared_code):
        N, C, d, h, w = input.size()
        shared_code = self.basic_fc(shared_code)
        fc_mean = self.fc_mean(shared_code)
        fc_var = self.ReLU(self.fc_var(shared_code))

        # each channel have the same mean value and variance value
        fc_mean_np = fc_mean.view(N, C, 1, 1, 1).expand(N, C, d, h, w)
        fc_var_np = fc_var.view(N, C, 1, 1, 1).expand(N, C, d, h, w)

        return fc_mean_np, fc_var_np


# -------------------------------------------------------

class one_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super(one_conv, self).__init__()
        if stride == 1:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1),
                nn.InstanceNorm3d(out_ch),
            )
        elif stride == 2:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1),
                nn.InstanceNorm3d(out_ch)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class adain(nn.Module):
    def __init__(self, out_ch):
        super(adain, self).__init__()
        self.adain = adain_code_generator_seperate(out_ch)
        self.Leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x_in, shared_code, alpha):
        mean_y, sigma_y = self.adain(x_in, shared_code)
        x_out = sigma_y * (x_in) + mean_y
        x_out = x_out * (alpha) + x_in * (1 - alpha)
        x_out = self.Leakyrelu(x_out)
        return x_out


# ----------------------initial------------------------

class inconv(nn.Module):  # in_conv 1 out_conv 64      (1,256,256,256)    (32,256,256,256)
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv1 = one_conv(in_ch, out_ch, stride=1)
        self.adain = adain(out_ch)

    def forward(self, x, shared_code, alpha):
        x = self.conv1(x)
        x = self.adain(x, shared_code, alpha)
        return x


class Attention(nn.Module):
    def __init__(self, in_ch, num_heads):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv3d(in_ch, in_ch * 3, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) d h w -> b head (d h w) (c)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head (d h w) (c)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head (d h w) (c)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head (d h w) (c) -> b (head c) d h w', head=self.num_heads, d=d, h=h, w=w)
        return out + x


class down_half_polyphase(nn.Module):  # (32,256,256,256)  (64,128,128,128)
    def __init__(self, in_ch, out_ch):
        super(down_half_polyphase, self).__init__()
        self.conv1 = one_conv(in_ch, out_ch, stride=2)
        self.adain = adain(out_ch)

    def forward(self, x, shared_code, alpha):
        x = self.conv1(x)
        x = self.adain(x, shared_code, alpha)
        return x


class down_half_polyphase_end(nn.Module):  # (512,16,16,16)   -> (512,16,16,16)
    def __init__(self, in_ch, out_ch):
        super(down_half_polyphase_end, self).__init__()
        self.attention = Attention(in_ch, 8)
        self.conv1 = one_conv(in_ch, out_ch, stride=1)
        self.adain = adain(out_ch)

    def forward(self, x, shared_code, alpha):
        x = self.attention(x)
        x = self.conv1(x)
        x = self.adain(x, shared_code, alpha)
        return x


class up_half_polyphase(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_half_polyphase, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, padding=1, stride=2)
        self.conv1 = one_conv(out_ch * 2, out_ch, stride=1)
        self.adain = adain(out_ch)

    def forward(self, x1, x2, shared_code, alpha):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.adain(x, shared_code, alpha)
        return x


class up_half_polyphase_final(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_half_polyphase_final, self).__init__()
        self.up = nn.Sequential(
            *[nn.ConvTranspose3d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, padding=1, stride=2),
              nn.Tanh()])
        self.conv1 = one_conv(out_ch * 2, out_ch, stride=1)
        self.adain = adain(out_ch)

    def forward(self, x1, shared_code, alpha):
        x1 = self.up(x1)
        x = torch.cat([x1], dim=1)
        x = self.conv1(x)
        x = self.adain(x, shared_code, alpha)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv3d(in_ch, out_ch, 3, stride=1, padding=1), nn.Tanh()])

    def forward(self, x):
        x = self.conv(x)
        return x
