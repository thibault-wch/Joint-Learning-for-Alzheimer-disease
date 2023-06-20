import functools

import torch.nn as nn


# Defines the definedGAN discriminator with the specified arguments.
class DefinedDiscriminator(nn.Module):
    def __init__(self, input_nc=1, output_nc=64, use_sigmoid=False):
        super(DefinedDiscriminator, self).__init__()
        self.inconv = nn.Sequential(*[nn.Conv3d(input_nc, 32, kernel_size=3, stride=2, padding=1),
                                      nn.InstanceNorm3d(32),
                                      nn.LeakyReLU(0.2, True)])
        self.disc_CBR1 = disc_CBR(32, 128, kernel_size=3, stride=2, padding=1)
        self.disc_CBR2 = disc_CBR(128, 256, kernel_size=3, stride=2, padding=1)
        self.disc_CBR3 = disc_CBR(256, 512, kernel_size=3, stride=2, padding=1)
        self.disc_CBR4 = disc_CBR(512, 512, kernel_size=3, stride=2, padding=1)
        self.disc_CBR5 = disc_CBR(512, 512, kernel_size=3, stride=1, padding=1)
        self.outconvlist = [nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1)]
        if use_sigmoid:
            self.outconvlist.append(nn.Sigmoid())
        self.outconv = nn.Sequential(*self.outconvlist)

    def forward(self, x):
        x = self.inconv(x)
        x = self.disc_CBR1(x)
        x = self.disc_CBR2(x)
        x = self.disc_CBR3(x)
        x = self.disc_CBR4(x)
        x = self.disc_CBR5(x)
        x = self.outconv(x)

        return x


class disc_CBR(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=1, padding=1):
        super(disc_CBR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, stride=stride, bias=True),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


# patchDiscriminator
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# PixelDiscriminator
class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)
