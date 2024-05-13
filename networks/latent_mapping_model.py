import torch
import logging
from torch import nn
from einops import rearrange


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)
    # return torch.nn.Identity()


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class ResBlock(nn.Module):
    """
    Convolutional blocks
    """
    def __init__(self, in_channels, out_channels, leaky=True):
        super().__init__()
        # activation, support PReLU and common ReLU
        self.act1 = nn.PReLU() if leaky else nn.ReLU(inplace=True)
        self.act2 = nn.PReLU() if leaky else nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            Normalize(in_channels),
            self.act1,
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        )
        
        self.conv2 = nn.Sequential(
            Normalize(out_channels),
            self.act2,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        )

        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return self.skip_connection(x) + h


class ResAttBlock(nn.Module):
    """
    Convolutional blocks
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.resblock  = ResBlock(in_channels=in_channels, out_channels=out_channels)
        self.attention = SpatialSelfAttention(out_channels)


    def forward(self, x):
        h = self.resblock(x)
        h = self.attention(h)
        return h


class ResAttnUNet(nn.Module):
    def __init__(self, in_channel=8, out_channels=8, num_res_blocks=2, ch=32, ch_mult=(1,2,4,4)) -> None:
        super(ResAttnUNet, self).__init__()
        self.ch = ch
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(ch_mult) * [num_res_blocks]

        self.input_blocks = nn.Conv2d(in_channel, ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1_0 = ResAttBlock(in_channels=ch,              out_channels=ch * ch_mult[0])
        self.conv2_0 = ResAttBlock(in_channels=ch * ch_mult[0], out_channels=ch * ch_mult[1])
        self.conv3_0 = ResAttBlock(in_channels=ch * ch_mult[1], out_channels=ch * ch_mult[2])
        self.conv4_0 = ResAttBlock(in_channels=ch * ch_mult[2], out_channels=ch * ch_mult[3])

        self.conv3_1 = ResAttBlock(in_channels=ch * (ch_mult[2] + ch_mult[3]), out_channels=ch * ch_mult[2])
        self.conv2_2 = ResAttBlock(in_channels=ch * (ch_mult[1] + ch_mult[2]), out_channels=ch * ch_mult[1])
        self.conv1_3 = ResAttBlock(in_channels=ch * (ch_mult[0] + ch_mult[1]), out_channels=ch * ch_mult[0])
        self.conv0_4 = ResAttBlock(in_channels=ch * (1          + ch_mult[0]), out_channels=ch * 1)

        self.output_blocks = nn.Sequential(
            Normalize(ch*1),
            nn.SiLU(),
            nn.Conv2d(ch*1, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self._initialize_weights()
        self._print_networks(verbose=False)

    def forward(self, x) -> torch.Tensor:
        x0 = self.input_blocks(x)
        x1 = self.conv1_0(x0)
        x2 = self.conv2_0(x1)
        x3 = self.conv3_0(x2)
        x4 = self.conv4_0(x3)

        x3_1 = self.conv3_1(torch.cat([x3, x4], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, x3_1], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, x2_2], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, x1_3], dim=1))

        out = dict()
        out['out'] = self.output_blocks(x0_4)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _print_networks(self, verbose=False) -> None:
        logging.info('---------- Networks initialized -------------')
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        if verbose:
            logging.info(self.modules())
        logging.info('Total number of parameters : %.3f M' % (num_params / 1e6))
        logging.info('-----------------------------------------------')


class ResAttnUNet_DS(nn.Module):
    def __init__(self, in_channel=8, out_channels=8, num_res_blocks=2, ch=32, ch_mult=(1,2,4,4)) -> None:
        super(ResAttnUNet_DS, self).__init__()
        self.ch = ch
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(ch_mult) * [num_res_blocks]

        self.input_blocks = nn.Conv2d(in_channel, ch, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1_0 = ResAttBlock(in_channels=ch,              out_channels=ch * ch_mult[0])
        self.conv2_0 = ResAttBlock(in_channels=ch * ch_mult[0], out_channels=ch * ch_mult[1])
        self.conv3_0 = ResAttBlock(in_channels=ch * ch_mult[1], out_channels=ch * ch_mult[2])
        self.conv4_0 = ResAttBlock(in_channels=ch * ch_mult[2], out_channels=ch * ch_mult[3])

        self.conv3_1 = ResAttBlock(in_channels=ch * (ch_mult[2] + ch_mult[3]), out_channels=ch * ch_mult[2])
        self.conv2_2 = ResAttBlock(in_channels=ch * (ch_mult[1] + ch_mult[2]), out_channels=ch * ch_mult[1])
        self.conv1_3 = ResAttBlock(in_channels=ch * (ch_mult[0] + ch_mult[1]), out_channels=ch * ch_mult[0])
        self.conv0_4 = ResAttBlock(in_channels=ch * (1          + ch_mult[0]), out_channels=ch * 1)

        self.convds3 = nn.Sequential(Normalize(ch * ch_mult[2]), nn.SiLU(),
                                     nn.Conv2d(ch * ch_mult[2], out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.convds2 = nn.Sequential(Normalize(ch * ch_mult[1]), nn.SiLU(),
                                     nn.Conv2d(ch * ch_mult[1], out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.convds1 = nn.Sequential(Normalize(ch * ch_mult[0]), nn.SiLU(),
                                     nn.Conv2d(ch * ch_mult[0], out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.convds0 = nn.Sequential(Normalize(ch * 1), nn.SiLU(),
                                     nn.Conv2d(ch * 1, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        

        self._initialize_weights()
        self._print_networks(verbose=False)

    def forward(self, x) -> torch.Tensor:
        x0 = self.input_blocks(x)
        x1 = self.conv1_0(x0)
        x2 = self.conv2_0(x1)
        x3 = self.conv3_0(x2)
        x4 = self.conv4_0(x3)

        x3_1 = self.conv3_1(torch.cat([x3, x4], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, x3_1], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, x2_2], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, x1_3], dim=1))

        out = dict()
        out['level3'] = self.convds3(x3_1)
        out['level2'] = self.convds2(x2_2)
        out['level1'] = self.convds1(x1_3)
        out['out']    = self.convds0(x0_4)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _print_networks(self, verbose=False) -> None:
        logging.info('---------- Networks initialized -------------')
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        if verbose:
            logging.info(self.modules())
        logging.info('Total number of parameters : %.3f M' % (num_params / 1e6))
        logging.info('-----------------------------------------------')

if __name__ == '__main__':
    # Get UNet shape 
    model = ResAttnUNet_DS(
        in_channel=4, 
        out_channels=4, 
        num_res_blocks=2, 
        ch=32, 
        ch_mult=(1,2,4,4)
    )
    
    out_dict = model(torch.ones(2, 4, 64, 64))
    for key in out_dict.keys():
        print('{}, shape: {}'.format(
            key,
            out_dict[key].shape
        ))

   