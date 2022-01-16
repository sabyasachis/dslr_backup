import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torchvision
import numpy as np
import pdb
from utils import *


# --------------------------------------------------------------------------
# Core Models 
# --------------------------------------------------------------------------

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args

        # Encoder architecture
        mult = 1
        nz = args.z_dim * mult
        nc = 3 if args.no_polar else 2
        ndf = 64
        lf = (1,32)
        self.encoder_conv2d_1    = nn.Sequential(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.encoder_leakyrelu_2 = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_3    = nn.Sequential(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))

        self.encoder_batchnorm2d_4 = nn.Sequential(nn.BatchNorm2d(ndf * 2))
        self.encoder_leakyrelu_5   = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_6      = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))

        self.encoder_batchnorm2d_7 = nn.Sequential(nn.BatchNorm2d(ndf * 4))
        self.encoder_leakyrelu_8   = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))
        self.encoder_conv2d_9      = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, (3,4), 2, (0,1), bias=False))

        self.encoder_batchnorm2d_10 = nn.Sequential(nn.BatchNorm2d(ndf * 8))
        self.encoder_leakyrelu_11   = nn.Sequential(nn.LeakyReLU(0.2, inplace=True))

        self.encoder_conv2d_12      = nn.Sequential(nn.Conv2d(ndf * 8, nz, lf, 1, 0, bias=False))

        # Decoder architecture
        ngf=64
        base=4
        ff=(1,32)
        nz=args.z_dim
        nc=2
        self.decoder_convtranspose2d_13 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 8, ff, 1, 0, bias=False))
        self.decoder_batchnorm2d_14     = nn.Sequential(nn.BatchNorm2d(ngf * 8))
        self.decoder_relu_15            = nn.Sequential(nn.ReLU(True))


        self.decoder_convtranspose2d_16 = nn.Sequential(nn.ConvTranspose2d(ngf * 8, ngf * 4, (4,4), stride=2, padding=(0,1), bias=False))
        self.decoder_batchnorm2d_17     = nn.Sequential(nn.BatchNorm2d(ngf * 4))
        self.decoder_relu_18            = nn.Sequential(nn.ReLU(True))

        self.decoder_convtranspose2d_19 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, (4,4), stride=2, padding=(1,1), bias=False))
        self.decoder_batchnorm2d_20     = nn.Sequential(nn.BatchNorm2d(ngf * 2))
        self.decoder_relu_21            = nn.Sequential(nn.ReLU(True))


        self.decoder_convtranspose2d_22 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf * 1, (4,4), stride=2, padding=(1,1), bias=False))
        self.decoder_batchnorm2d_23     = nn.Sequential(nn.BatchNorm2d(ngf * 1))
        self.decoder_relu_24            = nn.Sequential(nn.ReLU(True))


        self.decoder_convtranspose2d_25 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        self.tanh_26                    = nn.Sequential(nn.Tanh())


    def forward(self, x):
        # Encoder network forward pass        # Input size : (3,29,512)
        x1 = self.encoder_conv2d_1(x)         # Size start : (64,14,256)
        x2 = self.encoder_leakyrelu_2(x1)
        x3 = self.encoder_conv2d_3(x2)        # Downsample : (128,7,128)

        x4 = self.encoder_batchnorm2d_4(x3)
        x5 = self.encoder_leakyrelu_5(x4)
        x6 = self.encoder_conv2d_6(x5)        # Downsample : (256,3,64)

        x7 = self.encoder_batchnorm2d_7(x6)
        x8 = self.encoder_leakyrelu_8(x7)
        x9 = self.encoder_conv2d_9(x8)        # Downsample : (512,1,32)

        x10 = self.encoder_batchnorm2d_10(x9)
        x11 = self.encoder_leakyrelu_11(x10)

        x12 = self.encoder_conv2d_12(x11)     # Hidden size : (Z_dim,1,1)
        z = x12

        while z.dim() != 2: 
            z = z.squeeze(-1)     

        # Decoder network forward pass
        x13 = self.decoder_convtranspose2d_13(x12)    # Size restart : (512,1,32)
        x14 = self.decoder_batchnorm2d_14(x13)
        x15 = self.decoder_relu_15(x14)

        x16 = self.decoder_convtranspose2d_16(x15)    # Upsample     : (256,4,64)
        x17 = self.decoder_batchnorm2d_17(x16)
        x18 = self.decoder_relu_18(x17)

        x19 = self.decoder_convtranspose2d_19(x18)    # Upsample     : (128,8,128)
        x20 = self.decoder_batchnorm2d_20(x19)
        x21 = self.decoder_relu_21(x20)

        x22 = self.decoder_convtranspose2d_22(x21)    # Upsample     : (64,16,256)
        x23 = self.decoder_batchnorm2d_23(x22)
        x24 = self.decoder_relu_24(x23)

        x25 = self.decoder_convtranspose2d_25(x24)    # Output size  : (2,32,512)
        x26 = self.tanh_26(x25)
        recon = x26

        return recon, None, z





    def sample(self, nb_samples=16, tmp=1):
        noise = torch.cuda.FloatTensor(nb_samples, self.args.z_dim).normal_(0, tmp)
        return self.decode(noise)

    @staticmethod
    def gaussian_kl(mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
    @staticmethod
    def log_gauss(z, params):
        [mu, std] = params
        return - 0.5 * (t.pow(z - mu, 2) * t.pow(std + 1e-8, -2) + 2 * t.log(std + 1e-8) + math.log(2 * math.pi)).sum(1) 


# --------------------------------------------------------------------------
# Baseline (AtlasNet), taken from https://github.com/ThibaultGROUEIX/AtlasNet
# --------------------------------------------------------------------------
class PointNetfeat_(nn.Module):
    def __init__(self, num_points = 40 * 256, global_feat = True):
        super(PointNetfeat_, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        batchsize = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 128):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class AE_AtlasNet(nn.Module):
    def __init__(self, num_points = 40 * 256, bottleneck_size = 1024, nb_primitives = 2, AE=True):
        super(AE_AtlasNet, self).__init__()
        bot_enc = bottleneck_size if AE else 2 * bottleneck_size
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.encoder = nn.Sequential(
        PointNetfeat_(num_points, global_feat=True),
        nn.Linear(1024, bot_enc),
        nn.BatchNorm1d( bot_enc),
        nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 + self.bottleneck_size) for i in range(0,self.nb_primitives)])


    def encode(self, x):
        if x.dim() == 4 : 
            if x.size(1) != 3: 
                assert x.size(-1) == 3 
                x = x.permute(0, 3, 1, 2).contiguous()
            x = x.reshape(x.size(0), 3, -1)
        else: 
            if x.size(1) != 3: 
                assert x.size(-1) == 3 
                x = x.transpose(-1, -2).contiguous()
        
        x = self.encoder(x)
        return x

    def decode(self, x):
        outs = []
        for i in range(0,self.nb_primitives):
            rand_grid = (torch.cuda.FloatTensor(x.size(0),2,self.num_points // self.nb_primitives))
            rand_grid.data.uniform_(0,1)
            y = x.unsqueeze(2).expand(x.size(0),x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs,2).contiguous().transpose(2,1).contiguous()



if __name__ == '__main__':
    points = torch.cuda.FloatTensor(10, 3, 40, 256).normal_()
    AE = AE_AtlasNet(num_points = 40 * 256).cuda()
    out = AE(points)
    loss = get_chamfer_dist()(points, out)
    x =1


# --------------------------------------------------------------------------
# Baseline (Panos's paper)
# --------------------------------------------------------------------------
class PointGenPSG2(nn.Module):
    def __init__(self, nz=100, num_points = 40 * 256):
        super(PointGenPSG2, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(nz, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, self.num_points * 3 // 2)

        self.fc11 = nn.Linear(nz, 256)
        self.fc21 = nn.Linear(256, 512)
        self.fc31 = nn.Linear(512, 1024)
        self.fc41 = nn.Linear(1024, self.num_points * 3 // 2)
        self.th = nn.Tanh()
        self.nz = nz
        
    
    def forward(self, x):
        batchsize = x.size()[0]
        
        x1 = x
        x2 = x
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        x1 = self.th(self.fc4(x1))
        x1 = x1.view(batchsize, 3, -1)

        x2 = F.relu(self.fc11(x2))
        x2 = F.relu(self.fc21(x2))
        x2 = F.relu(self.fc31(x2))
        x2 = self.th(self.fc41(x2))
        x2 = x2.view(batchsize, 3, -1)

        return torch.cat([x1, x2], 2)
