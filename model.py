

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torchsummary import summary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCHSIZE=48
class GaborConv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, device="cuda", stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(GaborConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode)
        self.freq = nn.Parameter(
            (3.14 / 2) * 1.41 ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor))
        self.theta = nn.Parameter((3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor))
        self.psi = nn.Parameter(3.14 * torch.rand(out_channels, in_channels))
        self.sigma = nn.Parameter(3.14 / self.freq)
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0]
        self.device = device

    def forward(self, input_image):
        y, x = torch.meshgrid([torch.linspace(-self.x0 + 1, self.x0, self.kernel_size[0]),
                               torch.linspace(-self.y0 + 1, self.y0, self.kernel_size[1])])
        x = x.to(self.device)
        y = y.to(self.device)
        weight = torch.empty(self.weight.shape, requires_grad=False).to(self.device)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = self.sigma[i, j].expand_as(y)
                freq = self.freq[i, j].expand_as(y)
                theta = self.theta[i, j].expand_as(y)
                psi = self.psi[i, j].expand_as(y)

                rotx = x * torch.cos(theta) + y * torch.sin(theta)
                roty = -x * torch.sin(theta) + y * torch.cos(theta)

                g = torch.zeros(y.shape)

                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + 1e-3) ** 2))
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * 3.14 * sigma ** 2)
                weight[i, j] = g
                self.weight.data[i, j] = g
        return F.conv2d(input_image, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class Generator(torch.nn.Module):
    def __init__(self,input):
        super(Generator, self).__init__()

        self.fc_layer1=torch.nn.Linear(input,32*32)
        self.dropout = torch.nn.Dropout(p=0.5)
        # self.fc_layer2=torch.nn.Linear(100,64*64)
        self.activation = torch.nn.LeakyReLU()
        self.activation2=torch.nn.Tanh()
        self.h1 = GaborConv2d(in_channels=1, out_channels=128, kernel_size=4,stride=2, padding=1)
        # self.h8 = GaborConv2d(in_channels=128,out_channels=1, kernel_size=4, stride=2, padding=1)
        # self.h1=torch.nn.Conv2d(1,128,kernel_size=4,stride=2,padding=1)
        self.b1=torch.nn.BatchNorm2d(128)

        self.h2 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.b2 = torch.nn.BatchNorm2d(256)

        self.h3 = torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.b3 = torch.nn.BatchNorm2d(512)


        self.h4 = torch.nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
        self.b4 = torch.nn.BatchNorm2d(1024)


        self.h5 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.b5 = torch.nn.BatchNorm2d(512)

        self.h6=torch.nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1)
        self.b6=torch.nn.BatchNorm2d(256)

        self.h7 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.b7 = torch.nn.BatchNorm2d(128)

        self.h8 = torch.nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
        self.b8 = torch.nn.BatchNorm2d(1)

        # self.h9 = torch.nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
        # self.b9 = torch.nn.BatchNorm2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,c,BATCHSIZE):
        c = c.reshape(BATCHSIZE, 1, 100*105)


        fc1=self.fc_layer1(c)
        fc1=self.activation(fc1)
        # z1=self.fc_layer2(z)
        # x=torch.cat([z1,fc1],1)

        x=fc1.reshape(BATCHSIZE,1,32,32)
        x=self.h1(x)
        x=self.b1(x)
        x=self.activation(x)
        x=self.dropout(x)

        x=self.h2(x)
        x=self.b2(x)
        x=self.activation(x)
        x = self.dropout(x)

        x = self.h3(x)
        x = self.b3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.h4(x)
        x = self.b4(x)
        x = self.activation(x)
        x = self.dropout(x)


        x = self.h5(x)
        x = self.b5(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.h6(x)
        x = self.b6(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.h7(x)
        x = self.b7(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.h8(x)
        x = self.b8(x)
        out = self.activation(x)

        # x = self.h9(x)
        # out = self.b9(x)
        # out = self.activation(x)

        return out

class MLP_GCNN(nn.Module):
    def __init__(self,input):
        super(MLP_GCNN, self).__init__()

        self.fc1=nn.Linear(input,8000)
        self.relu=nn.LeakyReLU()
        self.fc2=nn.Linear(8000,4000)
        self.fc3 = nn.Linear(4000, 1600)
        # self.fc3=nn.Linear(6000,4000)
        #
        # self.fc4=nn.Linear(4000,1600)
        self.dropout=nn.Dropout(p=0.25)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1=nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.m2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool=nn.MaxPool2d(2,stride=2,return_indices=True)
        self.batchnomalazation=nn.LayerNorm

        # self.gabor_cnn=nn.Sequential(
        self.gcnn0=GaborConv2d(in_channels=1, out_channels=64, kernel_size=(5, 5),padding=2, device=device)
        self.gcnn1 = GaborConv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=2, device=device)
        self.gcnn2 = GaborConv2d(in_channels=64, out_channels=1, kernel_size=(5, 5), padding=2, device=device)
        #self.gcnn1 = GaborConv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding=1, device=device)
        #self.gcnn2 = GaborConv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=1, device=device)


        self.cnn1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(5,5),padding=2,device=device)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=2, device=device)
        self.cnn3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5), padding=2, device=device)
        self.cnn4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(5, 5), padding=2, device=device)
        self.cnn5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=2, device=device)
        self.cnn6 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(5, 5), padding=2, device=device)


        # self.cnn2=nn.Conv2d(in_channels=16)
        self.uppool=nn.MaxUnpool2d(2,stride=2,padding=0)

    def forward(self,x,batchsize):

        out=self.fc1(x)
        out=self.relu(out)
        out=self.fc2(out)
        out=self.relu(out)
        #out=self.fc3(out)
       # print(out.shape)
        out=self.fc3(out)
        out=self.dropout(out)
        out=self.relu(out)
        # out=self.fc4(out)

        out=out.reshape(batchsize,1,40,40)
        # out=self.gabor_cnn(out)


        out = self.cnn1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.cnn2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.cnn3(out)
        out = self.bn3(out)
        #        out=self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        size=out.size()
        _, indices = self.pool(torch.empty(size[0], size[1],size[2]*2,size[3]*2))
        out=self.uppool(out,indices.to(device))

        out = self.cnn4(out)
        out = self.bn2(out)
        #        out=self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.cnn5(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.gcnn2(out)
        out = self.bn0(out)

        return out
