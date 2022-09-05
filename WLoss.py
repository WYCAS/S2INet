# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import warnings
import cv2
import torch
import torch.nn.functional as F
import numpy as np
RFW=[[4,6],[7,5],[20,30],[2,10],[3,16]]
imageW=32
batchsize=48
def _fspecial_gauss_1d(size, sigma):
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):

    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


# def _ehs(x, y):
#     """
#     Entropy-Histogram Similarity measure
#     """
#     x=x.cpu().detach().numpy()
#     y=y.cpu().detach().numpy()
#     H = (np.histogram2d(x.flatten(), y.flatten()))[0]
# 
#     return torch.tensor(-np.sum(np.nan_to_num(H * np.log2(H))))

def build_filters():
    filters = []
    ksize = [7, 9, 11, 13]  # gabor尺度，6个
    lamda = np.pi / 2.0  # 波长
    for theta in np.arange(0, np.pi, np.pi / 4):  # gabor方向，0°，45°，90°，135°，共四个
        for K in range(4):
            kern = cv2.getGaborKernel((ksize[K], ksize[K]), 1.0, theta, lamda, 0.5, 0, ktype=cv2.CV_32F)

            kern /= 1.5 * kern.sum()








class Weightloss(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.2),
        nonnegative_ssim=False,
    ):
        super(Weightloss, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

        weight = np.ones([imageW, imageW])  # .scatter_(1,RF,  1)
        for index in range(0, len(RFW)):
            i = RFW[index][0]
            j = RFW[index][1]
            weight[i - 1, j] = 3
            weight[i, j - 1] = 3
            weight[i, j] = 5
            weight[i, j + 1] = 3
            weight[i + 1, j] = 3
        weight = weight.reshape(1, 1, imageW, imageW)
        self.Wmse = np.expand_dims(weight, 0).repeat(batchsize, axis=0)

    def forward(self, X, Y):
        return Weightloss.ssim(self,
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )

    def weightmse(self,x, y):

        batchsize = len(x[:, 0])

        # kernel = torch.tensor([[3, 3, 3], [3, 5, 3], [3, 3, 3]])
        # RF = torch.tensor([[2, 3, 4], [2, 3, 4], [2, 3, 4]])
        #
        # weight = torch.ones(imageW, imageW, dtype=kernel.dtype).scatter_(1, RF, kernel)
        #
        # weight = weight.reshape(1, 1, imageW, imageW)
        # weight = weight.expand(batchsize, 1, imageW, imageW)

        weight = torch.from_numpy(self.Wmse).cuda()
        weightmse = torch.mean(((x - y) ** 2) * weight)

        return weightmse

    def _ssim(self,X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):
        K1, K2 = K
        # batch, channel, [depth,] height, width = X.shape
        compensation = 1.0

        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2

        win = win.to(X.device, dtype=X.dtype)

        mu1 = gaussian_filter(X, win)
        mu2 = gaussian_filter(Y, win)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
        sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
        sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
        ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

        ssim_per_channel =Weightloss.weightmse(self,X, Y)#0.2 * torch.flatten(ssim_map, 2).mean(-1) + 0.8 * Weightloss.weightmse(self,X, Y)
        cs = torch.flatten(cs_map, 2).mean(-1)
        return ssim_per_channel, cs

    def ssim(self,
            X,
            Y,
            data_range=255,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            win=None,
            K=(0.01, 0.03),
            nonnegative_ssim=False,
    ):

        if not X.shape == Y.shape:
            raise ValueError("Input images should have the same dimensions.")

        for d in range(len(X.shape) - 1, 1, -1):
            X = X.squeeze(dim=d)
            Y = Y.squeeze(dim=d)

        if len(X.shape) not in (4, 5):
            raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

        if not X.type() == Y.type():
            raise ValueError("Input images should have the same dtype.")

        if win is not None:  # set win_size
            win_size = win.shape[-1]

        if not (win_size % 2 == 1):
            raise ValueError("Window size should be odd.")

        if win is None:
            win = _fspecial_gauss_1d(win_size, win_sigma)
            win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

        ssim_per_channel, cs = Weightloss._ssim(self,X, Y, data_range=data_range, win=win, size_average=False, K=K)
        if nonnegative_ssim:
            ssim_per_channel = torch.relu(ssim_per_channel)

        if size_average:
            return ssim_per_channel.mean()
        else:
            return ssim_per_channel.mean(1)


