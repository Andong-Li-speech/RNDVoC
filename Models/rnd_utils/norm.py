import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init


class ChannelNormalization(nn.Module):
    def __init__(self, num_channels, ndim=3, affine=True):
        super(ChannelNormalization, self).__init__()
        self.num_channels = num_channels
        self.ndim = ndim
        self.affine = affine
        self.eps = 1e-5
        if affine:
            if ndim == 3:
                self.gain = Parameter(torch.empty([1, num_channels, 1]))
                self.bias = Parameter(torch.empty([1, num_channels, 1]))
            elif ndim == 4:
                self.gain = Parameter(torch.empty([1, num_channels, 1, 1]))
                self.bias = Parameter(torch.empty([1, num_channels, 1, 1]))
        else:
            self.register_parameter('gain', None)
            self.register_parameter('bias', None)
        # 
        self.reset_parameters()

    def reset_parameters(self):
        if self.gain is not None and self.bias is not None:
            init.constant_(self.gain, 1.)
            init.constant_(self.bias, 0.)

    def forward(self, input):
        """
        input: (B, C, T) or (B, C, X, T)
        return: xxx
        """
        if input.ndim == 3:
            mean_ = input.mean(dim=1, keepdims=True)
            std_ = torch.sqrt(torch.var(input, dim=1, keepdims=True, unbiased=False) + self.eps)
        elif input.ndim == 4:
            mean_ = input.mean(dim=1, keepdims=True)
            std_ = torch.sqrt(torch.var(input, dim=1, keepdims=True, unbiased=False) + self.eps)
        x = (input - mean_) / std_

        if self.affine:
            x = x * self.gain + self.bias

        return x


class TimeGlobalNormalization(nn.Module):
    def __init__(self, num_channels, ndim=3, affine=True):
        super(TimeGlobalNormalization, self).__init__()
        self.num_channels = num_channels
        self.ndim = ndim
        self.affine = affine
        self.eps = 1e-5
        #
        if affine:
            if ndim == 3:
                self.gain = Parameter(torch.empty([1, 1, num_channels]))
                self.bias = Parameter(torch.empty([1, 1, num_channels]))
            elif ndim == 4:
                self.gain = Parameter(torch.empty([1, 1, 1, num_channels]))
                self.bias = Parameter(torch.empty([1, 1, 1, num_channels]))
        else:
            self.register_parameter('gain', None)
            self.register_parameter('bias', None)
        #
        self.reset_parameters()

    def reset_parameters(self):
        if self.gain is not None and self.bias is not None:
            init.constant_(self.gain, 1.)
            init.constant_(self.bias, 0.)

    def forward(self, input):
        """
        input: (B, T, C) or (B, nband, T, C)
        return: (B, T, C) or (B, nband, T, C)
        """
        if input.ndim == 3:
            mean_ = input.mean(dim=[1, 2], keepdims=True)
            std_ = torch.sqrt(torch.var(input, dim=[1, 2], keepdims=True, unbiased=False) + self.eps)
        elif input.ndim == 4:
            mean_ = input.mean(dim=[2, 3], keepdims=True)
            std_ = torch.sqrt(torch.var(input, dim=[2, 3], keepdims=True, unbiased=False) + self.eps)
        x = (input - mean_) / std_

        if self.affine:
            x = x * self.gain + self.bias

        return x


class BandwiseLayerNorm(nn.Module):
    def __init__(self,
                 nband: int,
                 feature_dim: int,
                 affine = True,
                 ):
        super(BandwiseLayerNorm, self).__init__()
        self.nband = nband
        self.feature_dim = feature_dim
        self.affine = affine
        self.eps = 1e-5
        self.gain_matrix = Parameter(torch.ones([1, nband, feature_dim, 1]))
        self.bias_matrix = Parameter(torch.zeros([1, nband, feature_dim, 1]))

    def forward(self, input, nband=None):
        """
        input: (B*nband, C, T)
        nband: int or None, current nband, for SFI case
        return: (B*nband, C, T)
        """
        mean_ = torch.mean(input, dim=-2, keepdim=True)  # (B*nband, 1, T)
        std_ = torch.sqrt(torch.var(input, dim=-2, unbiased=False, keepdim=True) + self.eps)  # (B*nband, 1, T)

        b_size_, nch, seq_len = input.shape
        mean_ = mean_.view(int(b_size_/self.nband), self.nband, 1, -1)
        std_ = std_.view(int(b_size_/self.nband), self.nband, 1, -1)
        input = input.view(int(b_size_/self.nband), self.nband, input.shape[-2], -1) # (b_size, nband, C, T)

        if self.affine:
            if nband is None:
                output = self.gain_matrix * ((input - mean_) / std_) + self.bias_matrix
            else:
                output = self.gain_matrix[:, :nband] * ((input - mean_) / std_) + self.bias_matrix[:, :nband]
        else:
            output = (input - mean_) / std_
        
        return output.view(b_size_, nch, seq_len)


class BandwiseC2LayerNorm(nn.Module):
    def __init__(self,
                 nband: int,
                 feature_dim: int,
                 affine = True,
                 ):
        super(BandwiseC2LayerNorm, self).__init__()
        self.nband = nband
        self.feature_dim = feature_dim
        self.affine = affine
        self.eps = 1e-5
        self.gain_matrix = Parameter(torch.ones([1, feature_dim, nband, 1]))
        self.bias_matrix = Parameter(torch.zeros([1, feature_dim, nband, 1]))

    def forward(self, input):
        """
        input: (B, C, nband, T)
        return: (B, C, nband, T)
        """
        mean_ = torch.mean(input, dim=1, keepdim=True)  # (B, 1, nband, T)
        std_ = torch.sqrt(torch.var(input, dim=1, unbiased=False, keepdim=True) + self.eps)  # (B, 1, nband, T)

        if self.affine:
            output = self.gain_matrix * ((input - mean_) / std_) + self.bias_matrix 
        else:
            output = (input - mean_) / std_
        
        return output