import math
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init

from .norm import *


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
    

class LinearGroup(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True):
        super(LinearGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = Parameter(torch.empty([num_groups, out_features, in_features]))
        if bias:
            self.bias = Parameter(torch.empty([num_groups, out_features]))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        input: (BT, G, nband)
        return: (BT, G, nband)
        """
        x = torch.einsum('...gh,gkh->...gk', [input, self.weight])
        if self.bias is not None:
            x = x + self.bias[None, ...]

        return x


class BandSplit_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int,
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               ):
      super(BandSplit_22k, self).__init__()
      self.sr = sr      
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      fft_reso = sr / n_fft
      bw_250 = int(np.floor(250 / fft_reso))  # 11 bands
      bw_500 = int(np.floor(500 / fft_reso))  # 23 bands
      bw_1k = int(np.floor(1000 / fft_reso))  # 46 bands

      # total 24 bands
      self.band_width = [bw_250] * 12  # 3k  0~11
      self.band_width += [bw_500] * 8  # 4k  12~19
      self.band_width += [bw_1k] * 3  # 3k   20~22
      self.band_width.append(self.n_fft // 2 + 1 - np.sum(self.band_width))  # remains

      self.nband = len(self.band_width)
      print(f'Totally splitting {len(self.band_width)} bands for sampling rate: 22.05k.')
      
      self.encoder = nn.ModuleList([])
      for i in range(self.nband):
            self.encoder.append(
               nn.Sequential(
                  ChannelNormalization(self.band_width[i] * 2 + 1),
                  nn.Conv1d(self.band_width[i] * 2 + 1, self.feature_dim, 1)
               )
            )

   def get_nband(self):
      return self.nband

   def forward(self, input=None):
      """
      input: (B, F, T, 2)
      log_input: (B, F, T)
      return: (B, nband, C, T)
      """
      subband_spec_list = []
      band_idx = 0
      for i in range(len(self.band_width)):
            cur_subband_spec = input[:, band_idx: band_idx + self.band_width[i]].transpose(1, 2).contiguous() # (B, T, fw, 2)
            cur_subband_spec_power = torch.sqrt(torch.norm(cur_subband_spec, dim=-1, keepdim=True).pow(2).sum(dim=-2, keepdim=True) + self.eps)  # (B, T, 1, 1)
            b_size, seq_len, _, _ = cur_subband_spec.shape
            cur_subband_spec_ = (cur_subband_spec / cur_subband_spec_power).view(b_size, seq_len, -1)
            cur_subband_spec_ = torch.cat([cur_subband_spec_, torch.log(cur_subband_spec_power.squeeze(-1))], dim=-1).transpose(-2, -1).contiguous()
            subband_spec_list.append(self.encoder[i](cur_subband_spec_))  
            band_idx += self.band_width[i]

      out = torch.stack(subband_spec_list, dim=1)  # (B, nband, C, T)

      return out


class BandSplit_24k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int,
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               ):
      super(BandSplit_24k, self).__init__()
      self.sr = sr      
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      fft_reso = sr / n_fft
      bw_250 = int(np.floor(250 / fft_reso))  # 11 bands
      bw_500 = int(np.floor(500 / fft_reso))  # 23 bands
      bw_1k = int(np.floor(1000 / fft_reso))  # 46 bands

      # total 24 bands
      self.band_width = [bw_250] * 12  # 3k  0~11
      self.band_width += [bw_500] * 8  # 4k  12~19
      self.band_width += [bw_1k] * 4  # 4k   20~22
      self.band_width.append(self.n_fft // 2 + 1 - np.sum(self.band_width))  # remains

      self.nband = len(self.band_width)
      print(f'Totally splitting {len(self.band_width)} bands for sampling rate: 24k.')

      self.encoder = nn.ModuleList([])
      for i in range(self.nband):
            self.encoder.append(
               nn.Sequential(
                  ChannelNormalization(self.band_width[i] * 2 + 1),
                  nn.Conv1d(self.band_width[i] * 2 + 1, self.feature_dim, 1)
               )
            )

   def get_nband(self):
      return self.nband

   def forward(self, input=None):
      """
      input: (B, F, T, 2)
      log_input: (B, F, T)
      return: (B, nband, C, T)
      """
      subband_spec_list = []
      band_idx = 0
      for i in range(len(self.band_width)):
            cur_subband_spec = input[:, band_idx: band_idx + self.band_width[i]].transpose(1, 2).contiguous() # (B, T, fw, 2)
            cur_subband_spec_power = torch.sqrt(torch.norm(cur_subband_spec, dim=-1, keepdim=True).pow(2).sum(dim=-2, keepdim=True) + self.eps)  # (B, T, 1, 1)
            b_size, seq_len, _, _ = cur_subband_spec.shape
            cur_subband_spec_ = (cur_subband_spec / cur_subband_spec_power).view(b_size, seq_len, -1)
            cur_subband_spec_ = torch.cat([cur_subband_spec_, torch.log(cur_subband_spec_power.squeeze(-1))], dim=-1).transpose(-2, -1).contiguous()
            subband_spec_list.append(self.encoder[i](cur_subband_spec_))  
            band_idx += self.band_width[i]

      out = torch.stack(subband_spec_list, dim=1)  # (B, nband, C, T)

      return out


class SharedBandSplit_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int,
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               ):
      super(SharedBandSplit_22k, self).__init__()
      self.sr = sr      
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      self.reg1_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(12, 3), stride=(12, 1)),
          BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim)
      )
      self.reg2_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(24, 3), stride=(24, 1)),
          BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim)
      )
      self.reg3_encoder = nn.Sequential(
          nn.ConstantPad2d([1, 1, 0, 0], value=0.),
          nn.Conv2d(in_channels=2, out_channels=self.feature_dim, kernel_size=(44, 3), stride=(44, 1)),
          BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim)
      )

      self.nband = 12 + 8 + 4
      print(f'Totally splitting {self.nband} bands for sampling rate: 22.05k.')

   def get_nband(self):
      return self.nband

   def forward(self, input=None):
      """
      input: (B, F, T, 2)
      log_input: (B, F, T)
      return: (B, nband, C, T)
      """
      input = input.permute(0, 3, 1, 2).contiguous()
      x1, x2, x3 = input[..., :144, :], input[..., 144:336, :], input[..., 336:-1, :]
      y1, y2, y3 = self.reg1_encoder(x1), self.reg2_encoder(x2), self.reg3_encoder(x3)

      out = torch.cat([y1, y2, y3], dim=-2).transpose(1, 2).contiguous()  # (B, nband, C, T)

      return out


class BandMerge_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int, 
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64, 
               ):
      super(BandMerge_22k, self).__init__()
      self.sr = sr       
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      fft_reso = sr / n_fft
      bw_250 = int(np.floor(250 / fft_reso))  # 5 bands
      bw_500 = int(np.floor(500 / fft_reso))  # 10 bands
      bw_1k = int(np.floor(1000 / fft_reso))  # 20 bands

      # total 24 bands
      self.band_width = [bw_250] * 12  # 3k
      self.band_width += [bw_500] * 8  # 4k 
      self.band_width += [bw_1k] * 3  # 3k
      self.band_width.append(self.n_fft // 2 + 1 - np.sum(self.band_width))  # remains

      self.nband = len(self.band_width)
      print(f'Totally Merge {len(self.band_width)} bands for sampling rate: 22.05k.')
      self.decoder_mag, self.decoder_phase = nn.ModuleList([]), nn.ModuleList([])
      for i in range(self.nband):
         self.decoder_mag.append(
            nn.Sequential(
               ChannelNormalization(self.feature_dim),
               nn.Conv1d(self.feature_dim, 2 * self.feature_dim, 1),
               nn.GELU(),
               nn.Conv1d(2 * self.feature_dim, int(self.band_width[i]), 1),
               )
            )
         self.decoder_phase.append(
            nn.Sequential(
               ChannelNormalization(self.feature_dim),
               nn.Conv1d(self.feature_dim, 2 * self.feature_dim, 1),
               nn.GELU(),
               nn.Conv1d(2 * self.feature_dim, int(self.band_width[i]) * 2, 1)
               )
            )

   def forward(self, emb_input):
      """
      emb_input: (B, nband, C, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      decode_mag_list, decode_phase_list = [], []
      for i in range(len(self.band_width)):
         # mag
         this_mag = torch.exp(self.decoder_mag[i](emb_input[:, i].contiguous()))
         # phase
         this_comp = self.decoder_phase[i](emb_input[:, i].contiguous())
         this_real, this_imag = this_comp.chunk(2, dim=1)
         this_phase = torch.atan2(this_imag, this_real)

         decode_mag_list.append(this_mag)
         decode_phase_list.append(this_phase)
      mag, phase = torch.cat(decode_mag_list, dim=1), torch.cat(decode_phase_list, dim=1)  # (B, F, T)
      return mag, phase
      


class BandMerge_24k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int, 
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64, 
               ):
      super(BandMerge_24k, self).__init__()
      self.sr = sr       
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.eps = torch.finfo(torch.float32).eps

      fft_reso = sr / n_fft
      bw_250 = int(np.floor(250 / fft_reso))  # 5 bands
      bw_500 = int(np.floor(500 / fft_reso))  # 10 bands
      bw_1k = int(np.floor(1000 / fft_reso))  # 20 bands

      # total 24 bands
      self.band_width = [bw_250] * 12  # 3k
      self.band_width += [bw_500] * 8  # 4k 
      self.band_width += [bw_1k] * 4  # 3k
      self.band_width.append(self.n_fft // 2 + 1 - np.sum(self.band_width))  # remains

      self.nband = len(self.band_width)
      print(f'Totally Merge {len(self.band_width)} bands for sampling rate: 24k.')
      self.decoder_mag, self.decoder_phase = nn.ModuleList([]), nn.ModuleList([])
      for i in range(self.nband):
         self.decoder_mag.append(
            nn.Sequential(
               ChannelNormalization(self.feature_dim),
               nn.Conv1d(self.feature_dim, 2 * self.feature_dim, 1),
               nn.GELU(),
               nn.Conv1d(2 * self.feature_dim, int(self.band_width[i]), 1),
               )
            )
         self.decoder_phase.append(
            nn.Sequential(
               ChannelNormalization(self.feature_dim),
               nn.Conv1d(self.feature_dim, 2 * self.feature_dim, 1),
               nn.GELU(),
               nn.Conv1d(2 * self.feature_dim, int(self.band_width[i]) * 2, 1)
               )
            )


   def forward(self, emb_input):
      """
      emb_input: (B, nband, C, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      decode_mag_list, decode_phase_list = [], []
      for i in range(len(self.band_width)):
         # mag
         this_mag = torch.exp(self.decoder_mag[i](emb_input[:, i].contiguous()))
         # phase
         this_comp = self.decoder_phase[i](emb_input[:, i].contiguous())
         this_real, this_imag = this_comp.chunk(2, dim=1)
         this_phase = torch.atan2(this_imag, this_real)

         decode_mag_list.append(this_mag)
         decode_phase_list.append(this_phase)
      mag, phase = torch.cat(decode_mag_list, dim=1), torch.cat(decode_phase_list, dim=1)  # (B, F, T)
      return mag, phase


class SharedBandMerge_22k(nn.Module):
   def __init__(self,
               sr: int,
               win_size: int, 
               hop_size: int,
               n_fft: int,
               feature_dim: int = 64,
               decode_type: str = 'mag+phase',  
               ):
      super(SharedBandMerge_22k, self).__init__()
      self.sr = sr       
      self.n_fft = n_fft
      self.win_size = win_size
      self.hop_size = hop_size
      self.feature_dim = feature_dim
      self.decode_type = decode_type
      self.eps = torch.finfo(torch.float32).eps
      if self.decode_type.lower() == "mag+phase":
         self.reg1_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_mag_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=1, kernel_size=(44, 1), stride=(44, 1))
         )
         self.reg1_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=12, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(12, 1), stride=(12, 1))
         )
         self.reg2_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=8, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(24, 1), stride=(24, 1))
         )
         self.reg3_phase_decoder = nn.Sequential(
            BandwiseC2LayerNorm(nband=4, feature_dim=self.feature_dim),
            nn.Conv2d(in_channels=self.feature_dim, out_channels=self.feature_dim*2, kernel_size=(1, 1)),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels=self.feature_dim*2, out_channels=2, kernel_size=(44, 1), stride=(44, 1))
         )

   def forward(self, emb_input):
      """
      emb_input: (B, nband, C, T)
      return:
         mag: (B, F, T)
         phase: (B, F, T)
      """
      if self.decode_type.lower() == 'mag+phase':
         x1, x2, x3 = emb_input[:, :12].contiguous().transpose(1, 2).contiguous(), \
                      emb_input[:, 12:20].contiguous().transpose(1, 2).contiguous(), \
                      emb_input[:, 20:].contiguous().transpose(1, 2).contiguous()
         mag1, mag2, mag3 = self.reg1_mag_decoder(x1), self.reg2_mag_decoder(x2), self.reg3_mag_decoder(x3)
         com1, com2, com3 = self.reg1_phase_decoder(x1), self.reg2_phase_decoder(x2), self.reg3_phase_decoder(x3)
         mag = torch.exp(torch.cat([mag1, mag2, mag3], dim=-2))  # exp operation
         com = torch.cat([com1, com2, com3], dim=-2)
         last_mag, last_com = mag[..., -1, :].unsqueeze(-2), com[..., -1, :].unsqueeze(-2)
         mag, com = torch.cat([mag, last_mag], dim=-2), torch.cat([com, last_com], dim=-2)
         pha = torch.atan2(com[:, -1], com[:, 0])
         return mag.squeeze(1), pha


class BandShuffler(nn.Module):
   """
   The structure is from https://github.com/Audio-WestlakeU/NBSS/blob/main/models/arch/SpatialNet.py
   """
   def __init__(self, 
               nband: int,
               input_size: int,
               squeeze_size: int=64,
               f_kernel_size: int=3,
               f_conv_groups: int=8,
               ):
      super(BandShuffler, self).__init__()
      self.nband = nband
      self.input_size = input_size
      self.squeeze_size = squeeze_size
      self.f_kernel_size = f_kernel_size
      self.f_conv_groups = f_conv_groups
      #
      self.fconv1 = nn.Sequential(
         ChannelNormalization(input_size),
         nn.Conv1d(input_size, input_size, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode='zeros'),
         nn.PReLU(input_size)
      )
      self.fconv2 = nn.Sequential(
         ChannelNormalization(input_size),
         nn.Conv1d(input_size, input_size, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode='zeros'),
         nn.PReLU(input_size)
      )
      self.squeeze = nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=squeeze_size, kernel_size=1), nn.SiLU())
      self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=squeeze_size, out_channels=input_size, kernel_size=1), nn.SiLU())
      self.full = LinearGroup(nband, nband, squeeze_size)

   def forward(self, input):
      """
      input: (B, T, C, nband)
      return: (B, T, C, nband)
      """
      b_size, seq_len, c, nband = input.shape
      x = input.view(b_size * seq_len, c, nband)  # (B*T, C, nband)
      # f-conv1
      resi = x
      x = self.fconv1(x)
      x = resi + x
      # group
      resi = x
      x = self.squeeze(x)
      x = self.full(x)
      x = self.unsqueeze(x)
      x = resi + x
      # f-conv2
      resi = x
      x = self.fconv2(x)
      x = resi + x
      # reshape
      out = x.view(*input.shape)
      return out


class TimeResRNN(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 dropout: float = 0.,
                 causal: bool = True,
                 residual: bool = True,
                 ):
        super(TimeResRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.causal = causal
        self.residual = residual
        self.eps = torch.finfo(torch.float32).eps
        if not causal:
            self.norm = TimeGlobalNormalization(input_size, ndim=4)
        else:
            self.norm = nn.LayerNorm(input_size)

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=not causal)

        # linear projection layer
        self.proj = nn.Linear(hidden_size*(int(not causal) + 1), input_size)

    def forward(self, input):
        """
        input: (B, nband, C, T)
        return: (B, nband, C, T)
        """
        batch_size, t1, E, t2 = input.shape
        x = input.transpose(-2, -1).contiguous()
        x = self.norm(x)
        x = x.view(batch_size * t1, t2, E)
        rnn_output, _ = self.rnn(self.dropout(x))
        rnn_output = self.proj(rnn_output).transpose(-2, -1).contiguous().view(*input.shape)
        if self.residual:
            return input + rnn_output
        else:
            return rnn_output


class FreqResRNN(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 dropout: float = 0.,
                 causal: bool = True,
                 residual: bool = True,
                 ):
        super(FreqResRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.causal = causal
        self.residual = residual
        self.eps = torch.finfo(torch.float32).eps
        self.norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(input_size, hidden_size, 1, batch_first=True, bidirectional=not causal)

        # linear projection layer
        self.proj = nn.Linear(hidden_size * (int(not causal) + 1), input_size)

    def forward(self, input):
        """
        input: (B, T, C, nband)
        return: (B, T, C, nband)
        """
        batch_size, t1, E, t2 = input.shape
        x= input.transpose(-2, -1).contiguous()
        x = self.norm(x)
        x = x.view(batch_size * t1, t2, E)
        rnn_output, _ = self.rnn(self.dropout(x))
        rnn_output = self.proj(rnn_output).transpose(-2, -1).contiguous().view(*input.shape)
        if self.residual:
            return input + rnn_output
        else:
            return rnn_output


class BandWiseTimeModule(nn.Module):
   def __init__(self, 
                nband: int,
                nrep: int,
                input_channel: int,
                hidden_channel: int,
                kernel_size: int,
                causal: bool = False,
                ):
      super(BandWiseTimeModule, self).__init__()
      self.nband = nband
      self.nrep = nrep
      self.input_channel = input_channel
      self.hidden_channel = hidden_channel
      self.kernel_size = kernel_size
      self.causal = causal

      band_timenet_list = []
      for _ in range(self.nrep):
         band_timenet_list.append(
            nn.Sequential(
               nn.Conv1d(input_channel, input_channel, kernel_size, padding="same", padding_mode="zeros", groups=input_channel),
               BandwiseLayerNorm(self.nband, input_channel),
               nn.Conv1d(input_channel, hidden_channel, 1),
               nn.GELU(),
               GRN(hidden_channel),
               nn.Conv1d(hidden_channel, input_channel, 1)
            )
         )
      self.Ttband_timenet_list = nn.ModuleList(band_timenet_list)

   def forward(self, input):
      """
      inpt: (B, nband, C, T)
      return: (B, nband, C, T)
      """
      #
      b_size, nband, nch, seq_len = input.shape
      x = input.view(b_size*nband, nch, -1)
      for timenet in self.Ttband_timenet_list:
         bot = x.clone()
         x = timenet(x)
         x = x + bot
      out = x.view(b_size, nband, nch, seq_len)

      return out
