import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm import ChannelNormalization


class FeedForward(nn.Module):
   def __init__(self, 
                nband: int,
                input_channel: int,
                kernel_size: int=1,
                causal: bool = False,
                ):
      super(FeedForward, self).__init__()
      self.nband = nband
      self.input_channel = input_channel
      self.kernel_size = kernel_size
      self.causal = causal

      self.conv1 = nn.Sequential(
         ChannelNormalization(input_channel, ndim=4),
         nn.Conv2d(input_channel, input_channel * 2, kernel_size=1),
      )
      self.conv2 = nn.Conv2d(input_channel * 2, input_channel * 2, kernel_size=(3, 5), padding="same", padding_mode="zeros", groups=input_channel * 2)
      self.conv3 = nn.Conv2d(input_channel, input_channel, kernel_size=1)

   def forward(self, input):
      """
      inpt: (B, C, nband, T)
      return: (B, C, nband, T)
      """
      #
      x_res = input
      x = self.conv1(input)
      x1, x2 = self.conv2(x).chunk(2, dim=1)
      x = F.gelu(x1) * x2
      x = self.conv3(x)
      x = x_res + x
      return x


class ShufflerAttention(nn.Module):
   def __init__(self,
                dim,
                num_heads,
                f_win_size, 
                t_win_size,
                qkv_bias: bool = True,
                attn_drop=0.,
                proj_drop=0.,
                use_shuffle: bool = True,
                ):
      super(ShufflerAttention, self).__init__()
      self.dim = dim
      self.num_heads = num_heads
      self.f_win_size = f_win_size
      self.t_win_size = t_win_size
      self.qkv_bias = qkv_bias
      self.attn_drop = attn_drop
      self.proj_drop = proj_drop
      self.use_shuffle = use_shuffle
      #
      self.norm = nn.LayerNorm(dim)
      self.attn = WindowAttention(dim=dim, 
                                  f_win_size=f_win_size, 
                                  t_win_size=t_win_size,
                                  num_heads=num_heads,
                                  qkv_bias=qkv_bias,
                                  attn_drop=attn_drop,
                                  proj_drop=proj_drop,
                                  )
   def forward(self, inpt):
      """
      (B, N, T, C)
      return: (B, N, T, C)
      """
      b_size, nband, seq_len, C = inpt.shape
      # pad if not mod
      if nband % self.f_win_size != 0:
         f_mod = self.f_win_size - (nband % self.f_win_size)
      else:
         f_mod = 0
      if seq_len % self.t_win_size != 0:
         t_mod = self.t_win_size - (seq_len % self.t_win_size)
      else:
         t_mod = 0
      if f_mod != 0:
         pad_ = inpt[:, -f_mod:]
         inpt = torch.cat([inpt, pad_], dim=1)
      if t_mod != 0:
         pad_ = inpt[:, :, -t_mod:]
         inpt = torch.cat([inpt, pad_], dim=2)
      _, nband_, seq_len_, _ = inpt.shape
      x_resi = inpt
      x = self.norm(inpt)
      # shufle (optional)
      if self.use_shuffle:
         N_Index = list(range(0, nband_))
         T_Index = list(range(0, seq_len_))
         N_Shuffle = list(range(0, nband_))
         T_Shuffle = list(range(0, seq_len_))
         N_Shuffle_list, T_Shuffle_list = [], []
         shuffle_x_ = x.clone()
         for b in range(b_size):
            random.shuffle(N_Shuffle)
            random.shuffle(T_Shuffle)
            N_Shuffle_list.append(N_Shuffle)
            T_Shuffle_list.append(T_Shuffle)
            N_Shuffle = list(range(0, nband_))
            T_Shuffle = list(range(0, seq_len_))
         # 
         for b, (N_Shuffle, T_Shuffle) in enumerate(zip(N_Shuffle_list, T_Shuffle_list)):
            shuffle_x_[b, :, :, :] = x[b, N_Shuffle, :, :]
            shuffle_x_[b, :, :, :] = shuffle_x_[b, :, T_Shuffle, :]
      else:
         shuffle_x_ = x
      # partition windows
      x_windows = window_partition(shuffle_x_,
                                   f_win_size=self.f_win_size,
                                   t_win_size=self.t_win_size,
                                  )  # (B', f_win_size, t_win_size, C)
      x_windows = x_windows.view(-1, self.f_win_size*self.t_win_size, C)
      # W-MSA
      attn_windows = self.attn(x_windows)
      # merge windows
      attn_windows = attn_windows.view(-1, self.f_win_size, self.t_win_size, C)
      shuffle_x = window_reverse(attn_windows, self.f_win_size, self.t_win_size, nband_, seq_len_)

      # reverse shuffle (optional)
      if self.use_shuffle:
         RT_Shuffle_list, RN_Shuffle_list = [], []
         for b in range(b_size):
            RT_Shuffle_list.append([T_Shuffle_list[b][i] for i in T_Index])
            RN_Shuffle_list.append([N_Shuffle_list[b][i] for i in N_Index])
         for b, (RN_Shuffle, RT_Shuffle) in enumerate(zip(RN_Shuffle_list, RT_Shuffle_list)):
            shuffle_x[b, :, RT_Shuffle, :] = shuffle_x[b, :, T_Index, :]
            shuffle_x[b, RN_Shuffle, :, :] = shuffle_x[b, N_Index, :, :]

      x = (x_resi + shuffle_x)[:, :nband, :seq_len, :]
      return x


class LinearProjection(nn.Module):
   def __init__(self, 
                dim, 
                heads=8, 
                dim_head=64, 
                dropout=0.,
                bias=True,
                ):
      super(LinearProjection, self).__init__()
      self.dim = dim
      self.heads = heads
      self.dim_head = dim_head
      self.dropout = dropout
      self.bias = bias
      inner_dim = dim_head * heads
      self.to_q = nn.Linear(dim, inner_dim, bias=bias)
      self.to_k = nn.Linear(dim, inner_dim, bias=bias)
      self.to_v = nn.Linear(dim, inner_dim, bias=bias)
      self.inner_dim = inner_dim
   
   def forward(self, x):
      """
      x: (B, N, C)
      return: {k, v}->(B, heads, N, C//heads)
      """
      B_, N, C = x.shape
      # (B_, heads, N, C//heads)
      q = self.to_q(x).reshape(B_, N, self.heads, C // self.heads).transpose(1, 2)
      k = self.to_k(x).reshape(B_, N, self.heads, C // self.heads).transpose(1, 2)
      v = self.to_v(x).reshape(B_, N, self.heads, C // self.heads).transpose(1, 2)
      return q, k, v


class WindowAttention(nn.Module):
   def __init__(self,
                dim: int,
                f_win_size: int,
                t_win_size: int,
                num_heads: int,
                qkv_bias: bool = True,
                attn_drop=0.,
                proj_drop=0.,
                ):
      super().__init__()
      self.dim = dim
      self.f_win_size = f_win_size
      self.t_win_size = t_win_size
      self.num_heads = num_heads
      self.qkv_bias = qkv_bias
      head_dim = dim // num_heads
      self.scale = head_dim ** -0.5
      
      self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
      self.attn_drop = nn.Dropout(attn_drop)
      self.proj = nn.Linear(dim, dim)
      self.proj_drop = nn.Dropout(proj_drop)
      self.softmax = nn.Softmax(dim=-1)
   
   def forward(self, x):
      """
      x: (B, N, C)
      return: (B, N, C)
      """
      B_, N, C = x.shape
      q, k, v = self.qkv(x)  # (B, nheads, N, C//heads)
      q = q * self.scale
      attn = (q @ k.transpose(-2, -1))
      
      attn = self.attn_drop(self.softmax(attn))
      x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
      x = self.proj_drop(self.proj(x))
      
      return x


def window_partition(x, f_win_size, t_win_size):
   """
   x: (B, N, T, C)
   return: (B', W1, W2, C), where B'=B*(N//W1)*(T//W2)
   """
   B, N, T, C = x.shape
   x = x.view(B, N // f_win_size, f_win_size, T // t_win_size, t_win_size, C)
   windows = x.transpose(2, 3).contiguous().view(-1, f_win_size, t_win_size, C)
   return windows


def window_reverse(windows, f_win_size, t_win_size, N, T):
   """
   windows: (B', W1, W2, C), where B'=B*(N//W1)*(T//W2)
   return: (B, N, T, C)
   """
   B = int(windows.shape[0] / (N * T / f_win_size / t_win_size))
   x = windows.view(B, N//f_win_size, T//t_win_size, f_win_size, t_win_size, -1)
   x = x.transpose(2, 3).contiguous().view(B, N, T, -1)
   return x
     