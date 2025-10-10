import torch 
import torch.nn as nn
import random

from .rnd_utils.basic_arch import *
from .rnd_utils.norm import *
from librosa.filters import mel as librosa_mel_fn


class VocModule(nn.Module):
   def __init__(self,
                nrep: int,
                nband: int,
                input_channel: int,
                squeeze_size: int,
                hidden_channel: int,
                kernel_size: int,
                causal: bool = False,
                time_type: bool = 'convnext_v2',
                freq_type: bool = 'shuffler',
                ):
      super(VocModule, self).__init__()
      self.nrep = nrep 
      self.nband = nband
      self.input_channel = input_channel
      self.sqeeze_size = squeeze_size
      self.hidden_channel = hidden_channel
      self.kernel_size = kernel_size
      self.causal = causal
      self.time_type = time_type
      self.freq_type = freq_type
      # Frequency modeling
      if self.freq_type.lower() == "shuffler":
         self.bandnet = BandShuffler(nband, input_channel, squeeze_size)
      elif self.freq_type.lower() == "lstm":
         self.bandnet = FreqResRNN(input_channel, hidden_channel, causal=False)
      elif self.freq_type.lower() == "none":
         self.bandnet = nn.Identity()

      # Time modeling
      if self.time_type.lower() == "convnext_v2":
         self.timenet = BandWiseTimeModule(nband, nrep, input_channel, hidden_channel, kernel_size, causal)
      elif self.time_type.lower() == "lstm":
         self.timenet = TimeResRNN(input_channel, hidden_channel, causal=causal)
      elif self.time_type.lower() == "identity":
         self.timenet = nn.Identity()

   def forward(self, inpt):
      """
      inpt: (B, T, C, nband)
      return: (B, T, C, nband)
      """
      # band modeling
      x = self.bandnet(inpt)
      # time modeling
      x = x.transpose(1, 3).contiguous()
      x = self.timenet(x)

      return x.transpose(1, 3).contiguous()


class RNDVocoder24k(nn.Module):
   def __init__(self, 
                h,
                ):
      super(RNDVocoder24k, self).__init__()
      self.h = h
      self.sampling_rate = h.sampling_rate
      self.num_mels = h.num_mels
      self.win_size = h.win_size 
      self.hop_size = h.hop_size
      self.n_fft = h.n_fft
      self.fmin = h.fmin
      self.fmax = h.fmax
      self.nrep = h.nrep
      self.squeeze_size = h.squeeze_size
      self.null_nstage = h.null_nstage
      self.input_channel = h.input_channel
      self.hidden_channel = h.hidden_channel
      self.kernel_size = h.kernel_size
      self.causal = h.causal
      self.use_rnd = h.use_rnd
      self.time_type = h.time_type
      self.freq_type = h.freq_type
      self.use_shared_encoder = h.use_shared_encoder
      self.use_shared_decoder = h.use_shared_decoder
      self.phiphiT_learnable = h.phiphiT_learnable
      self.eps = torch.finfo(torch.float32).eps

      # mel matrix
      mel = librosa_mel_fn(sr=self.sampling_rate,
                           n_fft=self.n_fft,
                           n_mels=self.num_mels,
                           fmin=self.fmin,
                           fmax=self.fmax,
                           )
      mel_basis = torch.from_numpy(mel)
      inv_mel_basis = mel_basis.pinverse()
      # Phi: (Fmel, F), PhiT: (F, Fmel)
      if not self.phiphiT_learnable:
         self.register_buffer('Phi', mel_basis)
         self.register_buffer('PhiT', inv_mel_basis)
         self.register_buffer('PhiTPhi', inv_mel_basis@mel_basis)
      else:  # learnable
         self.Phi = nn.Parameter(mel_basis)
         self.PhiT = nn.Parameter(inv_mel_basis)
         self.PhiTPhi = nn.Parameter(inv_mel_basis@mel_basis)

      # null module
      if self.use_shared_encoder:
         # shared_22k also works herein
         self.null_enc = SharedBandSplit_22k(sr=self.sampling_rate,
                                             win_size=self.win_size,
                                             hop_size=self.hop_size,
                                             n_fft=self.n_fft,
                                             feature_dim=self.input_channel,
                                             )
      else:
         self.null_enc = BandSplit_24k(sr=self.sampling_rate, 
                                       win_size=self.win_size,
                                       hop_size=self.hop_size,
                                       n_fft=self.n_fft,
                                       feature_dim=self.input_channel,
                                       )
      self.null_nband = self.null_enc.get_nband()
      if self.use_shared_decoder:
         self.null_dec = SharedBandMerge_22k(sr=self.sampling_rate,
                                             win_size=self.win_size,
                                             hop_size=self.hop_size,
                                             n_fft=self.n_fft,
                                             feature_dim=self.input_channel,
                                             )
      else:
         self.null_dec = BandMerge_24k(sr=self.sampling_rate,
                                       win_size=self.win_size,
                                       hop_size=self.hop_size,
                                       n_fft=self.n_fft,
                                       feature_dim=self.input_channel,
                                       )
      null_module_list = []
      for _ in range(self.null_nstage):
         null_module_list.append(
            VocModule(nrep=self.nrep,
                      nband=self.null_nband,
                      input_channel=self.input_channel,
                      squeeze_size=self.squeeze_size,
                      hidden_channel=self.hidden_channel,
                      kernel_size=self.kernel_size,
                      causal=self.causal,
                      time_type=self.time_type,
                      freq_type=self.freq_type,
                      )
         )
      
      self.null_module_list = nn.ModuleList(null_module_list)
      self.alpha = nn.Parameter(torch.ones([1, 1, self.input_channel, self.null_nband]))
      self.rng = random.Random(1234)
      #
      self.apply(self._init_weights)

   def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

   def forward(self, mel):
      """
      mel: (B, Fmel, T)
      mel_basis: (Fmel, F)
      inv_mel_basis: (F, Fmel)
      """
      # initialize
      init_mag = (self.PhiT @ torch.exp(mel)).abs().clamp_min_(1e-5)  # (B, F, T)
      init_spec = torch.stack([init_mag, torch.zeros_like(init_mag)], dim=-1)  # (B, F, T, 2)

      # ----------- null domain ------------------
      null_x = self.null_enc(init_spec).transpose(1, 3).contiguous()  # (B, T, C, nband)
      x = null_x
     
      for i in range(self.null_nstage): 
         x = self.null_module_list[i](x)

      if self.use_rnd:  # whether to use RND mechanism
         null_x_ = (self.alpha * null_x + x).transpose(1, 3).contiguous()
         null_mag, null_pha = self.null_dec(null_x_)

         # ----------- RND ----------------------------
         null_filter = torch.eye(self.n_fft//2+1, dtype=null_mag.dtype, device=null_mag.device) - self.PhiTPhi
         null_filtered_mag = torch.einsum('fk,bkt->bft', [null_filter, null_mag]).abs().clamp_min_(1e-5)

         out_mag = null_filtered_mag + init_mag  # only in the mag-domain
         out_spec = torch.stack([out_mag * torch.cos(null_pha), out_mag * torch.sin(null_pha)], dim=-1)
      else:
         null_x_ = x.transpose(1, 3).contiguous()
         null_mag, null_pha = self.null_dec(null_x_)
         out_spec = torch.stack([null_mag * torch.cos(null_pha), null_mag * torch.sin(null_pha)], dim=-1)

      logamp = torch.log(torch.norm(out_spec, dim=-1) + 1e-7)
      pha = torch.atan2(out_spec[..., -1], out_spec[..., 0])
      rea, imag = out_spec[..., 0], out_spec[..., -1]

      out_spec = torch.complex(rea, imag)
      out_wav = torch.istft(out_spec,
                            n_fft=self.n_fft,
                            hop_length=self.hop_size,
                            win_length=self.win_size,
                            window=torch.hann_window(self.win_size).to(mel.device))
      
      return logamp, pha, rea, imag, out_wav


# if __name__ == "__main__":
#    net = RNDVocoder(sampling_rate=22050,
#                    num_mels=80,
#                    win_size=1024,
#                    hop_size=256,
#                    n_fft=1024,
#                    fmin=0,
#                    fmax=8000,
#                    range_nstage=2,
#                    null_nstage=6,
#                    ngroup=8,
#                    input_channel=256,
#                    hidden_channel=512,
#                    kernel_size=7,
#                    domain_type="stft",
#                    is_even=False,
#                    causal=False,
#                    ).cuda()
#    x = torch.rand([3, 80, 64]).cuda()
#    y = net(x)[-1]
#    print(f'{x.shape}->{y.shape}')
#    from ptflops import get_model_complexity_info
#    get_model_complexity_info(net, (80, 64))
#    # test speed
#    import time
#    from tqdm import tqdm
#    statis_list = []
#    for i in range(5):
#       len_tot = 0
#       time_tot = 0
#       for i in tqdm(range(30)):
#          x = torch.rand([1, 80, 512]).cuda()
#          start = time.time()
#          y_list = net(x)
#          end = time.time()
#          len_tot += (y_list[-1].shape[-1] / 22050)
#          time_tot += (end - start)
#       statis_list.append(len_tot / time_tot)
#    print(f'Mean Speed Ratio: x{np.mean(np.array(statis_list))}.')
