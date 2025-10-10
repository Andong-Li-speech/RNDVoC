import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from torchaudio.transforms import Resample 
from utils import init_weights, get_padding
from dataset_ljs import inverse_mel, mel_spectrogram
from typing import *
import numpy as np

LRELU_SLOPE = 0.1


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    )
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        if x.ndim == 2:
            x= x.unsqueeze(1)  # (B, 1, L)

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, mpd_reshapes):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(mpd_reshapes[0]),
                DiscriminatorP(mpd_reshapes[1]),
                DiscriminatorP(mpd_reshapes[2]),
                DiscriminatorP(mpd_reshapes[3]),
                DiscriminatorP(mpd_reshapes[-1]),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def phase_loss(phase_r, phase_g, n_fft, frames):
    GD_matrix = (
        torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=1)
        - torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=2)
        - torch.eye(n_fft // 2 + 1)
    )
    GD_matrix = GD_matrix.to(phase_g.device)

    GD_r = torch.matmul(phase_r.permute(0, 2, 1), GD_matrix)
    GD_g = torch.matmul(phase_g.permute(0, 2, 1), GD_matrix)

    PTD_matrix = (
        torch.triu(torch.ones(frames, frames), diagonal=1)
        - torch.triu(torch.ones(frames, frames), diagonal=2)
        - torch.eye(frames)
    )
    PTD_matrix = PTD_matrix.to(phase_g.device)

    PTD_r = torch.matmul(phase_r, PTD_matrix)
    PTD_g = torch.matmul(phase_g, PTD_matrix)

    IP_loss = torch.mean(anti_wrapping_function(phase_r - phase_g))
    GD_loss = torch.mean(anti_wrapping_function(GD_r - GD_g))
    PTD_loss = torch.mean(anti_wrapping_function(PTD_r - PTD_g))
    loss = IP_loss + GD_loss + PTD_loss
    return loss


class OmniPhaseLoss(nn.Module):
    def __init__(self):
        super(OmniPhaseLoss, self).__init__()
        kernel1 = torch.from_numpy(np.array([[-1., 0, 0], [0, 1, 0], [0, 0, 0]], dtype='float32'))
        kernel2 = torch.from_numpy(np.array([[0, -1., 0], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel3 = torch.from_numpy(np.array([[0, 0, -1.], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel4 = torch.from_numpy(np.array([[0, 0, 0], [-1., 1., 0], [0, 0, 0]], dtype='float32'))
        kernel5 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel6 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., -1.], [0, 0, 0]], dtype='float32'))
        kernel7 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [-1., 0, 0]], dtype='float32'))
        kernel8 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, -1., 0]], dtype='float32'))
        kernel9 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, 0, -1.]], dtype='float32'))
        kernels = torch.stack([kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8, kernel9], dim=0)  # (out_nch, 3, 3)
        kernels = kernels.unsqueeze(1)
        self.filters = kernels
    
    def forward(self, phase_r, phase_g):
        """
        phase_r: (B, F, T)
        phase_g: (B, F, T)
        """
        phase_r = phase_r.transpose(-2, -1).unsqueeze(1)  # (B,1,T,F)
        phase_g = phase_g.transpose(-2, -1).unsqueeze(1)  # (B,1,T,F)
        omni_phase_r = F.conv2d(phase_r, self.filters.to(phase_r.device), bias=None, stride=1, padding=1)  # (B,9,T,F)
        omni_phase_g = F.conv2d(phase_g, self.filters.to(phase_g.device), bias=None, stride=1, padding=1)  # (B,9,T,F)
        loss = 3 * torch.mean(anti_wrapping_function(omni_phase_g - omni_phase_r))

        return loss


# Multi-scale mel loss
class MultiResolutionMelLoss(nn.Module):
    def __init__(self,
                 resolutions=((32, 8, 32, 5),
                              (64, 16, 64, 10),
                              (128, 32, 128, 20),
                              (256, 64, 256, 40),
                              (512, 128, 512, 80),
                              (1024, 256, 1024, 160),
                              (2048, 512, 2048, 320),
                              ),
                sampling_rate=24000,
    ):
        super(MultiResolutionMelLoss, self).__init__()
        self.resolutions = resolutions
        self.sampling_rate = sampling_rate
    
    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        loss_tot = 0.
        for i, cur_reso in enumerate(self.resolutions):
            y_mel = mel_spectrogram(y, 
                                    n_fft=cur_reso[0], 
                                    num_mels=cur_reso[-1],
                                    sampling_rate=self.sampling_rate,
                                    hop_size=cur_reso[1],
                                    win_size=cur_reso[2],
                                    fmin=0,
                                    fmax=self.sampling_rate / 2,
                                    )
            y_hat_mel = mel_spectrogram(y_hat, 
                                        n_fft=cur_reso[0], 
                                        num_mels=cur_reso[-1],
                                        sampling_rate=self.sampling_rate,
                                        hop_size=cur_reso[1],
                                        win_size=cur_reso[2],
                                        fmin=0,
                                        fmax=self.sampling_rate / 2,
                                        )
            loss_tot = loss_tot + F.l1_loss(y_mel, y_hat_mel)
        loss_tot = loss_tot / len(self.resolutions)
        return loss_tot


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



class MultiResolutionDiscriminator(nn.Module):
    def __init__(
        self,
        resolutions=((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512)),
        num_embeddings: int = None,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(resolution=r, num_embeddings=num_embeddings)
                for r in resolutions
            ]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor, bandwidth_id: torch.Tensor = None
    ):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y, cond_embedding_id=bandwidth_id)
            y_d_g, fmap_g = d(x=y_hat, cond_embedding_id=bandwidth_id)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(
        self,
        resolution,
        channels: int = 64,
        in_channels: int = 1,
        num_embeddings: int = None,
        lrelu_slope: float = 0.1,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv2d(
                        in_channels,
                        channels,
                        kernel_size=(7, 5),
                        stride=(2, 2),
                        padding=(3, 2),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(5, 3),
                        stride=(2, 1),
                        padding=(2, 1),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=(5, 3),
                        stride=(2, 2),
                        padding=(2, 1),
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=3, stride=(2, 1), padding=1
                    )
                ),
                weight_norm(
                    nn.Conv2d(
                        channels, channels, kernel_size=3, stride=(2, 2), padding=1
                    )
                ),
            ]
        )
        if num_embeddings is not None:
            self.emb = torch.nn.Embedding(
                num_embeddings=num_embeddings, embedding_dim=channels
            )
            torch.nn.init.zeros_(self.emb.weight)
        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor = None):
        fmap = []
        if x.ndim == 3:
            x = x.squeeze(1)

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = torch.nn.functional.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        if cond_embedding_id is not None:
            emb = self.emb(cond_embedding_id)
            h = (emb.view(1, -1, 1, 1) * x).sum(dim=1, keepdims=True)
        else:
            h = 0
        x = self.conv_post(x)
        fmap.append(x)
        x += h
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        n_fft, hop_length, win_length = self.resolution
        magnitude_spectrogram = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(x.device),  # interestingly rectangular window kind of works here
            center=True,
            return_complex=True,
        ).abs()

        return magnitude_spectrogram


# Adapted from https://github.com/open-mmlab/Amphion/blob/main/models/vocoders/gan/discriminator/mssbcqtd.py under the MIT license.
#   LICENSE is in incl_licenses directory.
class DiscriminatorCQT(nn.Module):
    def __init__(self, h, hop_length: int, n_octaves:int, bins_per_octave: int):
        super().__init__()
        self.h = h

        self.filters = h.cqtd_filters
        self.max_filters = h.cqtd_max_filters
        self.filters_scale = h.cqtd_filters_scale
        self.kernel_size = (3, 9)
        self.dilations = h.cqtd_dilations
        self.stride = (1, 2)

        self.in_channels = h.cqtd_in_channels
        self.out_channels = h.cqtd_out_channels
        self.fs = h.sampling_rate
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        # Lazy-load
        from nnAudio import features

        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for _ in range(self.n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=self.get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Conv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=(dilation, 1),
                        padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
                    )
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            weight_norm(
                nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                    padding=self.get_2d_padding(
                        (self.kernel_size[0], self.kernel_size[0])
                    ),
                )
            )
        )

        self.conv_post = weight_norm(
            nn.Conv2d(
                out_chs,
                self.out_channels,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            )
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.fs, new_freq=self.fs * 2)

        self.cqtd_normalize_volume = False
        if self.cqtd_normalize_volume:
            print(
                f"[INFO] cqtd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in CQTD!"
            )

    def get_2d_padding(
        self,
        kernel_size: Tuple[int, int],
        dilation: Tuple[int, int] = (1, 1),
    ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x: torch.tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []

        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x = self.resample(x)

        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2))  # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)

            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return latent_z, fmap


class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, h):
        super().__init__()

        self.h = h
        # Using get with defaults
        self.h.cqtd_filters = h.cqtd_filters
        self.h.cqtd_max_filters = h.cqtd_max_filters
        self.h.cqtd_filters_scale = h.cqtd_filters_scale
        self.h.cqtd_dilations = h.cqtd_dilations
        self.h.cqtd_in_channels = 1
        self.h.cqtd_out_channels = 1
        # Multi-scale params to loop over
        self.h.cqtd_hop_lengths = self.h.cqtd_hop_lengths
        self.h.cqtd_n_octaves = self.h.cqtd_n_octaves
        self.h.cqtd_bins_per_octaves = self.h.cqtd_bins_per_octaves

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    self.h,
                    hop_length=self.h.cqtd_hop_lengths[i],
                    n_octaves=self.h.cqtd_n_octaves[i],
                    bins_per_octave=self.h.cqtd_bins_per_octaves[i],
                )
                for i in range(len(self.h.cqtd_hop_lengths))
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * math.pi)) * 2 * math.pi)


def amplitude_loss(log_amplitude_r, log_amplitude_g):
    MSELoss = torch.nn.MSELoss()

    amplitude_loss = MSELoss(log_amplitude_r, log_amplitude_g)

    return amplitude_loss


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean(torch.clamp(1 - dr, min=0))
        g_loss = torch.mean(torch.clamp(1 + dg, min=0))
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean(torch.clamp(1 - dg, min=0))
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def ls_discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def ls_generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def STFT_consistency_loss(rea_r, rea_g, imag_r, imag_g):
    C_loss = torch.mean(
        torch.mean((rea_r - rea_g) ** 2 + (imag_r - imag_g) ** 2, (1, 2))
    )

    return C_loss
