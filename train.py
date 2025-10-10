import warnings
import numpy as np
import math
import librosa as lib
import torch
from pesq import pesq
warnings.simplefilter(action="ignore", category=FutureWarning)

import itertools
import os
import argparse
import time
import json
import datetime
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from ptflops import get_model_complexity_info
# from dataset_ljs import Dataset, mel_spectrogram, amp_pha_specturm, get_dataset_filelist, spectrogram
from dataset_libritts import Dataset, mel_spectrogram, amp_pha_specturm, get_dataset_filelist, spectrogram

from Models.models import (
    MultiPeriodDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
    amplitude_loss,
    OmniPhaseLoss,
    phase_loss,
    STFT_consistency_loss,
    MultiResolutionMelLoss,
    MultiResolutionDiscriminator,
    MultiScaleSubbandCQTDiscriminator
)
# import models
from Models.rndvoc_22k import RNDVocoder22k
from Models.rndvoc_24k import RNDVocoder24k

from utils import (
    AttrDict,
    build_env,
    plot_spectrogram,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
    remove_older_checkpoint
)

torch.backends.cudnn.benchmark = True

def train(rank, a, h):
    device_flag = (rank == 0)
    device = torch.device(f"cuda:{rank:d}")
    if a.num_gpus > 1:
        if device_flag:
            print("Using DDP for training.")
        dist.init_process_group(backend=h.dist_backend,
                                init_method=h.dist_url,
                                timeout=datetime.timedelta(seconds=3400),
                                world_size=a.num_gpus,
                                rank=rank,
                                )
    
    h.batch_size_per_gpu = int(h.batch_size // a.num_gpus)
    torch.cuda.manual_seed(h.seed)
    torch.cuda.set_device(rank)
    if device_flag:
        print(f"Let us use {a.num_gpus} GPUs in total, and for each gpu, the batch size is {h.batch_size_per_gpu}.")

    generator = eval(h.model_name)(h).to(device)
    if device_flag:
        get_model_complexity_info(generator, (h.num_mels, h.sampling_rate//h.hop_size+1))
    mpd = MultiPeriodDiscriminator(h.mpd_reshapes).to(device)
    if device_flag:
        print("Using MultiPeriodDiscriminator")
    if h.use_cqtd_instead_of_mrd:
        mrd = MultiScaleSubbandCQTDiscriminator(h).to(device)
        if device_flag:
            print("Using MultiScaleSubbandCQTDiscriminator to replace MultiResolutionDiscriminator.")
    else:
        mrd = MultiResolutionDiscriminator(resolutions=h.mrd_resolutions).to(device)
        if device_flag:
            print("Using MultiResolutionDiscriminator.")
    
    if h.use_multiscale_mel:
        ms_mel_loss = MultiResolutionMelLoss(resolutions=h.mel_resolutions,
                                             sampling_rate=h.sampling_rate,
                                             ).to(device)
    if h.use_omni_phase_loss:
        omni_phase_loss = OmniPhaseLoss().to(device)

    # print(generator)
    os.makedirs(h.checkpoint_path, exist_ok=True)
    if device_flag:
        print("checkpoints directory: ", h.checkpoint_path)

    if os.path.isdir(h.checkpoint_path):
        cp_g = scan_checkpoint(h.checkpoint_path, "g_")
        cp_do = scan_checkpoint(h.checkpoint_path, "do_")

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        mrd.load_state_dict(state_dict_do["mrd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]
    
    if a.num_gpus > 1:
        generator = DDP(generator, device_ids=[rank], find_unused_parameters=True)

        mpd = DDP(mpd, device_ids=[rank], find_unused_parameters=True)
        mrd = DDP(mrd, device_ids=[rank], find_unused_parameters=True)

    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(mrd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    training_filelist, validation_filelist = get_dataset_filelist(
        h.input_training_wav_list, h.input_validation_wav_list, h.raw_wavfile_path
    )

    trainset = Dataset(
        training_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        h.meloss,
        n_cache_reuse=0,
        shuffle=True,
        device=device,
    )
    train_sampler = DistributedSampler(trainset) if a.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False if a.num_gpus > 1 else True,
        sampler=train_sampler,
        batch_size=h.batch_size_per_gpu,
        pin_memory=True,
        drop_last=True,
    )

    validset = Dataset(
        validation_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        h.meloss,
        False,
        False,
        n_cache_reuse=0,
        device=device,
    )
    validation_loader = DataLoader(
        validset,
        num_workers=1,
        shuffle=False,
        sampler=None,
        batch_size=1,
        pin_memory=True,
        drop_last=True,
    )

    sw = SummaryWriter(os.path.join(h.checkpoint_path, "logs"))

    generator.train()
    mpd.train()
    mrd.train()
    global_pesq_score = -math.inf
    for epoch in range(max(0, last_epoch), h.training_epochs):
        start = time.time()
        if device_flag:
            print("Epoch: {}".format(epoch + 1))
        
        if a.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(tqdm(train_loader)):
            start_b = time.time()
            x, logamp, pha, rea, imag, y, meloss = map(
                lambda x: x.to(device, non_blocking=True), batch
            )
            list_ = generator(x)
            logamp_g, pha_g, rea_g, imag_g, y_g = list_

            if y_g.ndim == 3:
                y_g = y_g.squeeze(1)
                
            # length alignment
            y_min = np.min([y_g.shape[-1], y.shape[-1]])
            y_g, y = y_g[..., :y_min], y[..., :y_min]

            y_g_mel = mel_spectrogram(
                y_g,
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.sampling_rate / 2,
            )
            if i % 2 == 0:  # update discriminator
                ######
                optim_d.zero_grad()

                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g.detach())
                loss_disc_f, _, _ = discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )

                y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g.detach())
                loss_disc_s, _, _ = discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )

                L_D = (loss_disc_s * h.mrd_weight + loss_disc_f) * h.loss_configs['DiscriminatorLoss']

                if torch.isnan(L_D).any():
                    print(f"Warning: Discriminator loss is NaN at step {steps}, batch {i}. Skipping this batch.")
                    continue
                
                L_D.backward()
                optim_d.step()
            else:
                ####### update Generator
                optim_g.zero_grad()

                # Define log-Amplitude Loss
                L_A = amplitude_loss(logamp, logamp_g)

                # Define Phase Loss
                if not h.use_omni_phase_loss:
                    L_P = phase_loss(pha, pha_g, h.n_fft, pha.size()[-1])
                else:
                    L_P = omni_phase_loss(pha, pha_g)

                # Define Consistency Loss
                _, _, rea_g_final, imag_g_final = amp_pha_specturm(
                    y_g, h.n_fft, h.hop_size, h.win_size
                )
                L_C = STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final)

                # Define RI Loss
                L_R = F.l1_loss(rea, rea_g)
                L_I = F.l1_loss(imag, imag_g)
                L_RI = L_R + L_I

                _, y_df_g, fmap_f_r, fmap_f_g = mpd(y, y_g)
                _, y_ds_g, fmap_s_r, fmap_s_g = mrd(y, y_g)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, _ = generator_loss(y_df_g)
                loss_gen_s, _ = generator_loss(y_ds_g)
                L_GAN_G = loss_gen_s * h.mrd_weight + loss_gen_f
                L_FM = loss_fm_s * h.mrd_weight + loss_fm_f
                if not h.use_multiscale_mel:
                    L_Mel = F.l1_loss(meloss, y_g_mel)
                else:
                    L_Mel = ms_mel_loss(y=y, y_hat=y_g)

                L_G = h.loss_configs['AmplitudeLoss'] * L_A + \
                        h.loss_configs['PhaseLoss'] * L_P + \
                        h.loss_configs['STFTConsistencyLoss'] * L_C + \
                        h.loss_configs['RILoss'] * L_RI + \
                        h.loss_configs['GeneratorLoss'] * L_GAN_G + \
                        h.loss_configs['FeatureMatchingLoss'] * L_FM + \
                        h.loss_configs['MelSpecReconstructLoss'] * L_Mel

                if torch.isnan(L_G).any():
                    print(f"Warning: Generator loss is NaN at step {steps}, batch {i}. Skipping this batch.")
                    continue
                
                L_G.backward()
                if h.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), h.clip_grad_norm)
                optim_g.step()

            # STDOUT logging
            if steps % h.stdout_interval == 0 and steps != 0:
                # for TF-domain methods...
                with torch.no_grad():
                    A_error = amplitude_loss(logamp, logamp_g).item()
                    if not h.use_omni_phase_loss:
                        Phase_error = phase_loss(
                            pha, pha_g, h.n_fft, pha.size()[-1]
                        ).item()
                    else:
                        Phase_error = omni_phase_loss(pha, pha_g).item()
                   
                    C_error = STFT_consistency_loss(
                        rea_g, rea_g_final, imag_g, imag_g_final
                    ).item()
                    R_error = F.l1_loss(rea, rea_g).item()
                    I_error = F.l1_loss(imag, imag_g).item()
                    if not h.use_multiscale_mel:
                        Mel_error = F.l1_loss(meloss, y_g_mel).item()
                    else:
                        Mel_error = ms_mel_loss(y=y, y_hat=y_g).item()

                if device_flag:
                    print(
                        "Steps : {:d}, Gen Loss Total : {:4.3f}, Amplitude Loss : {:4.3f}, Phase Loss : {:4.3f}, STFT Consistency Loss : {:4.3f}, Real Part Loss : {:4.3f}, Imaginary Part Loss : {:4.3f}, Mel Spectrogram Loss : {:4.3f}, s/b : {:4.3f}".format(
                            steps,
                            L_G,
                            A_error,
                            Phase_error,
                            C_error,
                            R_error,
                            I_error,
                            Mel_error,
                            time.time() - start_b,
                        )
                    )

            # checkpointing
            if steps % h.checkpoint_interval == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(checkpoint_path, {"generator": (generator.module if a.num_gpus > 1 else generator).state_dict()})
                remove_older_checkpoint(checkpoint_path, pre='g', max_to_keep=h.max_to_keep)
                checkpoint_path = "{}/do_{:08d}".format(h.checkpoint_path, steps)
                save_checkpoint(
                    checkpoint_path,
                    {
                        "mpd": (mpd.module if a.num_gpus > 1 else mpd).state_dict(),
                        "mrd": (mrd.module if a.num_gpus > 1 else mrd).state_dict(),
                        "optim_g": optim_g.state_dict(),
                        "optim_d": optim_d.state_dict(),
                        "steps": steps,
                        "epoch": epoch,
                    },
                )
                remove_older_checkpoint(checkpoint_path, pre='d', max_to_keep=h.max_to_keep)

            # Tensorboard summary logging
            if device_flag and steps % h.summary_interval == 0 and steps != 0:
                sw.add_scalar("Training/Generator_Total_Loss", L_G, steps)
                sw.add_scalar("Training/Mel_Spectrogram_Loss", Mel_error, steps)

            # Validation
            if device_flag and steps % h.validation_interval == 0 and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_A_err_tot = 0
                val_P_err_tot = 0
                val_C_err_tot = 0
                val_R_err_tot = 0
                val_I_err_tot = 0
                val_Mel_err_tot = 0
                pesq_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):
                        x, logamp, pha, rea, imag, y, meloss = map(
                            lambda x: x.to(device, non_blocking=True), batch
                        )
                        if hasattr(generator, "module"):
                            list_ = generator.module(x)
                        else:
                            list_ = generator(x)
                        logamp_g, pha_g, rea_g, imag_g, y_g = list_

                        if y_g.ndim == 3:
                            y_g = y_g.squeeze(1)
                        # 对齐长度
                        y_min = np.min([y_g.shape[-1], y.shape[-1]])
                        y_g, y = y_g[..., :y_min], y[..., :y_min]

                        y_g_mel = mel_spectrogram(
                            y_g,
                            h.n_fft,
                            h.num_mels,
                            h.sampling_rate,
                            h.hop_size,
                            h.win_size,
                            h.fmin,
                            h.sampling_rate / 2,
                        )

                        _, _, rea_g_final, imag_g_final = amp_pha_specturm(
                            y_g, h.n_fft, h.hop_size, h.win_size
                        )
                        val_A_err_tot = val_A_err_tot + amplitude_loss(logamp, logamp_g).item()
                        if not h.use_omni_phase_loss:
                            val_P_err = phase_loss(
                                pha, pha_g, h.n_fft, pha.size()[-1]
                            )
                        else:
                            val_P_err = omni_phase_loss(pha, pha_g)
                        val_P_err_tot = val_P_err_tot + val_P_err.item()
                        val_C_err_tot = val_C_err_tot + STFT_consistency_loss(
                            rea_g, rea_g_final, imag_g, imag_g_final
                        ).item()
                        
                        val_R_err_tot += F.l1_loss(rea, rea_g).item()
                        val_I_err_tot += F.l1_loss(imag, imag_g).item()

                        if not h.use_multiscale_mel:
                            val_Mel_err_tot = val_Mel_err_tot + F.l1_loss(meloss, y_g_mel).item()
                        else:
                            val_Mel_err_tot = val_Mel_err_tot + ms_mel_loss(y=y, y_hat=y_g).item()

                        y_g_np, y_np = y_g.cpu().squeeze().numpy(), y.cpu().squeeze().numpy()
                        
                        if h.sampling_rate != 16000:
                            y_g_np = lib.core.resample(y_g_np, orig_sr=h.sampling_rate, target_sr=16000)
                            y_np = lib.core.resample(y_np, orig_sr=h.sampling_rate, target_sr=16000)
                        cur_wb_score = pesq(16000, y_np, y_g_np, mode='wb')
                        pesq_tot += cur_wb_score

                        if j <= h.visualize_num:
                            sw.add_audio(
                                "gt/y_{}".format(j), y[0], steps, h.sampling_rate
                            )
                            y_spec= spectrogram(
                                y,
                                h.n_fft,
                                h.num_mels,
                                h.sampling_rate,
                                h.hop_size,
                                h.win_size,
                                h.fmin,
                                h.fmax,
                            )
                            sw.add_figure(
                                "gt/y_spec_{}".format(j),
                                plot_spectrogram(y_spec.squeeze(0).cpu()),
                                steps,
                            )
                            sw.add_audio(
                                "generated/y_g_{}".format(j),
                                y_g[0],
                                steps,
                                h.sampling_rate,
                            )
                            y_g_spec = spectrogram(
                                y_g,
                                h.n_fft,
                                h.num_mels,
                                h.sampling_rate,
                                h.hop_size,
                                h.win_size,
                                h.fmin,
                                h.fmax,
                            )
                            sw.add_figure(
                                "generated/y_g_spec_{}".format(j),
                                plot_spectrogram(y_g_spec.squeeze(0).cpu().numpy()),
                                steps,
                            )

                    val_A_err = val_A_err_tot / (j + 1)
                    val_P_err = val_P_err_tot / (j + 1)
                    val_C_err = val_C_err_tot / (j + 1)
                    val_R_err = val_R_err_tot / (j + 1)
                    val_I_err = val_I_err_tot / (j + 1)
                    val_Mel_err = val_Mel_err_tot / (j + 1)
                    val_pesq_score = pesq_tot / (j + 1)
                    sw.add_scalar("Validation/Amplitude_Loss", val_A_err, steps)
                    sw.add_scalar(
                        "Validation/Phase_Loss", val_P_err, steps
                    )
                    sw.add_scalar("Validation/STFT_Consistency_Loss", val_C_err, steps)
                    sw.add_scalar("Validation/Real_Part_Loss", val_R_err, steps)
                    sw.add_scalar("Validation/Imaginary_Part_Loss", val_I_err, steps)
                    sw.add_scalar("Validation/Mel_Spectrogram_loss", val_Mel_err, steps)
                    sw.add_scalar("Validation/PESQ_score", val_pesq_score, steps)
                    # save the model with the highest PESQ score (in case)
                    if "save_best" in h:
                        if h.save_best:
                            if val_pesq_score > global_pesq_score:
                                ckp_best_path = "{}/best_g".format(h.checkpoint_path)
                                save_checkpoint(ckp_best_path, {"generator": (generator.module if a.num_gpus > 1 else generator).state_dict()})
                                global_pesq_score = val_pesq_score

                generator.train()

            steps = steps + 1
            if steps == h.training_steps + 1:
                exit()

        scheduler_g.step()
        scheduler_d.step()

        print(
            "Time taken for epoch {} is {} sec\n".format(
                epoch + 1, int(time.time() - start)
            )
        )


def main():
    print("Initializing Training Process...")
    parse = argparse.ArgumentParser('Vocoder configs.')
    parse.add_argument('--cfg_filename', type=str, required=True, default='./cfgs/libritts/mid_rnd_shared_ende_libritts_config.json',
                        help='Json for configurations.')
    args = parse.parse_args()
    
    # Set GPU envs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    config_file = args.cfg_filename
    try:
        config_file_list = os.path.split(config_file)
        if len(config_file_list) > 1:
            config_filename = config_file_list[-1]
    except:
        config_filename = config_file

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(config_file, config_filename, h.checkpoint_path)
    args.num_gpus = torch.cuda.device_count()

    # torch.manual_seed(h.seed)
    if args.num_gpus > 1:
        mp.spawn(
            train,
            nprocs=args.num_gpus,
            args=(args, h),
        )
    else:
        train(0, args, h)

if __name__ == "__main__":
    main()
