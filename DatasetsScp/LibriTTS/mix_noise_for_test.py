import argparse
import os
import random
from datetime import datetime
import soundfile as sf
import librosa as lib
import numpy as np
import pymp
from typing import *
from tqdm import tqdm
from progressbar import ProgressBar, Bar, Timer, ETA, Percentage

test_raw_speech_path = "/data4/liandong/datasets/LibriTTS/LibriTTS"
test_speech_scp_path = ["/data4/liandong/PROJECTS/FreeV/DatasetsScp/LibriTTS/dev-clean.txt",
                        "/data4/liandong/PROJECTS/FreeV/DatasetsScp/LibriTTS/dev-other.txt"
                        ]
test_noise_path = "/data4/liandong/datasets/Noises/musan_freesound_1_50_noise.bin"
test_mixture_path = "/data4/liandong/datasets/LibriTTS/test_mixture_dev/dev_clean+dev_other_from_vits_division"

######## settings
num_sentences = 300

test_snr_list = [0.0, 1.0, 2.0, 3.0, 4.10, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
sr = 24000

# read speech
speech_filename_list = []
if isinstance(test_speech_scp_path, str):
    test_speech_scp_path = [test_speech_scp_path]
for cur_scp_path in test_speech_scp_path:
    lines = open(cur_scp_path, 'r').readlines()
    for l in lines:
        cur_filename = l.strip().split('|')[0] + '.wav'
        speech_filename_list.append(cur_filename)

num_sentences = np.min([num_sentences, len(speech_filename_list)])

print(f'Using {len(speech_filename_list)} wav files for mixtures.')
print(f'SNR levels: {test_snr_list}')

# read noises
n = np.memmap(test_noise_path, dtype=np.float32, mode='r')

# custom a progressbar
if not os.path.exists(test_mixture_path):
    os.makedirs(test_mixture_path)
write_info_scp = open(os.path.join('/'.join(test_mixture_path.split('/')[:-1]), 'dev_clean_other_from_vits.scp'), 'w')
for count in tqdm(range(num_sentences)):
    random.seed(datetime.now())
    snr_c = random.randint(0, len(test_snr_list) - 1)
    s, ori_sr = sf.read(os.path.join(test_raw_speech_path, speech_filename_list[count]))
    if ori_sr != sr:
        s = lib.core.resample(s, orig_sr=ori_sr, target_sr=sr)
    snr = test_snr_list[snr_c]

    # choose a point where we start to cut
    start_cut_point = random.randint(0, len(n) - len(s))
    while np.sum(n[start_cut_point:start_cut_point+s.size]**2.0) == 0.0:
        start_cut_point = random.randint(0,n.size-s.size)
    # cut noise
    n_t = n[start_cut_point: start_cut_point + len(s)]

    alpha = np.sqrt(np.sum(s**2.0)/(np.sum(n_t**2.0)*(10.0**(snr/10.0))))
    snr_check = 10.0*np.log10(np.sum(s**2.0)/(np.sum((n_t*alpha)**2.0)))
    mix = s + alpha * n_t
    
    mixture_file_path = os.path.join(test_mixture_path, os.path.split(speech_filename_list[count])[-1])
    write_info_scp.write(os.path.split(speech_filename_list[count])[-1] + '\t' + '%.2f dB' % (snr) + '\n')
    sf.write(mixture_file_path, mix, samplerate=sr)

write_info_scp.close()
