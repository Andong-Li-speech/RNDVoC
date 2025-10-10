from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import json
import torch
from utils import AttrDict
from dataset_ljs import mel_spectrogram, load_wav
import soundfile as sf
import numpy as np
import soundfile as sf
import librosa as lib
import time
import tqdm
from ptflops import get_model_complexity_info

from Models.rndvoc_22k import RNDVocoder22k
from Models.rndvoc_24k import RNDVocoder24k

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(h):
    generator = eval(h.model_name)(h).to(device)
    get_model_complexity_info(generator, (h.num_mels, 5*h.sampling_rate//h.hop_size))
    state_dict_g = load_checkpoint(h.checkpoint_file_load, device)
    generator.load_state_dict(state_dict_g['generator'])

    generator.cuda()
    if not h.test_mel_load:
        post_str = os.path.splitext(h.test_input_wavs_dir)[-1]
        if post_str in ['.txt', '.scp']:
            filelist = []
            lines = open(h.test_input_wavs_dir, 'r').readlines()
            for l in lines:
                if h.dataset_type.lower() == "libritts":
                    cur_filename = l.strip().split('|')[0] + ".wav"
                elif h.dataset_type.lower() == "ljspeech":
                    cur_filename = l.strip().split('/')[1].split('|')[0]
                filelist.append(os.path.join(h.raw_wavfile_path, cur_filename))
        else:  # dir
            filelist = sorted(os.listdir(h.test_input_mels_dir if h.test_mel_load else h.test_input_wavs_dir))
    else:
        filelist = sorted(os.listdir(h.test_input_mels_dir))

    os.makedirs(h.test_output_dir, exist_ok=True)

    generator.eval()
    try:
        generator.remove_weight_norm()
    except:
        pass
    l = 0
    with torch.no_grad():
        starttime = time.time()
        for filename in tqdm.tqdm(filelist):

            if h.test_mel_load:
                mel = np.load(os.path.join(h.test_input_mels_dir, filename))
                x = torch.FloatTensor(mel.astype('float32')).unsqueeze(0).cuda()
                x = torch.log(torch.clamp(torch.exp(x), min=1e-5))
            else:
                if post_str in ['.txt', '.scp']:
                    raw_wav, orig_sr = sf.read(filename)
                else:
                    raw_wav, orig_sr = sf.read(os.path.join(h.test_input_wavs_dir, filename))
                if orig_sr != h.sampling_rate:
                    raw_wav = lib.core.resample(raw_wav, orig_sr=orig_sr, target_sr=h.sampling_rate)

                raw_wav = torch.FloatTensor(raw_wav.astype('float32')).to(device)
                x = get_mel(raw_wav.unsqueeze(0)).cuda()
            
            y_g_list = generator(x)
            if isinstance(y_g_list, torch.Tensor):  # 对于时域方法
                y_g = y_g_list
            else:
                y_g = y_g_list[-1]  # 对于频域方法
            audio = y_g.squeeze()
            audio = audio.cpu().numpy()
            audiolen=len(audio)
            if not h.test_mel_load:
                if post_str in ['.txt', '.scp']:
                    sf.write(os.path.join(h.test_output_dir, filename.split('/')[-1]), audio, h.sampling_rate, 'PCM_16')
                else:
                    sf.write(os.path.join(h.test_output_dir, filename.split('.')[0] + '.wav'), audio, h.sampling_rate, 'PCM_16')
            else:
                audio = 0.5 * audio / np.max(np.abs(audio))
                sf.write(os.path.join(h.test_output_dir, filename.split('.')[0] + '.wav'), audio, h.sampling_rate, 'PCM_16')

            l += audiolen
            
        end=time.time()
        print(end-starttime)
        print(l/22050)
        print(l/22050/(end-starttime)) 


def main():
    print('Initializing Inference Process..')

    config_file = './cfgs/libritts/mid_rnd_shared_ende_libritts_config.json'

    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    device = torch.device('cpu')
    inference(h)


if __name__ == '__main__':
    main()
