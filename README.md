# Scalable Neural Vocoder from Range-Null Space Decomposition (submitted to IEEE TPAMI)

### Authors: Andong Li, Tong Lei, Zhihang Sun, Rilin Chen, Xiaodong Li, Dong Yu, and Chengshi Zheng

<a href=""><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a> ![Replicate](https://img.shields.io/badge/Demo-%F0%9F%9A%80%20Replicate-blue) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=Andong-Li-speech/RNDVoC)

---
## Update
- **2025.10.10**: Update load the project code and pretrained checkpoint.
- **2025.04.29**： Our work has been accepted by IJCAI 2025!!
- ~~**2025.01.09**: Create this repo.~~
  
## Abstract
Although deep neural networks have facilitated significant progress of neural vocoders in recent years, they usually suffer
from intrinsic challenges like opaque modeling, inflexible retraining under different input configurations, and parameter-performance
trade-off. These inherent hurdles can heavily impede the development of this field. To resolve these problems, in this paper, we
propose a novel neural vocoder in the time-frequency (T-F) domain. Specifically, we bridge the connection between the classical
range-null decomposition (RND) theory and the vocoder task, where the reconstruction of the target spectrogram is formulated into the
superimposition between range-space and null-space. The former aims to project the representation in the original mel-domain into the
target linear-scale domain, and the latter can be instantiated via neural networks to further infill the spectral details. To fully leverage
the spectrum prior, an elaborate dual-path framework is devised, where the spectrum is hierarchically encoded and decoded, and the
cross- and narrow-band modules are leveraged for effectively modeling along sub-band and time dimensions. To enable inference
under various configurations, we propose a simple yet effective strategy, which transforms the multi-condition adaption in the inference
stage into the data augmentation in the training stage. Comprehensive experiments are conducted on the LJSpeech and LibriTTS
benchmarks. Quantitative and qualitative results show that while enjoying lightweight network structure and scalable inference
paradigm, the proposed framework achieves state-of-the-art performance among existing advanced methods. 

## 🚀 Quick Start

### Checkpoint Download (Welcome for use)
Pre-trained model checkpoints are available. Please use the following links to download the checkpoint:
|  Model Name   |  Dataset    |  Sample Rate   |    NMel   | Training Steps
|---------------|-------------|----------------|-----------|---------------|
|🤗 [RNDVoC-LibriTTS](https://huggingface.co/AndongLi/RNDVoC/blob/main/best_g_libritts)  |   LibriTTS  |    24k Hz     |   100  |  2M (1M Generator + 1M Discriminator)
|🤗 [RNDVoC-LJSpeech](https://huggingface.co/AndongLi/RNDVoC/blob/main/best_g_ijs)  |   LJSpeech  |    22.05k Hz     |   80  |  2M (1M Generator + 1M Discriminator)

### Inference from Natural Mel
You can reconstruct target audio signals from natural mels, that it, you first extract the mels from the original waveforms, and then synthesis based on the neural vocoder.

```bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 inference.py \
                        --cfg_filename ./cfgs/libritts/mid_rnd_shared_ende_libritts_config.json \
                        --test_input_wavs_dir ./test_wavs \
                        --test_output_dir ./test_wavs_est \
                        --checkpoint_file_load ./Experiments/LibriTTS/RNDVoc_shared_ende_benchmark/best_g_libritts
```
### Inference from Synthesis Mel
You can also generate wavefroms from synthesized mels, which is common in TTS/T2A...

```bash
CUDA_VISIBLE_DEVICES="0" python3 inference.py  \
                           --cfg_filename ./cfgs/libritts/mid_rnd_shared_ende_libritts_config.json \
                           --use_mel_format \
                           --test_input_mels_dir ./test_wavs \
                           --test_output_dir ./test_wavs_est \
                           --checkpoint_file_load ./Experiments/LibriTTS/RNDVoc_shared_ende_benchmark/best_g_libritts
```

### Training the neural vocoder with your own dataset
* Prepare your own scp file for training and validation, and put them in ./DatasetsScp
* Modify the configurations in the json file (e.g., ./cfg/libritts/mid_rnd_shared_ende_libritts_config.json), including input_training_wav_list, nput_validation_wav_list, raw_wavfile_path, and checkpoint_path...
* Modify the func  ``get_dataset_filelist'' in the dataset.py according to your wavfile format in the scp
* ```bash
  ./train.sh
  ```

## Experiments

### Bubble Figure on LibriTTS benchmark

<img src="./figure/bubble_figure.png" width="60%" height="auto">

### On LJSeech benchmark
<img src="./figure/performance_ljs.png" width="90%" height="auto">

### On LibriTTS benchmark
<img src="./figure/performance_libritts.png" width="90%" height="auto">

### Visualization on MUSDB18 test set
#### Vocal voice
<img src="./figure/vocal_visualization.png" width="90%" height="auto">

#### Bass sound
<img src="./figure/bass_visualization.png" width="90%" height="auto">

### Apply RND to existing T-F domain-based vocoders (Plug-and-play)
<img src="./figure/pnp.png" width="50%" height="auto">

### Toward light-weight design
<img src="./figure/light-weight-desgn.png" width="60%" height="auto">

## Citation
If you find this repo helpful, please cite our work:
```bibtex
@inproceedings{ijcai2025p904,
  title     = {Learning Neural Vocoder from Range-Null Space Decomposition},
  author    = {Li, Andong and Lei, Tong and Sun, Zhihang and Chen, Rilin and Yin, Erwei and Li, Xiaodong and Zheng, Chengshi},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {8131--8140},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/904},
  url       = {https://doi.org/10.24963/ijcai.2025/904},
}
```






