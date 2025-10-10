CUDA_VISIBLE_DEVICES="0" python3 inference.py  \
                           --cfg_filename ./cfgs/libritts/mid_rnd_shared_ende_libritts_config.json \
                           --use_mel_format \
                           --test_input_mels_dir ./test_wavs \
                           --test_output_dir ./test_wavs_est \
                           --checkpoint_file_load ./Experiments/LibriTTS/RNDVoc_shared_ende_benchmark/best_g
