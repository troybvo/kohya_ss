#!/usr/bin/env bash

/workspace/kohya_ss/venv/bin/python "/workspace/kohya_ss/sd-scripts/finetune/make_captions.py" \
  --batch_size="1" \
  --num_beams="1" \
  --top_p="0.9" \
  --max_length="75" \
  --min_length="5" \
  --beam_search \
  --caption_extension=".caption" \
  --train_data_dir="/workspace/input_images" \
  --caption_weights="https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth"