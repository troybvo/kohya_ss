#!/usr/bin/env bash
# 20_nightstand nightstand
# 1_bedroom
# reg beta = 2 if reg else 1
# max_train_step = num_images * repeat * epoch * reg beta = 4 * 20 * 5 * 1
source /workspace/kohya_ss/venv/bin/activate && accelerate launch \
  --num_cpu_threads_per_process=2 "/workspace/kohya_ss/sd-scripts/sdxl_train_network.py" \
  --bucket_no_upscale --bucket_reso_steps=64 \
  --cache_latents --enable_bucket --min_bucket_reso=256 \
  --max_bucket_reso=2048 --learning_rate="0.0005" \
  --logging_dir="nightstand/log" \
  --lr_scheduler="cosine" --lr_scheduler_num_cycles="1" \
  --lr_warmup_steps="80" --max_data_loader_n_workers="0" \
  --max_grad_norm="1" --resolution="512,512" \
  --max_train_steps="800" --mixed_precision="fp16" \
  --network_alpha="1" --network_dim=8 \
  --network_module=networks.lora --no_half_vae \
  --optimizer_type="AdamW8bit" \
  --output_dir="nightstand/model" \
  --output_name="ns_v1" \
  --pretrained_model_name_or_path="/workspace/stable-diffusion-webui/models/Stable-diffusion/sd_xl_base_1.0_0.9vae.safetensors" \
  --save_model_as=safetensors \
  --save_precision="fp16" --text_encoder_lr=0.0001 \
  --train_batch_size="1" \
  --train_data_dir="nightstand/img" --unet_lr=0.0001 \
  --xformers \
  --validation_prompt="a night stand" \
  --num_validation_images=1 \
  --sample_every_n_steps=100