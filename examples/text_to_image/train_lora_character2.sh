export MODEL_NAME="/home/yuruiqi/diffusion-nbs/sd15"
export DATASET_NAME="/home/yuruiqi/diffusion-nbs/character2/inpainting_res1"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora_rui.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir="/home/yuruiqi/diffusion-nbs/character2-lora-15" \
  --validation_prompt="a girl." \
  --report_to="wandb" \
  --snr_gamma=5.0 \
  --noise_offset=0.05 \
  --save_loss_threshold=0.002 \
  --loss_threshold_save_gap=150 \

