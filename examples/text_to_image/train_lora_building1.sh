export MODEL_NAME="/home/yuruiqi/diffusion-nbs/sd15"
export DATASET_NAME="/home/yuruiqi/diffusion-nbs/building1"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUDA_VISIBLE_DEVICES=4
#export CUDA_LAUNCH_BLOCKING=1

accelerate launch --mixed_precision="fp16" train_text_to_image_lora_rui.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --checkpointing_steps=500 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="/home/yuruiqi/diffusion-nbs/building-lora-15-2" \
  --validation_prompt="a building one the ground." \
  --report_to="wandb" \
  --snr_gamma=5.0 \
  --noise_offset=0.05 \
  --save_loss_threshold=0.001 \
  --loss_threshold_save_gap=100 \
#  --scale_lr \

#  --max_train_samples=10
# --num_train_epochs=100
# --gradient_accumulation_steps
# --gradient_checkpointing
