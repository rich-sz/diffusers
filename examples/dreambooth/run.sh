export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/yuruiqi/diffusion-nbs/pre_proc_dir_select"
export CLASS_DIR="/home/yuruiqi/diffusion-nbs/class_dir"
export OUTPUT_DIR="/home/yuruiqi/diffusion-nbs/DB_15"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export CUDA_VISIBLE_DEVICES=1
export CUDA_LAUNCH_BLOCKING=1

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of soldier-girl person" \
  --class_prompt="a photo of person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1\
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=10 \
  --num_class_images=300 \
  --max_train_steps=3000 \
  --checkpointing_steps=500 \
  --train_text_encoder \
#  --gpu_ids 1,2

#  --use_8bit_adam

