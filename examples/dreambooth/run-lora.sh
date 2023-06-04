export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/yuruiqi/diffusion-nbs/pre_proc_dir_select"
export OUTPUT_DIR="/home/yuruiqi/diffusion-nbs/DB_15_lora_2"
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

wandb login --relogin
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks person rides a motorcycle" \
  --validation_epochs=50 \
  --seed="0" \
#  --push_to_hub
