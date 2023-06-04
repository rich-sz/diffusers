export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR="/home/yuruiqi/diffusion-nbs/pre_proc_dir_select"
export OUTPUT_DIR="/home/yuruiqi/diffusion-nbs/DB_21_lora"
#export CLASS_DIR="/home/yuruiqi/diffusion-nbs/class_dir"
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128
# ["stabilityai/stable-diffusion-2-1",
# "stabilityai/stable-diffusion-2-1-base",
# "stabilityai/stable-diffusion-2",
# "stabilityai/stable-diffusion-2-base",
# "CompVis/stable-diffusion-v1-4",
# "runwayml/stable-diffusion-v1-5"]

# temporary no 768 data.
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of soldier-girl person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --report_to="wandb" \
  --lr_scheduler="polynomial" \
  --max_train_steps=3000 \
  --validation_prompt="A photo of soldier-girl person rides a motorcycle" \
  --validation_epochs=50 \
  --checkpointing_steps=400 \
  --seed="19940608" \

#  --train_text_encoder \
#  --class_data_dir=$CLASS_DIR \
#  --class_prompt="a photo of person" \
#  --push_to_hub
#  --with_prior_preservation --prior_loss_weight=1.0 \
#  --lr_warmup_steps=10 \



