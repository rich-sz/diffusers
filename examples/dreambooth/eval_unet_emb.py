from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

# Load the pipeline with the same arguments (model, revision) that were used for training
model_id = "CompVis/stable-diffusion-v1-4"

unet = UNet2DConditionModel.from_pretrained("/sddata/dreambooth/daruma-v2-1/checkpoint-100/unet")

# if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
text_encoder = CLIPTextModel.from_pretrained("/sddata/dreambooth/daruma-v2-1/checkpoint-100/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
pipeline.to("cuda")

# Perform inference, or save, or push to the hub
pipeline.save_pretrained("dreambooth-pipeline")
