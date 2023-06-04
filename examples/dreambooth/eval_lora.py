import torch
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel
from PIL import Image

model_base = "runwayml/stable-diffusion-v1-5"
model_path = "/home/yuruiqi/diffusion-nbs/DB_15_lora_2"

pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda:3")


def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    # grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# use half the weights from the LoRA finetuned model and half the weights from the base model
# image = pipe("A photo of sks person rides a motorcycle",
#              num_inference_steps=25,
#              guidance_scale=7.5,
#              cross_attention_kwargs={"scale": 0.5},
#              ).images[0]

# use the weights from the fully finetuned LoRA model
# image = pipe("A photo of sks person rides a motorcycle", num_inference_steps=25, guidance_scale=7.5).images[0]

# image.show()
# image.save("bucket-dog.png")


eval_seed = 19940608
key_word = "sks"
prompt = f"A photo of {key_word} person rides a motorcycle"
prompt1 = "A photo of person rides a motorcycle"
prompts = iter([prompt, prompt1])
print(prompt, "|", prompt1)

# Use negtive prompt.
neg_prompt = "icon, frame"

num_samples = 4
num_rows = 2

all_images = []
torch.manual_seed(eval_seed)
for _ in range(num_rows):
    images = pipe([next(prompts)] * num_samples, negative_prompt=[neg_prompt] * num_samples, num_inference_steps=30,
                  guidance_scale=7.5).images
    # images = pipe([next(prompts)] * num_samples, num_inference_steps=30, guidance_scale=7.5).images
    all_images.extend(images)

grid = image_grid(all_images, num_rows, num_samples)
grid.save("sks-person1.png")
