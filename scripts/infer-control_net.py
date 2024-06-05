import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler, ControlNetModel
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained(
    "../ptms/lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16,
)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "../ptms/runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    safty_checker=None,
    torch_dtype=torch.float16
)
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()
pipeline.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

condition_image = load_image("../data/images/hf-logo.png")
condition_image = np.array(condition_image)
low_threshold = 100
high_threshold = 200
condition_image = cv2.Canny(condition_image, low_threshold, high_threshold)
condition_image = Image.fromarray(condition_image)

image = pipeline(
    prompt="cinematic photo of a plush and soft midcentury style rug on a wooden floor, 35mm photograph, film, professional, 4k, highly detailed",
    image=condition_image,
    generator=generator,
).images[0]
image.show()
