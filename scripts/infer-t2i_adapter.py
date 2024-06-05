import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter
from diffusers.utils import load_image

adapter = T2IAdapter.from_pretrained(
    "../ptms/TencentARC/t2iadapter_canny_sd15v2",
    torch_dtype=torch.float16
)
pipeline = StableDiffusionAdapterPipeline.from_pretrained(
    "../ptms/runwayml/stable-diffusion-v1-5",
    adapter=adapter,
    safety_checker=None,
    torch_dtype=torch.float16

)
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
