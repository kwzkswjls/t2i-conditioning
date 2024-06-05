import torch
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("../ptms/runwayml/stable-diffusion-v1-5", )
pipeline = pipeline.to("cuda")
generator = torch.Generator("cuda").manual_seed(0)

image = pipeline(
    prompt="cinematic photo of a plush and soft midcentury style rug on a wooden floor, 35mm photograph, film, professional, 4k, highly detailed",
    generator=generator
).images[0]
image.show()
