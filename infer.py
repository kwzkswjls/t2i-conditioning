import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from conditions import condition_canny as get_condition
from dataset import DatasetLaion, safe_collate
from models.utils import save_images


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script.")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        required=True,
        help="Choose between ['unconditional', 'control-net', 't2i-adapter']"
    )
    parser.add_argument(
        "--evaluation-samples",
        type=int,
        default=1000,
        required=False,
        help="Minimum number of samples to generate."
    )

    args = parser.parse_args()
    return args


dataset = DatasetLaion(
    r"data/datasets/laion400m/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet")
dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=safe_collate)
opt = parse_args()

if opt.model_type == "unconditional":
    from diffusers import StableDiffusionPipeline

    pipeline = StableDiffusionPipeline.from_pretrained(
        "ptms/runwayml/stable-diffusion-v1-5",
        safty_checker=None,
    )
elif opt.model_type == "control_net":
    from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler, ControlNetModel

    controlnet = ControlNetModel.from_pretrained(
        "ptms/lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16,
    )
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        "ptms/runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safty_checker=None,
        torch_dtype=torch.float16
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.enable_model_cpu_offload()
elif opt.model_type == "t2i_adapter":
    from diffusers import StableDiffusionAdapterPipeline, T2IAdapter

    adapter = T2IAdapter.from_pretrained(
        "ptms/TencentARC/t2iadapter_canny_sd15v2",
        torch_dtype=torch.float16
    )
    pipeline = StableDiffusionAdapterPipeline.from_pretrained(
        "ptms/runwayml/stable-diffusion-v1-5",
        adapter=adapter,
        safety_checker=None,
        torch_dtype=torch.float16
    )
else:
    raise ValueError(f"Unknown model type: {opt.model_type}")
pipeline = pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=True)

counts = 0
for batch in dataloader:
    if batch is None:
        continue

    prompt = batch["TEXT"]
    if opt.model_type == "unconditional":
        generated_image = pipeline(
            prompt=prompt,
        ).images
    elif opt.model_type == "control_net":
        condition_image = get_condition(batch["IMAGE"])
        generated_image = pipeline(
            prompt=prompt,
            image=condition_image,
        ).images
    elif opt.model_type == "t2i_adapter":
        condition_image = get_condition(batch["IMAGE"])
        generated_image = pipeline(
            prompt=prompt,
            image=condition_image,
        ).images
    else:
        raise ValueError(f"Unknown model type: {opt.model_type}")

    dire = Path(f"data/cache/{opt.model_type}")
    dire.mkdir(parents=True, exist_ok=True)
    save_images(generated_image, batch["TEXT"], dire)

    counts += len(batch)
    if counts >= opt.evaluation_samples:
        break
