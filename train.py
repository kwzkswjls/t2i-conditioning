import argparse
import gc
import logging
import math
from pathlib import Path

import accelerate
import accelerate.logging
import diffusers
import torch
import torch.nn.functional as F
import transformers
import transformers.utils.logging
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from dataset import DatasetLaion
from models.adapter import Adapter
from models.utils import encode_prompt

logger = accelerate.logging.get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Adapter training script.")
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--training_data",
        type=str,
        default=None,
        required=True,
        help="Path to training data file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/adapter",
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for the training dataloader."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="A higher guidance scale value encourages the model to generate images closely linked to the text"
             "`prompt` at the expense of lower image quality."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="The number of denoising steps. More denoising steps usually lead to a higher quality image at the"
             "expense of slower inference."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        )
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler."
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler."
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        )
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        )
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        )
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement)."
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="Erasing-Adapter",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
        )
    )

    if input_args is not None:
        for action in parser._actions:
            if action.dest != "==SUPPRESS==":
                if not hasattr(input_args, action.dest):
                    if action.default != "==SUPPRESS==":
                        setattr(input_args, action.dest, action.default)
        args = input_args
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    return args


def main(args):
    output_dir = Path(args.output_dir)
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = accelerate.utils.ProjectConfiguration(project_dir=args.output_dir,
                                                                       logging_dir=str(logging_dir))
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        accelerate.utils.set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

    sd = diffusers.StableDiffusionPipeline.from_pretrained(args.pretrained_model)
    noise_scheduler = sd.components["scheduler"]
    text_encoder = sd.components["text_encoder"]
    tokenizer = sd.components["tokenizer"]
    unet = sd.components["unet"]

    def save_model_hook(models, weights, output_dir):
        i = len(weights) - 1

        while len(weights) > 0:
            weights.pop()
            model = models[i]
            torch.save(model.state_dict(), Path(output_dir) / f"model_{i:02d}.pth")
            i -= 1

    accelerator.register_save_state_pre_hook(save_model_hook)

    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.batch_size * accelerator.num_processes
        )

    adapter = Adapter()
    summary(adapter)

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    adapter.to(accelerator.device, dtype=torch.float32)

    gc.collect()
    torch.cuda.empty_cache()

    dataset = DatasetLaion(args.training_data)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        drop_last=True
    )

    uncond_tokens = [""] * args.batch_size
    uncond_embeds, uncond_embeds_pooled = encode_prompt(uncond_tokens, text_encoder, tokenizer)

    num_update_steps_per_epoch = math.ceil(1e7 / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = diffusers.optimization.get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # noinspection PyTypeChecker
    adapter, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        adapter, optimizer, dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(
            args.tracker_project_name,
            config=dict(vars(args) if isinstance(args, argparse.Namespace) else args)
        )

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):

        for it, batch in enumerate(dataloader):
            with accelerator.accumulate(adapter):

                latents_shape = (
                    args.batch_size,
                    unet.config.in_channels,
                    unet.config.sample_size,
                    unet.config.sample_size
                )
                latents = randn_tensor(latents_shape, device=accelerator.device, dtype=weight_dtype)
                latents = latents * noise_scheduler.init_noise_sigma

                noise_scheduler.set_timesteps(args.num_inference_steps, device=accelerator.device)
                timesteps = noise_scheduler.timesteps

                prompts_text = batch["concept"] + batch["original prompt"] + batch["adapted prompt"]
                prompt_embeds, prompt_embeds_pooled = encode_prompt(
                    prompt_batch=prompts_text,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer
                )
                negative_prompt_embeds, original_prompt_embeds, adapted_prompt_embeds = prompt_embeds.chunk(3)
                concept_embeds, _, _ = prompt_embeds_pooled.chunk(3)

                for i, t in enumerate(timesteps):
                    adapter_cond = adapter(concept_embeds)
                    adapter_uncond = adapter(uncond_embeds_pooled)
                    latents_model_input = torch.cat([latents] * 3)
                    latents_model_input = noise_scheduler.scale_model_input(latents_model_input, t)

                    target = unet(
                        latents_model_input,
                        t,
                        encoder_hidden_states=torch.cat([
                            negative_prompt_embeds,
                            uncond_embeds,
                            adapted_prompt_embeds
                        ]),
                    )[0]
                    target_negative, target_uncond, target_adapted = target.chunk(3)
                    target = target_negative + args.guidance_scale * (target_adapted - target_negative)

                    pred = unet(
                        latents_model_input,
                        t,
                        encoder_hidden_states=torch.cat([
                            uncond_embeds,
                            original_prompt_embeds,
                            original_prompt_embeds
                        ]),
                        down_intrablock_additional_residuals=[
                            torch.cat([adapter_cond[idx], adapter_cond[idx], adapter_uncond[idx]])
                            for idx in range(unet.config.in_channels)
                        ]
                    )[0]
                    pred_uncond, pred_original, pred_no_concept = pred.chunk(3)
                    pred = pred_uncond + args.guidance_scale * (pred_original - pred_uncond)

                    loss_erasing = F.mse_loss(pred, target)
                    loss_quality = F.mse_loss(pred_no_concept, target_uncond)
                    accelerator.backward(loss_erasing + loss_quality)

                    latents = noise_scheduler.step(target, t, latents)[0]

                    if accelerator.sync_gradients:
                        params_to_clip = adapter.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        if accelerator.is_main_process:
                            if global_step % args.checkpointing_steps == 0:
                                save_path = output_dir / f"checkpoint-{global_step}"
                                accelerator.save_state(str(save_path))
                                logger.info(f"Saved state to {save_path}")

                    logs = {
                        "loss_erasing": loss_erasing.detach().item(),
                        "loss_quality": loss_quality.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0]
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                save_path = output_dir / f"checkpoint-{global_step}"
                accelerator.save_state(str(save_path))
                logger.info(f"Saved state to {save_path}")
                break

        save_path = output_dir / f"checkpoint-{global_step}"
        accelerator.save_state(str(save_path))
        logger.info(f"Saved state to {save_path}")


if __name__ == "__main__":
    main(parse_args())
