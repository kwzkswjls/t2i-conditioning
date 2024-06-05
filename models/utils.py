import torch
from PIL import Image
from torchvision.transforms import ToPILImage
from base64 import urlsafe_b64encode, urlsafe_b64decode
from pathlib import Path

def encode_prompt(prompt_batch, text_encoder, tokenizer):
    with torch.no_grad():
        tokens = tokenizer(
            prompt_batch,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = text_encoder(
            tokens.input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

    return prompt_embeds.last_hidden_state, prompt_embeds.pooler_output


def decode_image(latents, image_decoder) -> list[Image]:
    with torch.no_grad():
        images = image_decoder(latents, return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)
        transform = ToPILImage()
        images = [transform(img) for img in images]

    return images


def image_grid(imgs, rows, cols, padding=2, img_type="pil", save_to=None):
    n_imgs = len(imgs)
    assert n_imgs == rows * cols

    if img_type == "pil":
        w, h = imgs[0].size
        grid = Image.new(
            'RGB',
            size=(cols * (w + padding) - padding, rows * (h + padding) - padding)
        )
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * (w + padding), i // cols * (h + padding)))
    elif img_type == "tensor":
        w, h = imgs[0].shape[-2:]
        grid = torch.zeros(3, rows * (h + padding) - padding, cols * (w + padding) - padding, dtype=torch.uint8)
        for i, img in enumerate(imgs):
            row, col = i // cols, i % cols
            grid[:, row * (h + padding):row * (h + padding) + h, col * (w + padding):col * (w + padding) + w] = img
        grid = ToPILImage()(grid)
    else:
        raise ValueError("img_type must be either 'pil' or 'tensor'")

    if save_to:
        grid.save(save_to)
    return grid


def image_gif(imgs, duration=5, loop=0, save_to="output.gif"):
    imgs[0].save(
        save_to,
        save_all=True,
        append_images=imgs[1:],
        duration=duration,
        loop=loop
    )


def save_images(imgs, captions, dire):
    for img, caption in zip(imgs, captions):
        img.save(f"{dire}/{urlsafe_b64encode(caption.encode()).decode()}.png")


def load_images(dire: Path):
    imgs = []
    captions = []
    for img_path in dire.iterdir():
        img = Image.open(img_path)
        caption = img_path.stem
        caption = urlsafe_b64decode(caption.encode()).decode()
        imgs.append(img)
        captions.append(caption)
    return imgs, captions


