import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

pretrained_model = "ptms/openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(pretrained_model)
clip_processor = CLIPProcessor.from_pretrained(pretrained_model)
clip_model = clip_model.to("cuda")


def evaluate_clip_score(images, prompts, batch_size=None):
    if batch_size is None:
        batch_size = len(prompts)

    offset = 0
    clip_score = 0
    for i in tqdm(range(0, len(prompts), batch_size)):
        clip_input = clip_processor(
            text=prompts[i:i + batch_size],
            images=images[i:i + batch_size],
            return_tensors="pt",
            padding=True
        )
        clip_input = clip_input.to("cuda")
        clip_output = clip_model(**clip_input)
        text_embedding = clip_output.text_embeds
        image_embedding = clip_output.image_embeds
        clip_score = torch.tensordot(text_embedding, image_embedding)
        offset += batch_size

    return clip_score / len(prompts)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from models.utils import load_images

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('path', type=str, nargs=1,
                        help=('Paths to the generated images or '
                              'to .npz statistic files'))
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size to use')

    args = parser.parse_args()
    images, prompts = load_images(Path(args.path[0]))
    score = evaluate_clip_score(images, prompts, args.batch_size)
    print(f"CLIP score: {score}")
