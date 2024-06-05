# Evaluate image conditioning methods for text-to-image diffusion models

[Control-Net](https://arxiv.org/abs/2302.05543) and [T2I-Adatper](https://arxiv.org/abs/2302.08453) 
are two image conditioning methods that can be used to improve the performance of text-to-image diffusion models. 
This project aims to evaluate these methods on FID score and CLIP score.

The conditioning models are fine-tuned on the [Laion-400M](https://www.kaggle.com/datasets/romainbeaumont/laion400m) 
dataset with canny edge maps as the conditioning input.

The FID score is calculated on the validation set of the 
[COCO-2014](https://www.kaggle.com/datasets/jeffaudi/coco-2014-dataset-for-yolov3) dataset.
And the CLIP score is calculated by using [OpenAI/clip-vit-large](https://huggingface.co/openai/clip-vit-large-patch14).

## Setup

1. Install dependencies `pip install -r requirements.txt`
2. Download the pre-trained models `python scripts/download-ptms.py`
3. Download the datasets `python scripts/download-datasets.py`

## Training
`python train.py --pretrained_model=PATH_TO_PRETRAINED_MODEL --dataset=PATH_TO_TRAINING_DATA`

## Evaluation
1. Prepare generated images `python infer.py --model-type=MODEL_TYPE`
2. Calculate FID score `python -m pytorch_fid path/to/real_images path/to/generated_images`
3. Calculate CLIP score `python metrics/clip_score.py path/to/generated_images`


## Tips
- Change the proxy variable in the scripts if needed.

## Project Contributions
Research & Report: MaoYiHeng, ZhaoZhiXuan
Experimentation: YuFei, DengJie
