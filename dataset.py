import os

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision.transforms.functional import pil_to_tensor

from models.utils import image_grid

__all__ = ["DatasetLaion", "safe_collate", "DatasetCOCOVal", "img_collate"]

proxy = None


def safe_collate(batch):
    return default_collate([item for item in batch if item is not None])


def img_collate(batch):
    return batch


def resize(img, size=(512, 512)):
    width, height = img.size
    max_dim = max(width, height)
    scale = size[0] / max_dim
    img = img.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", size, "black")
    left = (size[0] - img.width) // 2
    top = (size[1] - img.height) // 2
    new_img.paste(img, (left, top))
    return new_img


class DatasetLaion(Dataset):
    def __init__(self, parquet_path):
        self.dataframe = pd.read_parquet(parquet_path)
        self.dataframe.drop(columns=["LICENSE", "SAMPLE_ID", "NSFW", "similarity", "HEIGHT", "WIDTH"], inplace=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx].to_dict()
        try:
            resp = requests.get(item["URL"], stream=True, proxies=proxy, timeout=3)
            del item["URL"]
        except requests.exceptions.RequestException:
            return None

        if resp.status_code != 200:
            return None

        try:
            image = Image.open(resp.raw)
            image = resize(image)
            item["IMAGE"] = pil_to_tensor(image)
        except UnidentifiedImageError:
            return None

        return item


class DatasetCOCOVal(Dataset[Image]):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_list = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_list[idx])
        image = Image.open(image_path)
        return image


if __name__ == "__main__":
    # Laion-400M
    dataset = DatasetLaion(
        r"data/datasets/laion400m/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet"
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=safe_collate)
    sample_batch = next(iter(dataloader))  # may be none
    print(sample_batch["TEXT"])

    grid = image_grid(sample_batch["IMAGE"][:4], 2, 2, img_type="tensor")
    grid.show()

    # COCO-2014-Validation
    dataset = DatasetCOCOVal(r"data/datasets/coco2014/images/val2014")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=img_collate)
    sample_batch = next(iter(dataloader))

    grid = image_grid(sample_batch, 2, 2)
    grid.show()
