from torchvision.transforms import ToPILImage

from conditions import condition_canny
from dataset import DatasetLaion
from models.utils import save_images

dataset = DatasetLaion(
    r"../data/datasets/laion400m/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet"
)

count = 0
idx = 0

while count < 101:
    item = dataset[idx]
    if item is None:
        idx += 1
        continue

    image = item["IMAGE"]
    image = image.unsqueeze(0)
    canny_image = condition_canny(image)

    caption = item["TEXT"]

    save_images(canny_image, [caption], "../data/cache/canny")
    save_images([ToPILImage()(item["IMAGE"])], [caption], "../data/cache/raw")
    count += 1
    idx += 1
    print(count)
