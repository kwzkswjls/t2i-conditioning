import torch
import cv2
from PIL import Image

def condition_canny(images, low_threshold=100, high_threshold=200):
    # condition = torch.zeros((images.shape[0], images.shape[2], images.shape[3]), dtype=torch.uint8)
    # for i in range(images.shape[0]):
    #     canny = images[i].permute(1, 2, 0).numpy()
    #     canny = cv2.Canny(canny, low_threshold, high_threshold)
    #     condition[i] = torch.tensor(canny)
    condition = []
    for image in images:
        canny = image.permute(1, 2, 0).numpy()
        canny = cv2.Canny(canny, low_threshold, high_threshold)
        condition.append(Image.fromarray(canny))
    return condition


if __name__ == "__main__":
    from torchvision.transforms.functional import pil_to_tensor
    from PIL import Image
    from diffusers.utils import load_image

    raw_img = load_image("../data/images/hf-logo.png")
    raw_img.show()

    raw_tensor = pil_to_tensor(raw_img).unsqueeze(0)
    canny_tensor = condition_canny(raw_tensor)[0]

    canny_img = Image.fromarray(canny_tensor.numpy())
    canny_img.show()

