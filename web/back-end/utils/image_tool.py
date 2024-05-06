import cv2
import torch
import random
from torchvision import transforms


def random_image_resize(image: torch.Tensor, low=0.25, high=3):
    assert (len(image.shape) == 4 and image.shape[0] == 1)
    # assert image.requires_grad == True
    scale = random.random()
    shape = image.shape
    h, w = shape[-2], shape[-1]
    h, w = int(h * (scale * (high - low) + low)), int(w * (scale * (high - low) + low))
    image = transforms.Resize([h, w])(image)
    return image


def repeat_fill(patch: torch.Tensor, h_real, w_real) -> torch.Tensor:
    patch_h, patch_w = patch.shape[-2:]
    h_num = h_real // patch_h + 1
    w_num = w_real // patch_w + 1
    patch = patch.repeat(1, h_num, w_num)
    patch = patch[:, :h_real, :w_real]
    return patch


def image_to_tensor(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return transforms.ToTensor()(image)
