import os
from typing import Dict, Tuple

import PIL
import torchvision
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random


# Image splitter
def split_horizontally(image: Image) -> Tuple[Image, Image]:
    width, height = image.size
    # PIL.Image.crop(box=None): The box is a 4-tuple defining the (left, upper), (right, lower) pixel coordinate
    return image.crop((width / 2, 0, width, height)), image.crop((0, 0, width / 2, height))

def preprocess(image: Image, outputImage):
    image.resize((286, 286), PIL.Image.BICUBIC)
    x = random.randint(0, 286 - 256)
    y = random.randint(0, 286 - 256)
    image.crop((x, y, x + 256, y + 256))

    if random.random() > 0.5:
        image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        outputImage.transpose(PIL.Image.FLIP_LEFT_RIGHT)

def make_dir(path: os.path):
    if not os.path.isdir(path):
        os.makedirs(path)


def split_and_save(path: os.path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    make_dir(os.path.join(path, 'A'))
    make_dir(os.path.join(path, 'B'))

    for i, f in enumerate(files):
        image = PIL.Image.open(os.path.join(path, f))

        a_image, b_image = split_horizontally(image)
        a_image.save(os.path.join(path, 'A', 'A_' + f), 'JPEG')
        b_image.save(os.path.join(path, 'B', 'B_' + f), 'JPEG')


# Custom data set to handle ab images
class DataSet(Dataset):

    def __init__(self, path_to_data: os.path):
        self._paths_to_pictures = [os.path.join(path_to_data, file_name) for file_name in os.listdir(path_to_data)
                                   if os.path.isfile(os.path.join(path_to_data, file_name))]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        path_to_picture = self._paths_to_pictures[index]

        ab_image = PIL.Image.open(path_to_picture).convert('RGB')
        a_image, b_image = split_horizontally(ab_image)
        preprocess(a_image, b_image)

        a_image = torchvision.transforms.ToTensor()(a_image)
        b_image = torchvision.transforms.ToTensor()(b_image)

        return {'A': a_image, 'B': b_image}

    def __len__(self) -> int:
        return len(self._paths_to_pictures)


def get_data_loader(path_to_data: os.path, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset=DataSet(path_to_data),
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle
    )

# val_dir = os.path.join(os.getcwd(), 'dataset', 'cityscapes', 'val')
# split_and_save(val_dir)