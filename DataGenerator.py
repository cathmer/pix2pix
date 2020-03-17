import os
from typing import Dict, Tuple

import PIL
import torchvision
from PIL.Image import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# Image splitter
def split_horizontally(image: Image) -> Tuple[Image, Image]:
    width, height = image.size
    # PIL.Image.crop(box=None): The box is a 4-tuple defining the (left, upper), (right, lower) pixel coordinate
    return image.crop((0, 0, width / 2, height)), image.crop((width / 2, 0, width, height))


# Custom data set to handle ab images
class DataSet(Dataset):

    def __init__(self, path_to_data: os.path):
        self._paths_to_pictures = [os.path.join(path_to_data, file_name) for file_name in os.listdir(path_to_data)]

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        path_to_picture = self._paths_to_pictures[index]

        ab_image = PIL.Image.open(path_to_picture).convert('RGB')
        a_image, b_image = split_horizontally(ab_image)

        a_image = torchvision.transforms.ToTensor()(a_image)
        b_image = torchvision.transforms.ToTensor()(b_image)

        return {'A': a_image, 'B': b_image}

    def __len__(self) -> int:
        return len(self._paths_to_pictures)


def get_data_loader(path_to_data: os.path, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=DataSet(path_to_data),
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

# train_dir = os.path.join(os.getcwd(), 'dataset', 'cityscapes', 'train')
# val_dir = os.path.join(os.getcwd(), 'dataset', 'cityscapes', 'val')
#
# x = get_data_loader(train_dir, 64)
#
