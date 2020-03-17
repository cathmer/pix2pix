import os

from torch.utils.data import DataLoader

import DataGenerator
from Pix2PixOptimizer import Pix2PixOptimizer


def train(model: Pix2PixOptimizer, data_loader: DataLoader, no_epochs: int):
    for i in range(no_epochs):
        print('Epoch %s of %s', i + 1, no_epochs)

        for data in data_loader:
            model.set_input(data)
            model.optimize()


if __name__ == '__main__':
    model = Pix2PixOptimizer(is_train=True, use_GAN=True, is_conditional=True, has_L1=True)
    train_dir = os.path.join(os.getcwd(), 'dataset', 'cityscapes', 'train')
    data_loader = DataGenerator.get_data_loader(train_dir, 1)

    train(model=model, data_loader=data_loader, no_epochs=2)
