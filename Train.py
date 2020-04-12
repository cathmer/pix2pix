import os
import sys
from datetime import datetime

import torch
import torchvision
from torch.utils.data import DataLoader

import DataGenerator
from Pix2PixOptimizer import Pix2PixOptimizer


def save_network(model: Pix2PixOptimizer, epoch: int, save_path: os.path):
    new_Gnet_path = os.path.join(save_path, str(epoch + 1) + '_Gnet')
    new_Dnet_path = os.path.join(save_path, str(epoch + 1) + '_Dnet')

    torch.save(model.Gnet.cpu().state_dict(), new_Gnet_path)
    torch.save(model.Dnet.cpu().state_dict(), new_Dnet_path)


def is_valid_stored_model(path: os.path, epoch: int):
    return os.path.isfile(os.path.join(path, str(epoch) + '_Dnet')) and os.path.isfile(
        os.path.join(path, str(epoch) + '_Gnet'))


def get_trained_pix2pix(path_to_Gnet: os.path, path_to_Dnet: os.path, use_cuda: bool, use_Gan: bool,
                        is_conditional: bool, has_L1: bool):
    if path_to_Dnet is not None:
        is_train = True
    else:
        is_train = False

    model = Pix2PixOptimizer(is_train=is_train, use_GAN=use_Gan, is_conditional=is_conditional, has_L1=has_L1,
                             use_cuda=use_cuda)
    model.Gnet.load_state_dict(torch.load(path_to_Gnet))
    model.Gnet.eval()

    if path_to_Dnet is not None:
        model.Dnet.load_state_dict(torch.load(path_to_Dnet))
        model.Dnet.eval()
    return model


def save_image(epoch: int, model: Pix2PixOptimizer, data):
    image = model.forward(return_image=True)
    torchvision.utils.save_image(data['A'], os.path.join(os.getcwd(), 'training', str(epoch + 1) + '_input.png'))
    torchvision.utils.save_image(data['B'], os.path.join(os.getcwd(), 'training', str(epoch + 1) + '_real.png'))
    torchvision.utils.save_image(image, os.path.join(os.getcwd(), 'training', str(epoch + 1) + '_output.png'))


def print_losses(iteration: int, generator_loss: float, discriminator_loss: float):
    gloss_str = str(format(generator_loss, '8.5f'))
    dloss_str = str(format(discriminator_loss, '8.5f'))

    print("Iteration: " + str(iteration + 1) + '\tGenerator loss: ' + gloss_str + '\tDiscriminator loss: ' + dloss_str)


def print_avg_losses(epoch: int, no_epochs: int, cumulative_generator_loss: float, cumulative_discriminator_loss: float,
                     no_images: int):
    avg_generator_loss_str = str(format(cumulative_generator_loss / no_images, '8.5f'))
    avg_discriminator_loss_str = str(format(cumulative_discriminator_loss / no_images, '8.5f'))

    sys.stdout.write('\r')
    sys.stdout.write('Epoch ' + format(str(epoch), '3') + ' of ' + str(no_epochs) + " - Avg generator Loss: " + str(
        avg_generator_loss_str) + '\tAvg discriminator loss: ' + str(avg_discriminator_loss_str))
    sys.stdout.write('\n')
    sys.stdout.flush()


def train(model: Pix2PixOptimizer, data_loader: DataLoader, no_epochs: int, save_path: os.path, start: int,
          debug: bool = False, print_interations: bool = False):
    no_images = len(data_loader)

    for i in range(start - 1, no_epochs):
        percentage = -1

        if print_interations:
            print()
            print('Epoch {} of {}'.format(i + 1, no_epochs))

        cumulative_generator_loss = 0.0
        cumulative_discriminator_loss = 0.0

        for j, data in enumerate(data_loader):
            if debug and j == 100:
                break
            else:
                model.set_input(data)
                gloss, dloss = model.optimize()
                gloss = gloss.item()
                dloss = dloss.item()

                cumulative_generator_loss += gloss
                cumulative_discriminator_loss += dloss

                if not print_interations:
                    new_percentage = int(((j + 1) / no_images) * 50)

                    if new_percentage > percentage:
                        percentage = new_percentage
                        sys.stdout.write('\r')
                        sys.stdout.write(
                            'Epoch ' + format(str(i + 1), '3') + ' of ' + str(no_epochs) + " - [%-50s]  %d%%" % (
                                '=' * percentage, 2 * percentage))
                        sys.stdout.flush()

                if print_interations and (j + 1) % 10 == 0:
                    print_losses(j, gloss, dloss)

                if j == (no_images - 1):
                    save_image(i, model, data)

        print_avg_losses(i, no_epochs, cumulative_generator_loss, cumulative_discriminator_loss, no_images)

        if (i + 1) % 10 == 0:
            save_network(model, i, save_path)


def start_training(model: Pix2PixOptimizer, no_epochs: int, start):
    save_path = os.path.join(os.getcwd(), 'storage', datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_path)
    train_dir = os.path.join(os.getcwd(), 'dataset', 'cityscapes', 'train')
    data_loader = DataGenerator.get_data_loader(train_dir, 1)
    train(model=model, data_loader=data_loader, no_epochs=no_epochs, save_path=save_path, start=start)


def start_new_training(use_cuda: bool, use_GAN: bool, is_conditional: bool, has_L1: bool, no_epochs: int):
    model = Pix2PixOptimizer(is_train=True, use_GAN=use_GAN, is_conditional=is_conditional, has_L1=has_L1,
                             use_cuda=use_cuda)
    start_training(model, no_epochs, start=1)


def restart_training(date_time: str, no_epochs: int, use_cuda: bool, use_Gan: bool, is_conditional: bool, has_L1: bool):
    path = os.path.join(os.getcwd(), 'storage', date_time)

    if os.path.isdir(path):
        epochs = list(set([int(f.split('_')[0]) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]))
        epochs.sort(reverse=True)

        i = 0
        restart = False
        while i < len(epochs) and not restart:
            epoch = epochs[i]

            if epoch < no_epochs:
                if is_valid_stored_model(path, epoch):
                    path_to_Gnet = os.path.join(path, str(epoch) + '_Gnet')
                    path_to_Dnet = os.path.join(path, str(epoch) + '_Dnet')

                    model = get_trained_pix2pix(path_to_Gnet, path_to_Dnet, use_cuda=use_cuda, use_Gan=use_Gan,
                                                is_conditional=is_conditional, has_L1=has_L1)

                    start_training(model, no_epochs, start=epoch + 1)
                    restart = True
                i += 1
            else:
                print("Training is already finished.")
                return
        if not restart:
            print("Unable to restart training, time date dir is empty or lacks valid files.")
    else:
        print("Directory not present.")


def generate(dataset: str, date_time: str, epoch: int, use_cuda: bool, no_images: int = 10):
    path = os.path.join(os.getcwd(), 'storage', date_time, str(epoch) + '_Gnet')
    print(path)

    storage_path = os.path.join(os.getcwd(), 'results', dataset, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(storage_path)

    if os.path.isfile(path):
        model = get_trained_pix2pix(path_to_Gnet=path, path_to_Dnet=None, use_cuda=use_cuda, use_Gan=True,
                                    is_conditional=True, has_L1=True)
        data_loader = DataGenerator.get_data_loader(os.path.join(os.getcwd(), 'dataset', dataset, 'val'), 1,
                                                    shuffle=False)

        for j, data in enumerate(data_loader):
            if j < no_images:
                print("Generated picture " + str(j + 1))
                model.set_input(data)
                image = model.forward(return_image=True)
                torchvision.utils.save_image(image, os.path.join(storage_path, str(j + 1) + '_generated.png'))
            else:
                break
    else:
        print("Unable to generate, model not present.")


if __name__ == '__main__':
    training_images_path = os.path.join(os.getcwd(), 'training')
    if not os.path.exists(training_images_path):
        os.makedirs(training_images_path)
    # Train a model
    start_new_training(use_cuda=True, use_GAN=True, is_conditional=False, has_L1=True, no_epochs=200)

    # Restart a model
    # restart_training(date_time='20200408_102555', no_epochs=200, use_cuda=True, use_Gan=True, is_conditional=True, has_L1=True)

    # Look at content produced by model
    # generate('cityscapes', '20200408_102555', 200, use_cuda=False)
