import os
import sys
from datetime import datetime

import torch
import torchvision
from torch.utils.data import DataLoader

import argparse

import DataGenerator
from Pix2PixOptimizer import Pix2PixOptimizer

parser = argparse.ArgumentParser()
parser.add_argument("--use_cuda", type=bool, default=False, help="Set to true for GPU usage")
parser.add_argument("--use_GAN", type=bool, default=True, help="Set to true to use GAN")
parser.add_argument("--is_conditional", type=bool, default=True, help="True for conditional GAN")
parser.add_argument("--use_L1", type=bool, default=True, help="True to use L1")
parser.add_argument("--restart", type=bool, default=False, help="True if you want to restart from a given model")
parser.add_argument("--model_path", type=str, help="Set name of folder in which gNet and dNet are saved")
parser.add_argument("--generate", type=str, default=False, help="Generate validation images from given model")
parser.add_argument("--epoch", type=int, default=200, help="Epoch number of net to generate ima")
args = parser.parse_args()

def save_network(model: Pix2PixOptimizer, epoch: int, save_path: os.path):
    new_Gnet_path = os.path.join(save_path, str(epoch + 1) + '_Gnet')
    new_Dnet_path = os.path.join(save_path, str(epoch + 1) + '_Dnet')
    new_Gopt_path = os.path.join(save_path, str(epoch + 1) + '_Gopt')
    new_Dopt_path = os.path.join(save_path, str(epoch + 1) + '_Dopt')

    torch.save(model.generator.state_dict(), new_Gnet_path)
    torch.save(model.discriminator.state_dict(), new_Dnet_path)
    torch.save(model.generator_optimizer.state_dict(), new_Gopt_path)
    torch.save(model.discriminator_optimizer.state_dict(), new_Dopt_path)


def is_valid_stored_model(path: os.path, epoch: int):
    return os.path.isfile(os.path.join(path, str(epoch) + '_Dnet')) \
           and os.path.isfile(os.path.join(path, str(epoch) + '_Gnet')) \
           and os.path.isfile(os.path.join(path, str(epoch) + '_Gopt')) \
           and os.path.isfile(os.path.join(path, str(epoch) + '_Dopt'))


def get_trained_pix2pix(path_to_Gnet: os.path,
                        path_to_Dnet: os.path,
                        path_to_Gopt: os.path,
                        path_to_Dopt: os.path,
                        use_cuda: bool,
                        use_Gan: bool,
                        is_conditional: bool,
                        has_L1: bool):
    if path_to_Dnet is not None:
        is_train = True
    else:
        is_train = False

    model = Pix2PixOptimizer(is_train=is_train, use_GAN=use_Gan, is_conditional=is_conditional, has_L1=has_L1,
                             use_cuda=use_cuda)
    model.generator.load_state_dict(torch.load(path_to_Gnet))
    if is_train:
        model.generator_optimizer.load_state_dict(torch.load(path_to_Gopt))
    model.generator.eval()

    if path_to_Dnet is not None and is_train:
        model.discriminator.load_state_dict(torch.load(path_to_Dnet))
        model.discriminator_optimizer.load_state_dict(torch.load(path_to_Dopt))
        model.discriminator.eval()

    return model


def save_image(epoch: int, image, data):
    torchvision.utils.save_image(data['A'], os.path.join(os.getcwd(), 'training', str(epoch + 1) + '_input.png'))
    torchvision.utils.save_image(data['B'], os.path.join(os.getcwd(), 'training', str(epoch + 1) + '_real.png'))
    torchvision.utils.save_image(image, os.path.join(os.getcwd(), 'training', str(epoch + 1) + '_output.png'))


def print_avg_losses(epoch: int, no_epochs: int, cumulative_generator_loss: float, cumulative_discriminator_loss: float,
                     no_images: int):
    avg_generator_loss_str = str(format(cumulative_generator_loss / no_images, '8.5f'))
    avg_discriminator_loss_str = str(format(cumulative_discriminator_loss / no_images, '8.5f'))

    sys.stdout.write('\r')
    sys.stdout.write('Epoch ' + format(str(epoch + 1), '3') + ' of ' + str(no_epochs) + " - Avg generator Loss: " + str(
        avg_generator_loss_str) + '\tAvg discriminator loss: ' + str(avg_discriminator_loss_str))
    sys.stdout.write('\n')
    sys.stdout.flush()


def train(model: Pix2PixOptimizer, data_loader: DataLoader, no_epochs: int, save_path: os.path, start: int,
          debug: bool = False):
    no_images = len(data_loader)

    for i in range(start - 1, no_epochs):
        percentage = -1

        cumulative_generator_loss = 0.0
        cumulative_discriminator_loss = 0.0

        for j, data in enumerate(data_loader):
            if debug and j == 5:
                break
            else:
                generator_loss, discriminator_loss, fake_B = model.optimize(data)

                cumulative_generator_loss += generator_loss
                cumulative_discriminator_loss += discriminator_loss

                new_percentage = int(((j + 1) / no_images) * 50)

                if new_percentage > percentage:
                    percentage = new_percentage
                    sys.stdout.write('\r')
                    sys.stdout.write(
                        'Epoch ' + format(str(i + 1), '3') + ' of ' + str(no_epochs) + " - [%-50s]  %d%%" % (
                            '=' * percentage, 2 * percentage))
                    sys.stdout.flush()

                if j == (no_images - 1):
                    save_image(i, fake_B, data)

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
                    Gnet_path = os.path.join(path, str(epoch) + '_Gnet')
                    Dnet_path = os.path.join(path, str(epoch) + '_Dnet')
                    Gopt_path = os.path.join(path, str(epoch) + '_Gopt')
                    Dopt_path = os.path.join(path, str(epoch) + '_Dopt')

                    model = get_trained_pix2pix(Gnet_path, Dnet_path, Gopt_path, Dopt_path, use_cuda=use_cuda,
                                                use_Gan=use_Gan, is_conditional=is_conditional, has_L1=has_L1)

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
    Gnet_path = os.path.join(os.getcwd(), 'storage', date_time, str(epoch) + '_Gnet')

    storage_path = os.path.join(os.getcwd(), 'results', dataset, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(storage_path)

    if os.path.isfile(Gnet_path):
        model = get_trained_pix2pix(path_to_Gnet=Gnet_path,
                                    path_to_Dnet=None,
                                    path_to_Gopt=None,
                                    path_to_Dopt=None,
                                    use_cuda=use_cuda,
                                    use_Gan=True,
                                    is_conditional=True,
                                    has_L1=True)
        data_loader = DataGenerator.get_data_loader(os.path.join(os.getcwd(), 'dataset', dataset, 'val'), 1,
                                                    shuffle=False)

        for j, data in enumerate(data_loader):
            if j < no_images:
                print("Generated picture " + str(j + 1))
                image = model.generate(data)
                torchvision.utils.save_image(image, os.path.join(storage_path, str(j + 1) + '_generated.png'))
            else:
                break
    else:
        print("Unable to generate, model not present.")


if __name__ == '__main__':
    training_images_path = os.path.join(os.getcwd(), 'training')
    if not os.path.exists(training_images_path):
        os.makedirs(training_images_path)

    use_cuda = args.use_cuda
    use_gan = args.use_GAN
    is_conditional = args.is_conditional
    use_l1 = args.use_L1
    restart = args.restart
    model_path = args.model_path

    # Train a model
    if not restart and not generate:
        start_new_training(use_cuda=use_cuda, use_GAN=use_gan, is_conditional=is_conditional, has_L1=use_l1, no_epochs=200)

    # Restart a model
    if restart and not model_path is None:
        restart_training(date_time=model_path, no_epochs=200, use_cuda=use_cuda, use_Gan=use_gan, is_conditional=is_conditional, has_L1=use_l1)
    elif restart and model_path is None:
        print("Specify a path to the model!")

    # Look at content produced by model
    if generate and not model_path is None:
        generate('cityscapes', model_path, args.epoch, use_cuda=use_cuda)
    elif generate and model_path is None:
        print("Specify a path to the model")
