from Pix2PixOptimizer import Pix2PixOptimizer





if __name__ == '__main__':

    use_GAN = True
    is_conditional = True
    has_L1 = True

    model = Pix2PixOptimizer()
    # images_dataset = get_images()
    #
    # for images in images_dataset:
    #     yield images