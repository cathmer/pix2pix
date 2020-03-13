import os
from os import listdir
from os.path import isfile, join
from PIL import Image

cur_dir = os.getcwd()
dataset_dir = cur_dir + "\dataset\cityscapes\\train"

if not os.path.exists(dataset_dir + "\A"):
    os.mkdir(dataset_dir + "\A")
    os.mkdir(dataset_dir + "\B")
only_files = [f for f in listdir(dataset_dir) if isfile(join(os.path.join(dataset_dir), f))]

bboxA = (256,0,512,256)
bboxB = (0,0,256,256)

for i, img in enumerate(only_files):
    photo = Image.open(cur_dir + "\dataset\cityscapes\\train\\" + img)
    imgA = photo.crop(bboxA)
    imgB = photo.crop(bboxB)

    imgA.save(dataset_dir + "\A\\" + str(i + 1) + "A.jpg", "JPEG")
    imgB.save(dataset_dir + "\B\\" + str(i + 1) + "B.jpg", "JPEG")





