from __future__ import print_function
import os
from PIL import Image
import shutil as sh
import numpy as np

# Number of desired samples
fake_nb = 600
real_nb = 900

# Number of subfolders available
real_video_nb = 512
fake_video_nb = 227

# Absolute Path for input and output folders, and relative path classes' subfolder
input_path = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/v1/train/'
output_path = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/v0/train/'
real_subfolder_name = "true/"
fake_subfolder_name = "fake/"


np.random.seed(678)

# Choose random video to choose from
rd1 = np.random.randint(real_video_nb, size=real_nb)
rd2 = np.random.randint(fake_video_nb, size=fake_nb)

## Real class
for i, index in enumerate(rd1):
    # List all frames for the picked index'th video
    file_list = os.listdir(input_path + real_subfolder_name  + str(index) + "/")

    # Pick a random frame
    r = int(np.random.randint(len(file_list), size=1))

    # Copy the file from input to destination
    sh.copyfile(input_path + real_subfolder_name + str(index) + "/" + file_list[r],
    output_path + real_subfolder_name + str(i) + ".png")

    # Verbose
    if i%100 == 99:
        print("Real images: " + str(i) + " / " + str(real_nb))

## Fake class
for i, index in enumerate(rd2):

    file_list = os.listdir(input_path + fake_subfolder_name + str(index) + "/")
    r = int(np.random.randint(len(file_list), size=1))
    sh.copyfile(input_path + fake_subfolder_name + str(index) + "/" + file_list[r],
    output_path + "fake_subfolder_name + str(i) + ".png")
    if i%100 == 99:
        print("Fake images: " + str(i) + " / " + str(fake_nb))
