from __future__ import print_function
import os
from PIL import Image
import shutil as sh
import numpy as np

# Number of desired samples
fake_nb = 1000
real_nb = 3000

# Number of subfolders available
real_video_nb = 5777
fake_video_nb = 418

# Absolute Path for input and output folders, and relative path classes' subfolder
input_path = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/v2/train/'
output_path = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/tmp/train/'
real_subfolder_name = "true/"
real_subfolder_name1 = "../../../../first-order-model/data/vox-png/train/"
fake_subfolder_name = "fake/"


np.random.seed(332)

# Choose random video to choose from
rd1 = np.random.randint(real_video_nb, size=real_nb)
rd2 = np.random.randint(fake_video_nb, size=fake_nb)

## Real class
for i, index in enumerate(rd1):
    # List all frames for the picked index'th video
    folder_list = os.listdir(input_path + real_subfolder_name1 + "/")

    file_list = os.listdir(input_path + real_subfolder_name1 + "/" + folder_list[index] + "/")
    # Pick a random frame
    r = int(np.random.randint(len(file_list), size=1))

    # Copy the file from input to destination
    sh.copyfile(input_path + real_subfolder_name1 + folder_list[index] + "/" + file_list[r],
    output_path + real_subfolder_name + str(i) + ".png")

    # Verbose
    if i%100 == 99:
        print("Real images: " + str(i) + " / " + str(real_nb))

print("Real images: Done")

## Fake class
for i, index in enumerate(rd2):

    file_list = os.listdir(input_path + fake_subfolder_name + str(index) + "/")
    r = int(np.random.randint(len(file_list), size=1))
    sh.copyfile(input_path + fake_subfolder_name + str(index) + "/" + file_list[r],
    output_path + fake_subfolder_name + str(i) + ".png")
    if i%100 == 99:
        print("Fake images: " + str(i) + " / " + str(fake_nb))

print("Fake images: Done")
