from __future__ import print_function
import os
from PIL import Image
import shutil as sh
import numpy as np

number = 2600
path_to_video_folders = 'E:/Programmation/Python/PAF 2020/first-order-model/data/vox-png/test/'
path_to_dataset = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/v2/valid/true/'

folders = os.listdir(path_to_video_folders)
n = len(folders)
permutation = np.random.permutation(range(n))

for i in range(number):
    folder = folders[permutation[i]]
    sh.copytree(path_to_video_folders + folder, path_to_dataset + str(i))
    print(str(i) + " / " + str(number))

print("Finished copy")