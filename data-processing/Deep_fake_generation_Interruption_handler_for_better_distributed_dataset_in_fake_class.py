from __future__ import print_function
import os
from PIL import Image
import shutil as sh
import numpy as np


"""
This script is used in order to remove from the dataset the videos that have already been
used for deepfake generation by moving them to another designated folder. Thus,
once all the video are used, it's possible to continue generating with first-order model
without re-using a video. This leads to a better distribution for the fake class in the final dataset
"""

"""
File name structure:
-------------Driving-------------      --------------Source-------------
id10001#DtdEYdViWdw#001613#001802.mp4-id10903#MsJKfBcIHrI#005322#005464.mp4
"""

fake_png_path = 'E:/Programmation/Python/PAF 2020/first-order-model/checkpoint/animation/png/'
dataset_video_path = 'E:/Programmation/Python/PAF 2020/first-order-model/data/vox-png/train/'
move_to_path = 'E:/Programmation/Python/PAF 2020/first-order-model/data/vox-png/used_for_fake-eval/'

files = os.listdir(fake_png_path)
n = len(files)
for i, f in enumerate(files):

    f1 = f[:37]
    f2 = f[38:-4]
    if os.path.exists(dataset_video_path + f1):
        sh.move(dataset_video_path + f1, move_to_path + f1)

    if os.path.exists(dataset_video_path + f2):
        sh.move(dataset_video_path + f2, move_to_path + f2)

    print("Progress: " + str(i) + " / " + str(n))











