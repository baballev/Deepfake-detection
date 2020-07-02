from __future__ import print_function
import os
import shutil as sh

dataset_video_path = 'E:/Programmation/Python/PAF 2020/first-order-model/data/vox-png/test/'
corrupted_dump = 'E:/Programmation/Python/PAF 2020/first-order-model/data/vox-png/corrupted/'

def isFileCorrupted(path):

    return (os.stat(path).st_size <= 30) # If file size is inferior to 30 bytes, return true

folders = os.listdir(dataset_video_path)
n = len(folders)
corrupted_number = 0
for i, f in enumerate(folders):
    files = os.listdir(dataset_video_path + f)
    corrupted = False
    # Basically: If one image is corrupted, move the complete video out of the folder to the corrupted dump
    for file in files:
        if (isFileCorrupted(dataset_video_path + f + "/" + file)):
            corrupted = True
    if corrupted:
        sh.move(dataset_video_path + f , corrupted_dump + f)
        corrupted_number += 1


    if i % 100 == 99:
        print("Videos : " + str(i) + " / " + str(n))

print("Corrupted files pruning finished.")
print("Found [" + str(corrupted_number) + "] corrupted videos. ")