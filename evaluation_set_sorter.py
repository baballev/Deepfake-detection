from __future__ import print_function
import os
import filecmp
import shutil as sh
import numpy as np

used_for_true_dump = 'E:/Programmation/Python/PAF 2020/first-order-model/data/vox-png/used_for_true-train/'
true_path = 'E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/v2/train/true/'
dataset_path = 'E:/Programmation/Python/PAF 2020/first-order-model/data/vox-png/train/'


folders1 = os.listdir(dataset_path)
folders2 = os.listdir(true_path)

# O(n^2) :/

# Basically: compare if the first image in the dataset exists in the class and if yes, move it to the dump.
# at the end, there will only remain files that has not been used for the train set in the dataset folder.
# Thus, we will be able to make an evaluation set that has different videos from the training set.
count = 0
for i, folder in enumerate(folders1):
    print("Video " + str(i + 1) + " / " + str(len(folders1)))
    for fold in folders2:
        file1 = os.listdir(dataset_path + folder)[0]
        file2 = os.listdir(true_path + fold)[0]
        if filecmp.cmp(dataset_path + folder + "/" + file1, true_path + fold + "/" + file2):
            count += 1
            if count % 100 == 0 :
                print(str(count) + " videos already found and moved to the dump")
            sh.move(dataset_path + folder + "/", used_for_true_dump + folder + "/")
            break

