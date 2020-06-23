from __future__ import print_function
import os
from PIL import Image
from math import log, floor

target_height, target_width = 256, 256


def bigIMGtoFolder(img_path, output_folder_path):
    img = Image.open(img_path)
    frame_number = img.size[0] // target_width

    for i in range(frame_number):
        box = (i*target_width, 0, (i+1)*target_width, target_height)
        tmp = img.crop(box)
        name = "0" * (5 - int(floor(log(i+1, 10)))) + str(i)
        try:
            tmp = tmp.save(output_folder_path + name + ".png")
        except:
            print("An error occured in image [" + output_folder_path + "] while cropping slice number [" + str(i) + "]")


def folderConverter(input_folder_path, output_folder_path):
    """ Convert all PNG files representing each a video into sliced PNG in folder, 1 folder per video
        i.e 1 output folder created in [output_folder_path] per input image that is in the [input_folder_path].
        /!\ Use / at the end of the folder names to be safe
    """
    files = [f for f in os.listdir(input_folder_path) if os.path.isfile(input_folder_path + f)]
    n = len(files)
    for j, f in enumerate(files):
        print("Progress: Image " + str(j) + " / " + str(n))
        if not(os.path.exists(output_folder_path + str(j) + "/")):
            os.mkdir(output_folder_path + str(j) + "/")
        bigIMGtoFolder(input_folder_path + f, output_folder_path + str(j) + "/")


folderConverter("E:/Programmation/Python/PAF 2020/first-order-model/checkpoint/animation_test2_226images_over_512vids/png/",
"E:/Programmation/Python/PAF 2020/deepfake2/test/")