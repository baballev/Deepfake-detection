from __future__ import print_function
import os
from PIL import Image
from math import log, floor
import cv2


def getFrame(sec, vidcap, image_folder_path, count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(image_folder_path + str(count) + ".png", image)
    return hasFrames

def getVideo(video_path, image_folder_path):
    vidcap = cv2.VideoCapture(video_path)
    sec = 0
    frameRate = 0.1 #//it will capture image in each 0.1 second
    count=1
    success = getFrame(sec, vidcap, image_folder_path, count)

    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec, vidcap, image_folder_path, count)
    return

def getVideoFolder(video_folder_path, output_folder_path):
    files = [f for f in os.listdir(video_folder_path) if os.path.isfile(video_folder_path + f)]
    n = len(files)
    for j, f in enumerate(files):
        print("Progress: Video " + str(j) + " / " + str(n))
        if not(os.path.exists(output_folder_path + str(j) + "/")):
            os.mkdir(output_folder_path + str(j) + "/")
        getVideo(video_folder_path + f, output_folder_path + str(j) + "/")

getVideoFolder("E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/v1/train/true/", "E:/Programmation/Python/PAF 2020/deepfake2/dataset-paf/v1/train/true/")
