# Deepfake detection - PAF 2020

This project was realized as part of the 2020 PAF in Télécom Paris by Arthur Tran-Lambert, Mona Mokart and Vincent Josse, and supervised by Stéphane Safin, computer vision researcher at Télécom Paris. &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ![Télécom Paris](sup-mat/telecom.png)

### What is this repository ?
This repository propose a deeplearning solution to identifiy deepfake videos. We worked as a group of 3 for 2 weeks and came up with an architecture with a very good accuracy (See details below). Our final solution uses deep convolutional & recurrent neural networks in order to classify whether a video was generated by state of the art deep video-generating models, or not.

## First Method: Image sampling over the video
Our first solution consisted in taking random frames from the video and classify them using traditional CNNs. 
The dataset we used was composed of 6000 .png images extracted from real videos from the VoxCeleb dataset as well as fake generated videos using 2019 state of the art first-order-model.
Using Residual Networks, we achieved a validation accuracy of 97%


## Second Method: CNN as input of RNN 
