# traffic-sign
# Classifying traffic sign using keras
## Overview
### This is my first time using keras to build a deep neural networks. This project is for Macau AI challenge. Those data are provide by University Of Macau, therefore i cannot give it out, you may use other dataset. The dataset provided by University Of Macau included images, an excel file with images name and ROI.
## Data processing 
### program will read the ROI from the excel file, transform those data into HSV and cut the ROI out. Because the model need all images have same size, and images size are limit by the ram size, with several thst i choose 40*40 for it. all image will save in a  parameter call imgs, at last it will transform to array.
## Model 
### The model is base on LeNet, with some modify, i cut some layer, since i did not have enough data and not using data generator. I choose SGD instand of ADAM, because using ADAM, i got some bug that i can't fix. i set epoch at 10, step per epoch as 200. Lr is 0.01. One epoch cost 100 mins. overfitting starting at first epoch end.   
