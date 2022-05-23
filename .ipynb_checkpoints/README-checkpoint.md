# Deep Neural Networks with Outdoor Bridge Image Datasets for Concrete Crack Detection and Quantification


This repository includes the implementation for [Deep Neural Networks with Outdoor Bridge Image Datasets for Concrete Crack Detection and Quantification]() based on the Mask R-CNN architecture with additional crack loss. 


## Abstract

Cracking is one of the most common deficiencies observed in concrete bridges. The location of the crack can be recorded in the inspection report with the visual inspection from the human inspectors. Due to the arbitrary shapes and types of bridge defects, conventional computer vision techniques are challenging to make a generalized model to detect or localize concrete cracks with images taken from various outdoor conditions. In this paper, we demonstrates how various outdoor bridge image datasets can be utilized for deep neural network training for defect detection and quantification. We collected bridge images with ground vehicles and unmanned aerial vehicles to generate full scale bridge maps. The collected images are used for training a deep learning segmentation model for pixel-level crack detection on the generated maps. Detected cracks are also used to measure the crack width and spacing. The experimental results show that the proposed algorithm is feasible for detecting and measuring cracks from concrete bridges in real world compared to the state-of-the-art deep learning segmentation algorithms.

## Overview
![overview](/assets/overview.pdf)

The data collected with the ground vehicles and UAVs are pre-processed for neural network training and full-scale bridge map generation. The detected cracks are then used to generate bridge crack maps and to analyze the crack widths. The crack loss is calculated after skeletonized mask by measuring Euclidean distances between center and boundary pixels. 

![architecture](/assets/arch.pdf)

The quantification was performed with the random concrete bridge cracks and the prediction results was measured compared with the crack guage meter.
![evaluation](/assets/crackmeter.pdf)

## Dataset
Under `/dataset/UAVCRACK` directory, you can find datasets pre-divided for training, validation, and testing. Each data has corresponding 256x256 RGB image and N-dimensional binary Numpy array for indicating crack (1) or non-crack (0). Each dimension exclusively  contains only one instance. 

|       | train |  val  | test  |
| ----- | ----- | ----- | ----- |
| #     | 10393 | 2602  | 2602  |

## Inspection
Under `/project/uavcrack` directory, you can find Jupyter Notebooks for data, network, and trained results respectively.
* [data.ipynb](project/uavcrack/data.ipynb)
* [model.ipynb](project/uavcrack/model.ipynb)
* [evaluation.ipynb](project/uavcrack/evaluation.ipynb)


## Pre-trained Network
You can download the pre-trained network for UAVCRACK dataset [here](https://drive.google.com/file/d/19AYEglv8cAIz6vEos27OxCQScQhiTJiO/view?usp=sharing) and locate under `/checkpoint` directory.


## Reference
This work was referenced from [Mask R-CNN for Object Detection and Segmentation](https://github.com/matterport/Mask_RCNN). You can find step by step installation and instructions for training and inference with various demos from the original repository.