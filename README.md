# ST-ResNet in Tensorflow 2

A TensorFlow 2 implementation of Deep Spatio-Temporal Residual Networks (ST-ResNet) from the paper ["Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction"](https://arxiv.org/abs/1610.00081). ST-ResNet is an end-to-end deep learning model which uses the unique properties of temporal closeness, period and trend of spatio-temporal data, to forecast the in-flow and out-flow of crowds in a city region. The current code is an implementation of the ST-ResNet architecture, which is adapted from the work by [Sneha Singhania](https://github.com/snehasinghania/STResNet) and the work by [Tolic Wang](https://github.com/TolicWang/DeepST). Singhania used randomly generated data to run through the model, which was written in TensorFlow 1.8 and Python 2.7. Wang obtained the original data and built the model using python 3.6 and tensorflow 1.5. This repository mainly inherit the contribution of Wang and update all the code to tensorflow 2. 

## Model architecture

<p align="center"> 
<img src="assets/st-resnet.png">
</p>

## Prerequisites

* Python 3.10.8
* Tensorflow 2.10.0
* Numpy 1.23.5

## Code Organization

File structure:

* `main.ipynb`: This file contains the main program. The computation graph for ST-ResNet is built, launched in a session and trained here. Hyperparaters are also declared in this section.
* `model.py`: This file contain each component of the STResNet architecture from low level units to the entire graph. 
* `preprocessing`: This directory contains helper functions for data preparation. 
* `preprocessing/timestamp.py`: Deals with different formats of timestamps.
* `preprocessing/MinMaxNormalization`: Contains functions to normalize the data to [-1,1] and inverse normalize the data back to [min,max].
* `preprocessing/BatchGenerator`: The function repeatedly yields batches of the specified size.
* `preprocessing/STMatrix.py`: Data cleaning & create all available (X,y) pairs to match the input of the model.
* `preprocessing/TaxiBJ`: Loading data, and perform normalization and run STMatrix.py, merging external variables, and save the processed data.

## Usage
#### TaxiBJ Data

  Data can be downloaded [here](https://pan.baidu.com/s/1Yce39YkfL7D9ltmmpDZBYg). Password: 8u38

  - BJ16_M32x32_T30_InOut.h5
  - BJ15_M32x32_T30_InOut.h5
  - BJ14_M32x32_T30_InOut.h5
  - BJ13_M32x32_T30_InOut.h5
  - BJ_Meteorology.h5
  - BJ_Holiday.txt
  - ReadMe.md 
  
  Download the above data and store them in the DATAPATH stated in the beginning of TaxiBJ.py
  
  For details of the data information, please refer to the README.md in the data directory.

## Reference

Zhang, Junbo, Yu Zheng, and Dekang Qi. "Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction." AAAI. 2017. [https://arxiv.org/pdf/1610.00081.pdf](https://arxiv.org/pdf/1610.00081.pdf)
