# Arrhythmia-detection-using-2D-CNN

# About

An endeavor to create an arrhythmia detection model using 2D CNN

# Datasets
2D ECG Images from Kaggle (https://www.kaggle.com/datasets/erhmrai/ecg-image-data?resource=download)

# Processing & Results
All images were first resized (downscaled) to 224x224 dimensions and then were converted to grayscale. We classified the following signals:
N = Normal beat. This type is used for normal heartbeats.
S = Supraventricular ectopic beat. This type of beat originates from the atria (the upper chambers of the heart) but outside the sinoatrial node (the natural pacemaker of the heart), causing an irregular heartbeat.
V = Ventricular ectopic beat. This beat originates from the ventricles (the lower chambers of the heart) and is considered a premature beat. 
F = Fusion of ventricular and normal beat. A fusion beat occurs when a normal beat and an ectopic beat occur at the same time, causing a hybrid beat that has characteristics of both.
Q = Unclassifiable beat. This label is used for beats that cannot be clearly classified into any of the other categories due to various reasons, such as poor signal quality or atypical patterns.

# Our Minimal 2D CNN model achieved an accuracy of 98%

# My Dataset
My working grayscale & processed dataset can be found on Kaggle: ()

# LICENSE
This project is licensed under the Apache License - see the [LICENSE](https://github.com/GeoLek/Arrhythmia-detection-using-2D-CNN/blob/main/LICENSE) file for details.
