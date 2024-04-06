# Arrhythmia-detection-using-2D-CNN

# About

An endeavor to create an arrhythmia detection model using 2D CNN

# Datasets
2D ECG Images from Kaggle (https://www.kaggle.com/datasets/erhmrai/ecg-image-data?resource=download)

# Processing & Results
All images were first resized (downscaled) to 224x224 dimensions and then were converted to grayscale. We classified the following signals:  <br>
N = Normal beat. This type is used for normal heartbeats. <br>
S = Supraventricular ectopic beat. This type of beat originates from the atria (the upper chambers of the heart) but outside the sinoatrial node (the natural pacemaker of the heart), causing an irregular heartbeat. <br>
V = Ventricular ectopic beat. This beat originates from the ventricles (the lower chambers of the heart) and is considered a premature beat. <br>
F = Fusion of ventricular and normal beat. A fusion beat occurs when a normal beat and an ectopic beat occur at the same time, causing a hybrid beat that has characteristics of both. <br>
Q = Unclassifiable beat. This label is used for beats that cannot be clearly classified into any of the other categories due to various reasons, such as poor signal quality or atypical patterns. <br>

# Accuracy = 98%

# My Dataset
My working grayscale & processed dataset can be found on Kaggle: ()

![Classification report](https://github.com/GeoLek/Arrhythmia-detection-using-2D-CNN/assets/89878177/a66bb1bc-6d7d-45ce-b67a-714afdeef9c2)


![confusion_matrix](https://github.com/GeoLek/Arrhythmia-detection-using-2D-CNN/assets/89878177/fa42cec8-a5b1-4be2-af18-8d03c6b5d6ee)


# LICENSE
This project is licensed under the Apache License - see the [LICENSE](https://github.com/GeoLek/Arrhythmia-detection-using-2D-CNN/blob/main/LICENSE) file for details.
