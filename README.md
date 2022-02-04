# PAAD
This repository stores the Pytorch implementation of PAAD for the following paper:

### Proactive Anomaly Detection for Robot Navigation with Multi-Sensor Fusion

#### [Tianchen Ji](https://tianchenji.github.io/), Arun Narenthiran Sivakumar, [Girish Chowdhary](http://daslab.illinois.edu/), [Katie Driggs-Campbell](https://krdc.web.illinois.edu/)

published in *IEEE Robotics and Automation Letters (RA-L)*, 2022

**PAAD fuses camera, LiDAR, and planned path to predict the probability of future failure for robot navigation.** The code was tested on Ubuntu 20.04 with Python 3.8 and Pytorch 1.8.1.

The paper and the demonstration video will be available soon.

## Abstract
Despite the rapid advancement of navigation algorithms, mobile robots often produce anomalous behaviors that can lead to navigation failures. We propose a proactive anomaly detection network (PAAD) for robot navigation in unstructured and uncertain environments. PAAD predicts the probability of future failure based on the planned motions from the predictive controller and the current observation from the perception module. Multi-sensor signals are fused effectively to provide robust anomaly detection in the presence of sensor occlusion as seen in field environments. Our experiments on field robot data demonstrate superior failure identification performance than previous methods, and that our model can capture anomalous behaviors in real-time while maintaining a low false detection rate in cluttered fields.

<img src="/figures/sample_trajectory.png" height="280" /><img src="/figures/sample_lidar.png" height="278" />

## Dataset
The field robot data and the network weights can be found [here](https://uofi.box.com/s/n1qhun9u7lwgtgeyb6hd0tzxpbyxgpl7).

## Description of the code
More detailed comments can be found in the code. Here are some general descriptions:
* `nets`: Contains network architectures for PAAD.

* `rosnode`: Contains a rosnode which performs proactive anomaly detection in real-time using PAAD.

* `train.py` and `test.py`: Train and test PAAD on the dataset, respectively.

* `custom_dataset.py`: Loads the dataset from CSV files and image folders.

* `dataset_vis.py`: Visualizes a datapoint in the dataset. The annotated planned trajectory is projected onto the front-view image, and the 2D lidar scan around the robot is plotted.

* `utils.py`: Contains the code for loss functions and metrics for quantitative results.

## Citation
To be updated.

## Contact
Should you have any questions or comments on the code, please feel free to open an issue or contact the author at tj12@illinois.edu.
