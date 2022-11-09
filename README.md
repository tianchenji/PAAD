# PAAD
This repository stores the Pytorch implementation of PAAD for the following paper:

### Proactive Anomaly Detection for Robot Navigation with Multi-Sensor Fusion

#### [Tianchen Ji](https://tianchenji.github.io/), Arun Narenthiran Sivakumar, [Girish Chowdhary](http://daslab.illinois.edu/), [Katie Driggs-Campbell](https://krdc.web.illinois.edu/)

published in *IEEE Robotics and Automation Letters (RA-L)*, 2022

**PAAD fuses camera, LiDAR, and planned path to predict the probability of future failure for robot navigation.** The code was tested on Ubuntu 20.04 with Python 3.8 and Pytorch 1.8.1.

**[\[paper\]](https://ieeexplore.ieee.org/document/9720937) [\[video\]](https://youtu.be/7jFLdpNEiXM) [\[dataset\]](https://uofi.box.com/s/n1qhun9u7lwgtgeyb6hd0tzxpbyxgpl7)**

## Abstract
Despite the rapid advancement of navigation algorithms, mobile robots often produce anomalous behaviors that can lead to navigation failures. We propose a proactive anomaly detection network (PAAD) for robot navigation in unstructured and uncertain environments. PAAD predicts the probability of future failure based on the planned motions from the predictive controller and the current observation from the perception module. Multi-sensor signals are fused effectively to provide robust anomaly detection in the presence of sensor occlusion as seen in field environments. Our experiments on field robot data demonstrate superior failure identification performance than previous methods, and that our model can capture anomalous behaviors in real-time while maintaining a low false detection rate in cluttered fields.

<img src="/figures/sample_trajectory.png" height="280" /><img src="/figures/sample_lidar.png" height="278" />

## Description of the code
More detailed comments can be found in the code. Here are some general descriptions:
* `nets`: Contains network architectures for PAAD.

* `train.py` and `test.py`: Train and test PAAD on the dataset, respectively.

* `custom_dataset.py`: Loads the dataset from CSV files and image folders.

* `dataset_vis.py`: Visualizes a datapoint in the dataset. The annotated planned trajectory is projected onto the front-view image, and the 2D lidar scan around the robot is plotted.

* `utils.py`: Contains the code for loss functions and metrics for quantitative results.

* `rosnode`: Contains a rosnode which performs proactive anomaly detection in real-time using PAAD.

## Dataset
Both the offline dataset and the rosbags used for real-time test in the paper are available.

Each datapoint in the dataset consists of the following information:
* **image:** a front-view image of size 320 × 240.
* **LiDAR scan:** a range reading of size 1081, which covers a 270° range with 0.25° angular resolution.
* **x coordinates of the planned trajectory in bird's-eye view**
* **y coordinates of the planned trajectory in bird's-eye view**
* **label:** a vector of size 10, which indicates if the robot fails the navigation task in the next 10 time steps.

The reference frame, in which planned trajectories are defined, is as follows:

<p align="center">
  <img src="/figures/reference_frame.png" height="140" />
</p>

Sample datapoints from the dataset are as follows:

<img src="/figures/dataset_failure_1.png" height="135" /> <img src="/figures/dataset_failure_2.png" height="135" /> <img src="/figures/dataset_normal_1.png" height="135" /> <img src="/figures/dataset_normal_2.png" height="135" />

The training set consists of 29292 datapoints and contains 2258 anomalous behaviors collected over 5 days, while the test set consists of 6869 datapoints and contains 689 anomalous behaviors from data colected on 2 additional days.

The 6 rosbags used for the real-time test were collected on additional days and contain all the necessary perception signals for PAAD. The detailed related rostopics can be found in the sample code provided in `rosnode`.

## ROS
A minimal ROS workspace for running the real-time test of PAAD could look like this:
<pre>
workspace_folder/         -- WORKSPACE
  src/
    CMakeLists.txt
    fpn_msgs/             -- catkin package for custom msgs
      CMakeLists.txt
      package.xml
      msg/
        <b>Terrabot.msg</b>
    proactive_ad/         -- catkin package for PAAD
      CMakeLists.txt
      package.xml
      src/
        <b>ad_paad.py</b>
        <b>nets/</b>
          <b>...</b>
</pre>
Please follow ROS [tutorials](http://wiki.ros.org/ROS/Tutorials) to create ROS [packages](http://wiki.ros.org/ROS/Tutorials/CreatingPackage) and [msgs](http://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv).

After the workspace is ready, PAAD can be tested in real time by executing the following steps:
1. run `catkin_make` in the workspace.
2. run `python3 ad_paad.py`
3. play a bag file of your choice.

The RGB images and LiDAR point clouds can be visualized in real time through [rviz](http://wiki.ros.org/rviz/UserGuide).

## Citation
If you find the code or the dataset useful, please cite our paper:
```
@article{ji2022proactive,
  title={Proactive Anomaly Detection for Robot Navigation With Multi-Sensor Fusion},
  author={Ji, Tianchen and Sivakumar, Arun Narenthiran and Chowdhary, Girish and Driggs-Campbell, Katherine},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  volume={7},
  number={2},
  pages={4975-4982},
  doi={10.1109/LRA.2022.3153989}}
```

## Contact
Should you have any questions or comments on the code, please feel free to open an issue or contact the author at tj12@illinois.edu.
