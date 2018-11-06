# Turtlesim_Hand_Gesture
Control ROS Turtlebot through hand gestures in real time.

You can download the dataset from here:
https://drive.google.com/open?id=1IHphiOmUaBfPuQVrCzUP40lfvzHLFbpb

Requirements:
Python 2.7,
Tensorflow,
Keras,
ROS Kinetic,
openCV.

Steps:
1. Train the model using main.py with your own dataset or the one downloaded from the above link. This way you have model.json and model.h5 files.
2. Attach a webcam.
3. Put the moveInCircle.launch file in thr launch directory of your ros-kinetic package and real_time_ros.py in the src directory.
4. Start roscore and Create an instance of tutlesimnode using rosrun turtlesim turtlesimnode
5. Run the launch file using roslaunch <package-name> moveInCircle.launch
