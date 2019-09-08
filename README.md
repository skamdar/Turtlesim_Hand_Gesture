# Control ROS Turtle through Hand-Gestures in Real Time
Control ROS Turtlebot through hand gestures in real time. When thumb is up in front of webcam the turtle starts speeding up, for left/right horizontal thumb turtle starts turning in left/right direction and stops as soon as it sees thumbs down. To achieve this, we trained a model to distinguish among different states of the hand and integrated it with ROS to move the turtle accordingly.
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
4. Start roscore and Create an instance of tutlesimnode using: rosrun turtlesim turtlesimnode
5. Run the launch file using: roslaunch \<package-name\> moveInCircle.launch
