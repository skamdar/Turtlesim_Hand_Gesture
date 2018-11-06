#!/usr/bin/env python
import rospy
from geometry_msgs.msg  import Twist
from turtlesim.msg import Pose
from math import pow,atan2,sqrt
from std_msgs.msg import String

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os
import PIL
from PIL import Image
from keras import optimizers
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from threading import Thread
from threading import Lock
import time
import cv2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


def readwebcam():
    cap = cv2.VideoCapture(0)
    global frames

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #print(frame.type())
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print(frame.shape)
        # Display the resulting frame
        cv2.namedWindow("frame_webcam")
        cv2.imshow("frame_webcam", frame)
        frame = cv2.resize(frame, (256, 256))
        #frame = np.transpose(frame, (2, 0, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        lock.acquire()
        frames = frame
        lock.release()
        #time.sleep(0.1)
    # When everything done, release the capture
    cap.release()



class turtlebot():

    def __init__(self):
        #Creating our node,publisher and subscriber
        rospy.init_node('turtlebot_controller', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber('/turtle1/pose', Pose, self.callback)
        self.pose = Pose()
        self.rate = rospy.Rate(10)

    #Callback function implementing the pose value received
    def callback(self, data):
        self.pose = data
        self.pose.x = round(self.pose.x, 4)
        self.pose.y = round(self.pose.y, 4)

    def get_distance(self, goal_x, goal_y):
        distance = sqrt(pow((goal_x - self.pose.x), 2) + pow((goal_y - self.pose.y), 2))
        return distance

    def move2goal(self):
        goal_pose = Pose()
	print("move2goal called")
        goal_pose.x = 3 #input("Set your x goal:")
        goal_pose.y = 3 #input("Set your y goal:")
        distance_tolerance = 0.1 #input("Set your tolerance:")
        vel_msg = Twist()

	json_file = open('/home/sonu/Desktop/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("/home/sonu/Desktop/model.h5")
	print("Loaded model from disk")

	loaded_model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

	#lock = Lock()
	frames = np.random.rand(256, 256, 3)
	#t = Thread(target=readwebcam)
	#t.start()
	
	cap = cv2.VideoCapture(0)
	
	while True:
    		#time.sleep(0.1)
		ret, frame = cap.read()

		#cv2.namedWindow("frame_webcam")
        	cv2.imshow("frame_webcam", frame)
        	frame = cv2.resize(frame, (256, 256))
		if cv2.waitKey(1) & 0xFF == ord('q'):
            		break
    		#frame = frames
    		#cv2.imshow('ff', frame)
    		frame = frame.reshape(1, 256, 256, 3) / 255.0 
    		#print("prediction...")
    		result = loaded_model.predict(frame)
    		#print(len(result))
		action = 0
    		if result[0][0] > result[0][1]:
    			print("thumbs down")
			action = 1	
    		else:
    			print("thumbs up")
			action = 0
        	#Testing our function        
	
		if action == 0: #Porportional Controller
            #linear velocity in the x-axis:
            		vel_msg.linear.x = 10
            		vel_msg.linear.y = 0
            		vel_msg.linear.z = 0
	    		print(self.pose.x)
	    		print(self.pose.y)
            #angular velocity in the z-axis:
            		vel_msg.angular.x = 0
            		vel_msg.angular.y = 0
            		vel_msg.angular.z = 4	
		else:
        	#Stopping our robot after the movement is over
        		vel_msg.linear.x = 0
        		vel_msg.angular.z =0
        	self.velocity_publisher.publish(vel_msg)

	vel_msg.linear.x = 0
        vel_msg.angular.z =0
        self.velocity_publisher.publish(vel_msg)
        rospy.spin()
	cap.release()

if __name__ == '__main__':
    try:
        #Testing our function
        x = turtlebot()
        x.move2goal()

    except rospy.ROSInterruptException: pass

