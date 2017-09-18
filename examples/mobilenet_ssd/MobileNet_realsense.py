#!/usr/bin/env python

# packages

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np  
import sys,os   
import caffe  
from caffe.model_libs import *
from google.protobuf import text_format
import time
import math
import shutil 
import stat
import subprocess 
import time

caffe.set_device(1)
caffe.set_mode_gpu()

directory = '/home/fla-objdet/code/clCaffe'
caffe_root = directory

net_file=   '{}/examples/mobilenet_ssd/MobileNet_VOC0712_SSD_300x300.prototxt'.format(directory)  
caffe_model='{}/examples/mobilenet_ssd/MobileNet_VOC0712_SSD_300x300.caffemodel'.format(directory)  
test_dir = "images"
os.stat(caffe_model)

if not os.path.exists(caffe_model):
# if not os.path.exists('/home/nick/models/MobileNet/VOC0712/MobileNetSSD_deploy.caffemodel'):
	print("MobileNetSSD_deploy.caffemodel does not exist,")
	print("use merge_bn.py to generate it.")
	exit()

net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


class MobileNet_SSD(object):

	def __init__(self):

		# self.image_sub = rospy.Subscriber('/realsense/r200/color/image_raw', Image, self.image_cb, queue_size = 1)
		self.image_sub = rospy.Subscriber('throttled_msgs', Image, self.image_cb, queue_size = 1)
		self.image_pub = rospy.Publisher('obj_det', Image, queue_size=1)
		self.bridge = CvBridge()
		self.count = 0 
		self.desired_fps = 1
		self.sleep = False

	def image_cb(self, msg):

		# print 'realsense img time:', msg.header.stamp

		try:
			cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
		except CvBridgeError as e:
			print e

		start_time = time.time()
		obj_det = self.detect(cv_image)
		# cv2.imshow('SSD', obj_det)
		
		try:
			ros_img = self.bridge.cv2_to_imgmsg(obj_det, 'bgr8')
			ros_img.header.stamp = rospy.get_rostime()
			self.image_pub.publish(ros_img)
			# print 'detect time:', ros_img.header.stamp
			
		except CvBridgeError as e:
			print e
		# cv2.imshow('Test', cv_image)
		# cv2.waitKey(1)
		end_time = time.time()
		comp_time = end_time - start_time
		
		

		if self.sleep:
			sleep_time = (1./self.desired_fps) - comp_time
			time.sleep(max(0, sleep_time))

		end_time = time.time()
		fps = 1./(end_time - start_time)

		self.count += 1
		if self.count%25 == 0:
			print 'max fps: ', fps
			print 'computation time: ', comp_time
			print 'dif time:', ros_img.header.stamp.secs - msg.header.stamp.secs

	def preprocess(self, src):
	    img = cv2.resize(src, (300,300))
	    img = img - 127.5
	    img = img * 0.007843
	    return img

	def postprocess(self, img, out):   
	    h = img.shape[0]
	    w = img.shape[1]
	    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

	    cls = out['detection_out'][0,0,:,1]
	    conf = out['detection_out'][0,0,:,2]
	    return (box.astype(np.int32), conf, cls)

	def detect(self, imgfile):
	    img = self.preprocess(imgfile)
	    
	    img = img.astype(np.float32)
	    img = img.transpose((2, 0, 1))

	    net.blobs['data'].data[...] = img
	    out = net.forward()
	    box, conf, cls = self.postprocess(imgfile, out)

	    for i in range(len(box)):
	    	p1 = (box[i][0], box[i][1])
	    	p2 = (box[i][2], box[i][3])
	    	cv2.rectangle(imgfile, p1, p2, (0,255,0))
	    	p3 = (max(p1[0], 15), max(p1[1], 15))
	    	title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
	    	cv2.putText(imgfile, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)

	    return imgfile

if __name__=="__main__":
	rospy.init_node('MobileNet_SSD')
	MNSSD = MobileNet_SSD()
	rospy.spin()
