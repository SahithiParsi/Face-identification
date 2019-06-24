# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:22:06 2018

@author: sahithi
"""

# coding: utf-8

# Importing Libraries


import cv2
import os
#os.chdir("C:/Users/sahithi/Desktop/Openpose/openpose-1.4.0-win64-cpu-binaries")

# # Loading the network 

# Specify the paths for the 2 files
protoFile = "coco/pose_deploy_linevec.prototxt"
weightsFile = "coco/pose_iter_440000.caffemodel"
 
# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#  Reading input from webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    cv2.imwrite("test.PNG", frame)
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.01
    
    # Specify the input image dimensions
    inWidth = 368
    inHeight = 368
    
    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
     
    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)
    out = net.forward()
    print(out.shape)
 
    H = out.shape[2]
    W = out.shape[3]
    
    probMap = out[0, 0, :, :]
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    print(point)
    
    # Empty list to store the detected keypoints
    points = []
    for i in range(0,18):
        # confidence map of corresponding body's part.
        probMap = out[0, i, :, :]
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        points.append((int(x), int(y),prob,i))
    
    print(points[17])    
    
    if 0.3 < points[0][2] < 0.75 :    
        cv2.putText(frame,"Standing",(points[1][0],points[1][1]),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),1,lineType=cv2.LINE_AA)
    elif points[0][2] < 0.3 :
        cv2.putText(frame,"Bending",(points[1][0],points[1][1]),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),1,lineType=cv2.LINE_AA)
    #elif  points[0][2] < 0.09 :
      #  cv2.putText(frame,"Looking Sideways",(points[1][0],points[1][1]),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),1,lineType=cv2.LINE_AA)
		##elif points[0][2] < 0.09 :
        #cv2.putText(frame,"Looking Sideways",(points[1][0],points[1][1]),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),1,lineType=cv2.LINE_AA)
    else :    
        cv2.putText(frame,"Sitting",(points[1][0],points[1][1]),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),1,lineType=cv2.LINE_AA)
    cv2.imshow('Output',frame)
    #out.write(im)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
#out.release()
cv2.destroyAllWindows()
