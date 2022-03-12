#!/usr/bin/python

import cv2 as cv
import tensorflow as tf
import sys
import pickle

from tensorflow.keras.models import load_model
from utilities import facemask_detect, get_video_frames

TARGET_SIZE = (1000, 562)

video = "webcam" if len(sys.argv) < 3 else sys.argv[1]
classifier = "ssd" if len(sys.argv) < 3 else sys.argv[2]

#Load face detection model: SSD-MobileNetV2
print("Load face detection model")
model = load_model("models/retinaface-model")
print("Face detection model loaded.")

#Load face mask classifier
print("Load face mask classifier")
if classifier == "svm":
    with open("models/SVC-model.pkl", 'rb') as file:
        clf = pickle.load(file)    
else:
    clf = load_model("models/mobilenet_epoch-19_loss-0.0002_val_loss-0.0057.h5")
print("Classifier loaded.")

#Create preview window
preview_name = "image"
cv.namedWindow(preview_name)

if video == "webcam":
    
    #Start webcam
    print("Starting webcam...")
    vc = cv.VideoCapture(0)
    rval, frame = vc.read() if vc.isOpened() else False, None

    while rval:
        
        #Start inference on webcam frame
        rval, frame = vc.read()
        img = facemask_detect(frame, model, clf, classifier)

        #Finish
        key = cv.waitKey(20)
        if key == 27:
            break

        #Show
        cv.imshow(preview_name, img) 

else:

    #Extract video frames
    print("Extracting frames.")
    frames = get_video_frames(video, target_size = TARGET_SIZE)[180:]
    print("Frames: ", len(frames))

    #Start inference on video frames
    for i, frame in enumerate(frames):
        img = facemask_detect(frame, i, model, clf, classifier)
        
        #Finish
        key = cv.waitKey(20)
        if key == 27:
            break

        #Show
        cv.imshow(preview_name, img) 

cv.destroyWindow(preview_name)
print("Stop.")