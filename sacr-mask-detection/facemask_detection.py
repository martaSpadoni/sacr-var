#!/usr/bin/python

import cv2 as cv
import tensorflow as tf
import sys
import pickle

from tensorflow.keras.models import load_model
from modules.models import RetinaFaceModel
from utilities import load_yaml, facemask_detect, get_video_frames

TARGET_SIZE = (1000, 562)

video = "webcam" if len(sys.argv) < 3 else sys.argv[1]
classifier = "ssd" if len(sys.argv) < 3 else sys.argv[2]

#Build model using SSD-MobileNetV2
print("Build model using SSD-MobileNetV2")
config = load_yaml("configs/retinaface_mbv2.yaml")
model = RetinaFaceModel(config, training = False, iou_th = 0.5, score_th = 0.7)
print("Model built.")

#Load weights of Wider-Face trained network
print("Load weights of Wider-Face trained network")
checkpoints = "./checkpoints/retinaface_mbv2"
checkpoint = tf.train.Checkpoint(model = model)
if tf.train.latest_checkpoint(checkpoints):
    print("Loading checkpoints...")
    checkpoint.restore(tf.train.latest_checkpoint(checkpoints))
else:
    print("Cannot find checkpoints.")
    exit()
print("Checkpoints loaded.")

#Load face mask classifier
print("Load face mask classifier")
if classifier == "svm":
    with open("models/SVC-model.pkl", 'rb') as file:
        clf = pickle.load(file)    
else:
    clf = load_model("mobilenet_epoch-08_loss-0.0027_val_loss-0.1257.h5")
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
    frames = get_video_frames(video, target_size = TARGET_SIZE, inc = 0)[180:]
    print("Frames: ", len(frames))

    #Start inference on video frames
    for frame in frames:
        img = facemask_detect(frame, model, clf, classifier)
        
        #Finish
        key = cv.waitKey(20)
        if key == 27:
            break

        #Show
        cv.imshow(preview_name, img) 

cv.destroyWindow(preview_name)
print("Stop.")