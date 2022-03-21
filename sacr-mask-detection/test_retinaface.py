#!/usr/bin/python

import cv2 as cv
import tensorflow as tf
import numpy as np
import yaml
import time

from tensorflow.keras.models import load_model
from utilities import average_precision, iou, get_video_frames, pad_input_image, recover_pad_output

H = 562
W = 1000
TARGET_SIZE = (W, H)

#Load face detection model: SSD-MobileNetV2
print("Load face detection model")
model = load_model("models/retinaface-model")
print("Face detection model loaded.")

#Extract video frames
print("Extracting frames.")
frames = get_video_frames("video/room6-short.mp4", target_size = TARGET_SIZE)[180:330]
print("Frames: ", len(frames))

precisions = []
recalls = []
average_precisions = []
true_positives = []

#Start inference on video frames
for i, frame in enumerate(frames):
    #Start
    start = time.time()
    
    #Preprocess frame
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #frame = cv.convertScaleAbs(frame, alpha=2, beta=50)
    padded_frame, padding = pad_input_image(frame, 32)
    
    #Face inference
    prediction = model(padded_frame[np.newaxis, ...]).numpy()
    result = recover_pad_output(prediction, padding)

    pred_bboxes = []
    label_bboxes = []

    for bbox in result:
        x1 = int(bbox[0] * W)
        y1 = int(bbox[1] * H)
        x2 = int(bbox[2] * W)
        y2 = int(bbox[3] * H)
        pred_bboxes.append((x1,y1, x2, y2))
    
    for j in range(0,6):
        with open("labels/"+str(i)+"-"+str(j)+".txt", "r") as f:
            pts = f.readlines()
            p1 = [int(x) for x in pts[0].replace("[", "").replace("]", "").split(" ")]
            p2 = [int(x) for x in pts[1].replace("[", "").replace("]", "").split(" ")]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            label_bboxes.append((x1,y1, x2,y2))

    #faces = [frame[bbox[1]:bbox[3], bbox[0]:bbox[2]] for bbox in pred_bboxes]

    tps = []
    precision = []
    recall = []
    for rank, pred_bbox in enumerate(pred_bboxes):
        tp = False
        for label_bbox in label_bboxes:
            if iou(label_bbox, pred_bbox) > 0.2:
                tp = True

        if tp:
            tps.append(tp)
            true_positives.append(pred_bbox)

        precision.append(len(tps)/(rank+1))
        recall.append(len(tps)/6)

    ap = average_precision(precision, recall)
    precisions.append(precision)
    recalls.append(recall)
    average_precisions.append(ap)

    #Finish
    finish = time.time()
    print("Elapsed "+str(finish-start)+" for frame "+str(i))
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Average Precision:", ap)
    print()
    key = cv.waitKey(20)
    if key == 27:
        break
