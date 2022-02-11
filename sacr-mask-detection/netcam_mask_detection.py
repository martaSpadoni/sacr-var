import cv2 as cv
import numpy as np
import time
import tensorflow as tf
import pickle

from tensorflow.keras.models import load_model
from keras.applications import resnet
from modules.models import RetinaFaceModel
from modules.utils import pad_input_image, recover_pad_output
from utilities import apply_boxes, extract_hog, get_faces, get_video_frames, save_video, load_yaml

TARGET_SIZE = (1000, 562)

#Build model using SSD-MobileNetV2
config = load_yaml("configs/retinaface_mbv2.yaml")
model = RetinaFaceModel(config, training = False, iou_th = 0.5, score_th = 0.7)
print("Model built.")

#Load weights of Wider-Face trained network
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
#model found at https://github.com/aome510/Mask-Classifier
clf = load_model("models/resnet50.h5")
print("Classifier loaded.")

#Start webcam
cv.namedWindow("preview")
vc = cv.VideoCapture(0)
rval, frame = vc.read() if vc.isOpened() else False, None

while rval:
    #Start
    start = time.time()
    rval, frame = vc.read()
    
    #Preprocess frame
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    padded_frame, padding = pad_input_image(frame, 32)
    
    #Face inference
    prediction = model(padded_frame[np.newaxis, ...]).numpy()
    result = recover_pad_output(prediction, padding)
    img = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    
    #Face mask inference
    faces = get_faces(img, result)
    resized_faces = [cv.resize(f, (224, 224), interpolation=cv.INTER_CUBIC) for f in faces]
    processed_faces = [resnet.preprocess_input(np.expand_dims(f, axis=0)) for f in resized_faces]
    predictions = [1 if clf.predict(f)[0][0] < 0.8 else 0 for f in processed_faces]
    
    #Bounding boxes
    boxed_frame = apply_boxes(img, result, predictions)

    #Finish
    key = cv.waitKey(20)
    if key == 27:
        break
    end = time.time()

    #Show
    cv.putText(boxed_frame, "fps: %.2f" % (1 / (end - start)), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
    cv.imshow("image", boxed_frame) 

cv.destroyWindow("preview")
print("Stop.")
