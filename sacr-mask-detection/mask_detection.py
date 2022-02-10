import cv2 as cv
import numpy as np
import time
import tensorflow as tf

from modules.models import RetinaFaceModel
from modules.utils import pad_input_image, recover_pad_output
from utilities import apply_boxes, get_video_frames, save_video, load_yaml

TARGET_SIZE = (1200, 675)

#Build model using SSD-MobileNetV2
config = load_yaml("configs/retinaface_mbv2.yaml")
model = RetinaFaceModel(config, training = False, iou_th = 0.7, score_th = 0.5)
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

#Extract video frames
frames = get_video_frames("video/room4.mp4", target_size = TARGET_SIZE, inc = 30)
print("Frames extracted.")
frames = frames[0:200]
print("Frames: ", len(frames))

#Pad input images to avoid unmatched shape input
paddings = []
padded_frames = []
for i, frame in enumerate(frames):
    padded_frame, padding = pad_input_image(frame, 32)
    padded_frames.append(padded_frame)
    paddings.append(padding)
print("Padding applied.")

#Inference
print("Start inferences.")
times = []
predictions = []
for frame in padded_frames:
    start = time.time()
    prediction = model(frame[np.newaxis, ...]).numpy()
    end = time.time()
    predictions.append(prediction)
    times.append(end-start)
print("Inference finished.")
print("Average inference time: ", sum(times) / len(times))
print("Max inference: ", max(times))

#Recover padding
results = []
for i, prediction in enumerate(predictions):
    result = recover_pad_output(prediction, paddings[i])
    results.append(result)
print("Padding recovered.")

#Apply bounding boxes
print("Apply bounding boxes.")
boxed_frames = []
for i, result in enumerate(results):
    frame = cv.cvtColor(frames[i], cv.COLOR_RGB2BGR)
    boxed_frame = apply_boxes(frame, result)
    boxed_frames.append(boxed_frame)
print("Bounding boxes applied.")

#Save video
print("Save video.")
save_video(boxed_frames)
print("Video saved")