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
model = RetinaFaceModel(config, training = False, iou_th = 0.7, score_th = 0.7)
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

cv.namedWindow("preview")

#Inference
print("Start inferences.")
for i, frame in enumerate(frames):
    start = time.time()
    padded_frame, padding = pad_input_image(frame, 32)
    prediction = model(padded_frame[np.newaxis, ...]).numpy()
    result = recover_pad_output(prediction, padding)
    img = cv.cvtColor(frames[i], cv.COLOR_RGB2BGR)
    boxed_frame = apply_boxes(img, result)
    end = time.time()
    cv.putText(boxed_frame, "fps: %.2f" % (1 / (end - start)), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
    key = cv.waitKey(20)
    if key == 27:
        break
    cv.imshow("image", boxed_frame)  
print("Inference finished.")
