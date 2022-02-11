import cv2 as cv
import numpy as np
import time
import tensorflow as tf

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
clf = load_model("models/resnet50.h5")
print("Classifier loaded.")

#Extract video frames
frames = get_video_frames("video/room4-sparse.mp4", target_size = TARGET_SIZE, inc = 30)[:500]
print("Frames extracted.")
print("Frames: ", len(frames))

cv.namedWindow("preview")

#Inference
boxed_frames = []
print("Start inferences.")
for i, frame in enumerate(frames):
    start = time.time()
    padded_frame, padding = pad_input_image(frame, 32)
    prediction = model(padded_frame[np.newaxis, ...]).numpy()
    result = recover_pad_output(prediction, padding)
    img = cv.cvtColor(frames[i], cv.COLOR_RGB2BGR)
    faces = get_faces(img, result, margin = 20)
    resized_faces = [cv.resize(f, (224, 224), interpolation=cv.INTER_CUBIC) for f in faces]
    processed_faces = [resnet.preprocess_input(np.expand_dims(f, axis=0)) for f in resized_faces]
    predictions = [1 if clf.predict(f)[0][0] < 0.8 else 0 for f in processed_faces]
    boxed_frame = apply_boxes(img, result, predictions)
    end = time.time()
    _img = np.array(boxed_frame)
    boxed_frames.append(boxed_frame)
    cv.putText(_img, "fps: %.2f" % (1 / (end - start)), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
    key = cv.waitKey(20)
    if key == 27:
        break
    cv.imshow("image", _img)  
print("Inference finished.")

#Save video
save_video(boxed_frames)
print("Video saved.")
