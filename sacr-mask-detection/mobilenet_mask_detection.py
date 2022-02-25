import cv2 as cv
import numpy as np
import time
import tensorflow as tf

from tensorflow.keras.models import load_model
from keras.applications import mobilenet_v2 as mobilenet
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
clf = load_model("mobilenet-face-mask-detection-model-2")
print("Classifier loaded.")
#clf = load_model("models/resnet50.h5")
#print("Classifier loaded.")

#Extract video frames
frames = get_video_frames("video/room6-short.mp4", target_size = TARGET_SIZE, inc = 15)[180:]
print("Frames extracted.")
print("Frames: ", len(frames))

cv.namedWindow("preview")
cv.namedWindow("face")

#Inference
boxed_frames = []
print("Start inferences.")
for i, frame in enumerate(frames):
    start = time.time()
    padded_frame, padding = pad_input_image(frame, 32)
    prediction = model(padded_frame[np.newaxis, ...]).numpy()
    result = recover_pad_output(prediction, padding)
    img = cv.cvtColor(frames[i], cv.COLOR_RGB2BGR)
    faces = get_faces(img, result, margin = 5)
    if len(faces) > 0:
        for j, face in enumerate(faces):
            cv.imwrite("faces/"+str(i)+str(j)+".jpg", face)
        resized_faces = np.array([cv.resize(f, (224, 224), interpolation=cv.INTER_CUBIC) for f in faces])
        #resized_faces = np.array([cv.convertScaleAbs(f, alpha=1.5, beta=0) for f in resized_faces])
        processed_faces = mobilenet.preprocess_input(resized_faces)
        predictions = [np.argmax(r) for r in clf.predict(processed_faces)]
        boxed_frame = apply_boxes(img, result, predictions)
        _img = np.array(boxed_frame)
        boxed_frames.append(boxed_frame)
        end = time.time() 
        cv.putText(_img, "fps: %.2f" % (1 / (end - start)), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
    else:
        _img = np.array(frame)
    key = cv.waitKey(20)
    if key == 27:
        break
    cv.imshow("face", cv.resize(faces[0], (224, 224), interpolation=cv.INTER_CUBIC))
    cv.imshow("preview", _img) 
print("Inference finished.")

#Save video
save_video(boxed_frames)
print("Video saved.")
