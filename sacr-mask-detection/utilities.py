import cv2 as cv
import numpy as np
import yaml

from skimage.feature import hog

classes_color = {
    "0": (0, 0, 255),
    "1": (0, 255, 0),
    "2": (255, 0, 0)
}

def load_yaml(load_path):
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)
    return loaded

def get_video_frames(video_path, target_size, rotate_right=False, inc = None):
    vid = cv.VideoCapture(video_path)
    frames = []

    while(vid.isOpened()):
        ret, fr = vid.read()
        if ret == True:
            if target_size is not None:
                fr = cv.resize(fr, target_size)
            if rotate_right:
                fr = cv.rotate(fr, cv.cv2.ROTATE_90_CLOCKWISE)
            hsv = cv.cvtColor(fr, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv)
            if inc is not None:
                lim = 255 - inc
                v[v > lim] = 255
                v[v <= lim] += inc
                hsv = cv.merge((h,s,v))
                fr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            fr = cv.cvtColor(fr, cv.COLOR_BGR2RGB)
            frames.append(fr)
        else:
            break

    vid.release()
    return frames

def extract_hog(image, ppc = 16):
    features, _ = hog(image, orientations=8, pixels_per_cell=(ppc,ppc), cells_per_block=(4, 4), block_norm='L2', visualize=True)
    return features

def apply_boxes(img, labels, predictions, margin = 10):
    _img = np.array(img)
    h, w, _ = _img.shape
    for label, pred in zip(labels, predictions):
        x1 = max(0, int(label[0] * w) - margin)
        y1 = max(0, int(label[1] * h) - margin)
        x2 = min(w, int(label[2] * w) + margin)
        y2 = min(h, int(label[3] * h) + margin)
        cv.rectangle(_img, (x1,y1), (x2,y2), classes_color[str(pred)], 2)
    return _img

def get_faces(img, bboxes, margin = 15):
    images = []
    h, w, _ = img.shape
    for bbox in bboxes:
        x1 = max(0, int(bbox[0] * w) - margin)
        y1 = max(0, int(bbox[1] * h) - margin)
        x2 = min(w, int(bbox[2] * w) + margin)
        y2 = min(h, int(bbox[3] * h) + margin)
        subimg = img[y1:y2, x1:x2]
        images.append(subimg)
    return images

def save_video(frames, video_path="video/output.mp4"):
    h, w, _ = frames[0].shape
    out = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*"DIVX"), 15, (w, h))
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()

def tf_init():
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf

    config = tf.ConfigProto()
    # dynamically grow the memory used on the GPU
    config.gpu_options.allow_growth = True
    # to log device placement (on which device the operation ran)
    config.log_device_placement = True
    sess = tf.Session(config=config)
    # set this TensorFlow session as the default session for Keras
    set_session(sess)