import cv2 as cv
import numpy as np
import yaml

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

def apply_boxes(img, labels):
    _img = np.array(img)
    h, w, _ = _img.shape
    for label in labels:
        x1 = int(label[0] * w)
        y1 = int(label[1] * h)
        x2 = int(label[2] * w)
        y2 = int(label[3] * h)
        cv.rectangle(_img, (x1,y1), (x2,y2), (0, 255, 0), 2)
    return _img

def save_video(frames, video_path="video/output.mp4"):
    h, w, _ = frames[0].shape
    out = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*"DIVX"), 15, (w, h))
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()