import cv2 as cv
import numpy as np
import yaml
import time

from skimage.feature import hog
from tensorflow.keras.applications import mobilenet_v2 as mobilenet

classes_color = {
    "0": (0, 0, 255),
    "1": (0, 255, 0),
    "2": (255, 0, 0)
}

def load_yaml(load_path):
    with open(load_path, 'r') as f:
        loaded = yaml.load(f, Loader=yaml.Loader)
    return loaded

def pad_input_image(img, max_steps):
    """pad image to suitable shape"""
    img_h, img_w, _ = img.shape

    img_pad_h = 0
    if img_h % max_steps > 0:
        img_pad_h = max_steps - img_h % max_steps

    img_pad_w = 0
    if img_w % max_steps > 0:
        img_pad_w = max_steps - img_w % max_steps

    padd_val = np.mean(img, axis=(0, 1)).astype(np.uint8)
    img = cv.copyMakeBorder(img, 0, img_pad_h, 0, img_pad_w,
                             cv.BORDER_CONSTANT, value=padd_val.tolist())
    pad_params = (img_h, img_w, img_pad_h, img_pad_w)

    return img, pad_params


def recover_pad_output(outputs, pad_params):
    """recover the padded output effect"""
    img_h, img_w, img_pad_h, img_pad_w = pad_params
    recover_xy = np.reshape(outputs[:, :14], [-1, 7, 2]) * \
        [(img_pad_w + img_w) / img_w, (img_pad_h + img_h) / img_h]
    outputs[:, :14] = np.reshape(recover_xy, [-1, 14])

    return outputs

def get_video_frames(video_path, target_size, rotate_right=False):
    vid = cv.VideoCapture(video_path)
    frames = []

    while(vid.isOpened()):
        ret, fr = vid.read()
        if ret == True:
            if target_size is not None:
                fr = cv.resize(fr, target_size, interpolation=cv.INTER_CUBIC)
            if rotate_right:
                fr = cv.rotate(fr, cv.cv2.ROTATE_90_CLOCKWISE)
            frames.append(fr)
        else:
            break

    vid.release()
    return frames

def extract_hog(image, ppc = 16):
    features, _ = hog(image, orientations=8, pixels_per_cell=(ppc,ppc), cells_per_block=(4, 4), block_norm='L2', visualize=True)
    return features

def apply_boxes(img, labels, predictions, margin = 5):
    _img = np.array(img)
    h, w, _ = _img.shape
    for label, pred in zip(labels, predictions):
        x1 = max(0, int(label[0] * w) - margin)
        y1 = max(0, int(label[1] * h) - margin)
        x2 = min(w, int(label[2] * w) + margin)
        y2 = min(h, int(label[3] * h) + margin)
        cv.rectangle(_img, (x1,y1), (x2,y2), classes_color[str(pred)], 2)
    return _img

def get_faces(img, bboxes, margin = 5):
    images = []
    h, w, _ = img.shape
    for bbox in bboxes:
        x1 = max(0, int(bbox[0] * w) - margin)
        y1 = max(0, int(bbox[1] * h) - margin)
        x2 = min(w, int(bbox[2] * w) + margin)
        y2 = min(h, int(bbox[3] * h))
        subimg = img[y1:y2, x1:x2]
        images.append(subimg)
    return images

def save_video(frames, video_path="video/output.mp4"):
    h, w, _ = frames[0].shape
    out = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*"DIVX"), 30, (w, h))
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

def facemask_detect(frame, i, model, clf, clf_type):
    #Start
    start = time.time()
    
    #Preprocess frame
    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #frame = cv.convertScaleAbs(frame, alpha=2, beta=50)
    padded_frame, padding = pad_input_image(frame, 32)
    
    #Face inference
    prediction = model(padded_frame[np.newaxis, ...]).numpy()
    result = recover_pad_output(prediction, padding)
    #img = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    
    #Mask classification
    faces = get_faces(frame, result)
    if len(faces) > 0:
        for j, face in enumerate(faces):
            cv.imwrite("faces2/"+str(i)+str(j)+".png", face)
        if clf_type == "svm":
            gray_faces = [cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in faces]
            resized_faces = [cv.resize(f, (128, 128), interpolation=cv.INTER_CUBIC) for f in gray_faces]
            hog_faces = [extract_hog(f) for f in resized_faces]
            predictions = clf.predict(hog_faces)
        else:
            resized_faces = np.array([cv.resize(f, (224, 224), interpolation=cv.INTER_CUBIC) for f in faces])
            processed_faces = mobilenet.preprocess_input(resized_faces)
            predictions = [np.argmax(r) for r in clf.predict(processed_faces)]
        #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        boxed_frame = apply_boxes(frame, result, predictions)
        _img = np.array(boxed_frame)
        end = time.time() 
        cv.putText(_img, "fps: %.2f" % (1 / (end - start)), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0))
        return _img
    else:
        _img = np.array(frame)
        return _img

def area(rect):
  return (rect[0] - rect[2]) * (rect[1] - rect[3])

def iou(label_box, pred_box):
  pred_area = area(pred_box)
  label_area = area(label_box)

  ix = min(label_box[3], pred_box[3]) - max(label_box[1], pred_box[1])
  iy = min(label_box[2], pred_box[2]) - max(label_box[0], pred_box[0])

  ix = max(ix, 0)
  iy = max(iy, 0)

  intersection = ix * iy

  union_area = pred_area + label_area - intersection
  union_area = max(union_area, np.finfo(float).eps)

  return intersection / union_area

def find_corresponding_bbox(pred_box, label_boxes):
  ious = np.array([iou(label_box, pred_box) for label_box in label_boxes])
  return np.argmax(ious), np.amax(ious)

def compare_boxes(pred_boxes, label_boxes):
  result = [find_corresponding_bbox(pred_box, label_boxes) for pred_box in pred_boxes]
  return result

def average_precision(precision, recall):
    areas = []
    for i in range(0,len(precision)):
        next = i+1
        if next < len(precision):
            areas.append((recall[next]-recall[i])*precision[next])

    return sum(areas)
