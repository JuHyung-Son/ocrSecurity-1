import os
import sys

os.environ['CUDA_VISIBLE_DEVICE'] = '0'

import cv2
import json
import glob
import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

# from keras_ocr_master.keras_ocr import *
from ocr_model import *
from ocr_model_utils import *

TRAIN_DATA_FOLDER = '../../data/20200311_ocr_train_data/'
GROUND_TRUTH_FOLDER = '../../data/ground_truth/'

IMAGE_SIZE = 512
TF_RECORD_NAME = "20200311_ocr_train_data.tfrecord"

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_array_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def image_example(image_string, label):
#   image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
    #   'height': _int64_feature(image_shape[0]),
    #   'width': _int64_feature(image_shape[1]),
    #   'depth': _int64_feature(image_shape[2]),
      'image_raw': _bytes_feature(image_string),
      'label': _float_array_feature(label)
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

def get_ground_truth(train_data_count, file_name):
    annotation_list = glob.glob(TRAIN_DATA_FOLDER + "train_label/*.json")

    if train_data_count == 0:
        train_data_count = len(annotation_list)
    
    writer = tf.io.TFRecordWriter(file_name)
    for a in tqdm(range(train_data_count)):
        with open(annotation_list[a]) as json_file:
            annotation = json.load(json_file)
            
        img = cv2.imread(TRAIN_DATA_FOLDER + "train_image/" + annotation["images"][0]["file_name"])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = compute_input(read_tool(img))

        cv2.imwrite("im.jpg", x)

        image_string = cv2.imread("im.jpg").tostring()

        y = compute_maps(IMAGE_SIZE, IMAGE_SIZE, annotation)
        label_array = np.array(y)
        label_array = np.reshape(label_array, int(IMAGE_SIZE / 2) * int(IMAGE_SIZE / 2) * 2)

        tf_example = image_example(image_string, label_array)
        writer.write(tf_example.SerializeToString())

    os.remove("im.jpg")

def parse_image_function(example_proto):
    # Create a dictionary describing the features.
    image_feature_description = {
        # 'height': tf.io.FixedLenFeature([], tf.int64),
        # 'width': tf.io.FixedLenFeature([], tf.int64),
        # 'depth': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.VarLenFeature(tf.float32)
    }
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

def load_tf_record(tf_record_name):
    raw_image_dataset = tf.data.TFRecordDataset(tf_record_name)
    parsed_image_dataset = raw_image_dataset.map(parse_image_function)

    images = []
    labels = []
    for image_features in tqdm(parsed_image_dataset):
        image = tf.io.decode_raw(image_features['image_raw'], tf.uint8)
        image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
 
        label = tf.sparse.to_dense(image_features['label'])
        label = tf.reshape(label, [int(IMAGE_SIZE / 2), int(IMAGE_SIZE / 2), 2])

        image = np.array(image)
        label = np.array(label)

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)

def train():
    train_x, train_y = load_tf_record(TF_RECORD_NAME)

    model = build_keras_model()
    model.summary()
    model.compile(loss='mse', optimizer='adam')

    checkpoint_path = "checkpoint/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True, period=10)
    model.save_weights(checkpoint_path.format(epoch=0))

    model.fit(train_x, train_y, epochs=30, batch_size=10, shuffle=True, callbacks = [cp_callback])

    # model.fit(train_x, train_y, epochs=50, batch_size=10, shuffle=True)
    model.save("result_weights/result.h5")

    images = []

    img1 = cv2.imread("test1.jpg")
    img2 = cv2.imread("test4.jpg")
    img3 = cv2.imread("test3.jpg")
    img3 = cv2.resize(img3, (IMAGE_SIZE,IMAGE_SIZE))

    images.append(img1)
    images.append(img2)
    images.append(img3)

    boxes = []
    detection_threshold=0.7
    text_threshold=0.4
    link_threshold=0.4
    size_threshold=10

    t_images = [compute_input(read_tool(image)) for image in images]

    for image in t_images:
        img = tf.cast(image, tf.float32)
        pred = model.predict(img[np.newaxis], batch_size=1)
        boxes.append(
            getBoxes(pred,
                        detection_threshold=detection_threshold,
                        text_threshold=text_threshold,
                        link_threshold=link_threshold,
                        size_threshold=size_threshold)[0])

    c = 0
    for box in boxes:
        b_len = len(box)
        for b in tqdm(range(b_len)):
            cv2.rectangle(images[c], tuple(box[b][0]), tuple(box[b][2]), (0, 0, 255))
        cv2.imwrite("result" + str(c) + ".jpg", images[c])
        c += 1

get_ground_truth(0, TF_RECORD_NAME)
# train()
