import os
import sys

# os.environ['CUDA_VISIBLE_DEVICE'] = '0'

import cv2
import json
import glob
import io
import argparse
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm

#from keras_ocr_master.keras_ocr import *
from ocr_model import *
from ocr_model_utils import *

DATA_PATH = 'data'
TRAIN_DATA_FOLDER = 'data/20200319_ocr_train_data/'

IMAGE_SIZE = 512
TF_RECORD_NAME= "0319_train_data_A"
RANGE = 1

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

def image_example(image_string, label, size):
  # image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
      'size': _int64_feature(size),
      'image': _bytes_feature(image_string),
      'label': _float_array_feature(label)
  }

  return tf.train.Example(features=tf.train.Features(feature=feature))

def get_ground_truth(train_data_count, file_name):
    annotation_list = glob.glob(TRAIN_DATA_FOLDER + "train_label/*.json")

    if train_data_count == 0:
        train_data_count = len(annotation_list)
    
    writer = tf.io.TFRecordWriter("tfrecord_data/" + file_name)
    for a in tqdm(range((RANGE - 1) * train_data_count, train_data_count * RANGE)):
        with open(annotation_list[a]) as json_file:
            annotation = json.load(json_file)
            
        img = cv2.imread(TRAIN_DATA_FOLDER + "train_image/" + annotation["images"][0]["file_name"])
        #ran_size = random.randint(224, IMAGE_SIZE)
        #if ran_size % 2 != 0: ran_size -= 1

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        ran_size = IMAGE_SIZE
        #img = cv2.resize(img, (ran_size, ran_size))
        img = compute_input(read_tool(img))

        cv2.imwrite("ghost" + TF_RECORD_NAME + ".jpg", img)

        image_string = cv2.imread("ghost" + TF_RECORD_NAME + ".jpg").tostring()
        
        y = compute_maps(ran_size, ran_size, annotation)
        label_array = np.array(y)
        label_array = np.reshape(label_array, int(ran_size / 2) * int(ran_size / 2) * 2)

        tf_example = image_example(image_string, label_array, ran_size)
        writer.write(tf_example.SerializeToString())

    os.remove("ghost" + TF_RECORD_NAME + ".jpg")

def parse_image_function(example_proto):
    # Create a dictionary describing the features.
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.VarLenFeature(tf.float32),
        'size' : tf.io.FixedLenFeature([], tf.int64)
    }
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

def load_tf_record():
    images = []
    labels = []
    
    tf_record_list = glob.glob(os.getcwd() + "/tfrecord_data/*.tfrecord")

    for tf_record in tf_record_list:
        f_name = tf_record.split('/')
        tf_name = f_name[len(f_name) - 1]
        tf_name = tf_record
        raw_image_dataset = tf.data.TFRecordDataset(tf_name)
        parsed_image_dataset = raw_image_dataset.map(parse_image_function)

        for image_features in tqdm(parsed_image_dataset):

            size = tf.cast(image_features['size'], tf.int64)
            image = tf.io.decode_raw(image_features['image'], tf.uint8)
            image = tf.reshape(image, [size, size, 3])
        
            label = tf.sparse.to_dense(image_features['label'])
            label = tf.reshape(label, [int(size / 2), int(size / 2), 2])

            image = np.array(image)
            label = np.array(label)

            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)

def train_predict(tp_flag):
    if tp_flag is True:
        train_x, train_y = load_tf_record()

        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        with mirrored_strategy.scope():
            model = build_keras_model()
            model.summary()
            model.compile(loss='mse', optimizer='adam')

            # checkpoint_path = "checkpoint/cp-{epoch:04d}.ckpt"
            # checkpoint_dir = os.path.dirname(checkpoint_path)
            # cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=10)
            # model.save_weights(checkpoint_path.format(epoch=0))
            # model.fit(train_x, train_y, epochs=20, batch_size=10, shuffle=True, callbacks = [cp_callback])
            # model.load_weights("result_weights/result0319.h5")
            model.fit(train_x, train_y, epochs=100, batch_size=10, shuffle=True)
            model.save("result_weights/result0319_10000.h5")
    else:
        model = build_keras_model()
        model.summary()
        # model.compile(loss='mse', optimizer='adam')
        model.load_weights("result_weights/result0319_10000.h5")
        images = []

        img1 = cv2.imread("test_data/test1.jpg")
        img2 = cv2.imread("test_data/test2.jpg")
        img3 = cv2.imread("test_data/test3.jpg")
        img4 = cv2.imread("test_data/test4.jpg")
        img5 = cv2.imread("test_data/test5.jpg")
        img1 = cv2.resize(img1, (IMAGE_SIZE,IMAGE_SIZE))
        img2 = cv2.resize(img2, (IMAGE_SIZE,IMAGE_SIZE))
        img3 = cv2.resize(img3, (IMAGE_SIZE,IMAGE_SIZE))
        img4 = cv2.resize(img4, (IMAGE_SIZE,IMAGE_SIZE))
        img5 = cv2.resize(img5, (IMAGE_SIZE,IMAGE_SIZE))

        images.append(img1)
        images.append(img2)
        images.append(img3)
        images.append(img4)
        images.append(img5)

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

        c = 1
        for box in boxes:
            b_len = len(box)
            for b in tqdm(range(b_len)):
                cv2.rectangle(images[c - 1], tuple(box[b][0]), tuple(box[b][2]), (0, 0, 255))
            cv2.imwrite("test/test_result" + str(c) + ".jpg", images[c - 1])
            c += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train")

    args = parser.parse_args()

    train_flag = args.train

    if train_flag == "train":
        train_predict(True)
    elif train_flag == "test":
        train_predict(False)
    else:
        #get_ground_truth(1000, TF_RECORD_NAME + ".tfrecord")
        print("tfrecord")

if __name__ == "__main__":
    main()
