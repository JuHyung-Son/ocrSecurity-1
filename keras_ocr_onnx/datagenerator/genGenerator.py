import tensorflow as tf
import os
import matplotlib.pylab as plt
import cv2
from itertools import cycle
from utils.utils import *
from PIL import Image
# tf.compat.v1.disable_eager_execution()
import numpy as np
import io
import random
import argparse

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_record(images_id,images_dl,target_bg_natural):

    target_bg_natural=target_bg_natural[:max(len(images_dl),len(images_id))*2]

    if max(len(images_dl),len(images_id),len(target_bg_natural))== len(images_dl):
        zip_list=zip(images_dl,cycle(images_id),cycle(target_bg_natural))
    elif max(len(images_dl),len(images_id),len(target_bg_natural))==len(images_id):
        zip_list=zip(cycle(images_dl),images_id,cycle(target_bg_natural))
    else:
        zip_list=zip(cycle(images_dl),cycle(images_id),target_bg_natural)


    with tf.io.TFRecordWriter('images.tfrecords') as writer:
        for dl, id, bg in zip_list:
            img_id = tf.keras.preprocessing.image.load_img(id)
            img_dl = tf.keras.preprocessing.image.load_img(dl)
            img_bg = tf.keras.preprocessing.image.load_img(bg)
            img_id_array = tf.keras.preprocessing.image.img_to_array(img_id)
            img_dl_array = tf.keras.preprocessing.image.img_to_array(img_dl)
            img_bg_array = tf.keras.preprocessing.image.img_to_array(img_bg)

            crop_img_id_array = img_id_array[27:90,90:368,0].astype(np.uint8)
            crop_img_dl_array = img_id_array[10:40, 20:120, 0].astype(np.uint8)
            img_nt_array = img_nt_array[...,0]

            if random.random() > 0.5:
                rand_h_integer = random.randint(90,img_id_array.shape[0]-63)
                rand_w_integer = random.randint(0, img_id_array.shape[1] - 278)
                img_nm_array = img_id_array[rand_h_integer:rand_h_integer+63,rand_w_integer:rand_w_integer+278,0].astype(np.uint8)
            else:
                rand_h_integer = random.randint(40, img_dl_array.shape[0] - 30)
                rand_w_integer = random.randint(0, img_dl_array.shape[1] - 100)
                img_nm_array = img_dl_array[rand_h_integer:rand_h_integer + 63, rand_w_integer:rand_w_integer + 278,0].astype(np.uint8)

            resized_id_array = np.reshape(cv2.resize(crop_img_id_array,(320,320,)),(320,320,1))
            resized_dl_array = np.reshape(cv2.resize(crop_img_dl_array, (320, 320,)), (320, 320, 1))
            resized_nm_array = np.reshape(cv2.resize(img_nm_array, (320, 320,)), (320, 320, 1))
            resized_nt_array = np.reshape(cv2.resize(img_nt_array, (320, 320,)), (320, 320, 1)).astype(np.uint8)

            img_id_bytes = resized_id_array.tostring()
            img_dl_bytes = resized_dl_array.tostring()
            img_nm_bytes = resized_nm_array.tostring()
            img_nt_bytes = resized_nt_array.tostring()
            def image_example(img_id_raw,img_dl_raw,img_nm_raw,img_nt_raw):
                feature={
                    'image/image_id_raw': _bytes_feature(img_id_array),
                    'image/image_dl_raw': _bytes_feature(img_dl_array),
                    'image/image_nm_raw': _bytes_feature(img_nm_array),
                    'image/image_nt_raw': _bytes_feature(img_nt_array),

                    'image/format': _bytes_feature(b'jpg'),

                    'image/id_height': _int64_feature(63),
                    'image/id_width': _int64_feature(278),
                    'image/dl_height': _int64_feature(30),
                    'image/dl_width': _int64_feature(100),
                }

                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                return example_proto.SerializetoString()

            tf_example = image_example(img_id_bytes,img_dl_bytes,img_nm_bytes,img_nm_bytes,img_nt_bytes)
            writer.write(tf_example)


def write_record_binary_cls(images_id,images_dl,target_bg_natural):

    target_bg_natural=target_bg_natural[:max(len(images_dl),len(images_id))*2]

    if max(len(images_dl), len(images_id), len(target_bg_natural)) == len(images_dl):
        zip_list = zip(images_dl, cycle(images_id), cycle(target_bg_natural))
    elif max(len(images_dl), len(images_id), len(target_bg_natural)) == len(images_id):
        zip_list = zip(cycle(images_dl), images_id, cycle(target_bg_natural))
    else:
        zip_list = zip(cycle(images_dl), cycle(images_id), target_bg_natural)

    with tf.io.TFRecordWriter('images.tfrecords') as writer:
        for dl, id, bg in zip_list:
            img_id = tf.keras.preprocessing.image.load_img(id)
            img_dl = tf.keras.preprocessing.image.load_img(dl)
            img_bg = tf.keras.preprocessing.image.load_img(bg)
            img_id_array = tf.keras.preprocessing.image.img_to_array(img_id)
            img_dl_array = tf.keras.preprocessing.image.img_to_array(img_dl)
            img_bg_array = tf.keras.preprocessing.image.img_to_array(img_bg)

            crop_img_id_array = img_id_array[27:90,90:368,0].astype(np.uint8)
            crop_img_dl_array = img_id_array[10:40, 20:120, 0].astype(np.uint8)
            img_nt_array = img_nt_array[...,0]

            if random.random() > 0.5:
                rand_h_integer = random.randint(90,img_id_array.shape[0]-63)
                rand_w_integer = random.randint(0, img_id_array.shape[1] - 278)
                img_nm_array = img_id_array[rand_h_integer:rand_h_integer+63,rand_w_integer:rand_w_integer+278,0].astype(np.uint8)
            else:
                rand_h_integer = random.randint(40, img_dl_array.shape[0] - 30)
                rand_w_integer = random.randint(0, img_dl_array.shape[1] - 100)
                img_nm_array = img_dl_array[rand_h_integer:rand_h_integer + 63, rand_w_integer:rand_w_integer + 278,0].astype(np.uint8)

            resized_id_array = np.reshape(cv2.resize(crop_img_id_array,(320,320,)),(320,320,1))
            resized_dl_array = np.reshape(cv2.resize(crop_img_dl_array, (320, 320,)), (320, 320, 1))
            resized_nm_array = np.reshape(cv2.resize(img_nm_array, (320, 320,)), (320, 320, 1))
            resized_nt_array = np.reshape(cv2.resize(img_nt_array, (320, 320,)), (320, 320, 1)).astype(np.uint8)

            img_id_bytes = resized_id_array.tostring()
            img_dl_bytes = resized_dl_array.tostring()
            img_nm_bytes = resized_nm_array.tostring()
            img_nt_bytes = resized_nt_array.tostring()
            def image_example(img_id_raw,img_dl_raw,img_nm_raw,img_nt_raw):
                feature={
                    'image/image_id_raw': _bytes_feature(img_id_array),
                    'image/image_dl_raw': _bytes_feature(img_dl_array),
                    'image/image_nm_raw': _bytes_feature(img_nm_array),
                    'image/image_nt_raw': _bytes_feature(img_nt_array),

                    'image/format': _bytes_feature(b'jpg'),

                    'image/id_height': _int64_feature(63),
                    'image/id_width': _int64_feature(278),
                    'image/dl_height': _int64_feature(30),
                    'image/dl_width': _int64_feature(100),
                }

                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                return example_proto.SerializetoString()

            def image_example_single_cls(img_raw,label):
                feature={
                    'image/image_id_raw': _bytes_feature(img_raw),
                    'image/format': _bytes_feature(b'jpg'),
                    'image/label': _int64_feature(label),
                    'image/height': _int64_feature(320),
                    'image/width': _int64_feature(320),
                }

                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                return example_proto.SerializetoString()

            seed = random.random()
            if seed <=0.25:
                tf_example = image_example_single_cls(img_id_bytes,label=1)
                writer.write(tf_example)
            elif seed <= 0.5:
                tf_example = image_example_single_cls(img_dl_bytes, label=1)
                writer.write(tf_example)
            elif seed <= 0.75:
                tf_example = image_example_single_cls(img_nm_bytes, label=0)
                writer.write(tf_example)
            else:
                tf_example = image_example_single_cls(img_nt_bytes, label=0)
                writer.write(tf_example)


def apply_aug(image_features,batch_size):
    decode_img_id = tf.reshape(image_features['id'],(batch_size,320,320,1))
    decode_img_dl = tf.reshape(image_features['dl'], (batch_size, 320, 320, 1))
    decode_img_nm = tf.reshape(image_features['nm'], (batch_size, 320, 320, 1))
    decode_img_nt = tf.reshape(image_features['nt'], (batch_size, 320, 320, 1))

    img_id = decode_img_id
    img_dl = decode_img_dl
    img_nm = decode_img_nm
    img_nt = decode_img_nt

    img_id = tf.image.random_flip_up_down(img_id)
    img_dl = tf.image.random_flip_up_down(img_dl)
    img_nm = tf.image.random_flip_up_down(img_nm)
    img_nt = tf.image.random_flip_up_down(img_nt)

    data = dict()
    data['id']=img_id
    data['dl'] = img_dl
    data['nm'] = img_nm
    data['nt'] = img_nt

    return data




def apply_aug_single_cls(image_features,batch_size):
    try:
        decode_img = tf.reshape(image_features['img'],(batch_size,320,320,1))
    except:
        print('fail to decode image')

    decode_label = tf.reshape(image_features['label'],(batch_size,1))

    decode_img = tf.image.random_brightness(decode_img,max_delta=0.1)
    decode_img = tf.image.random_flip_left_right(decode_img)
    decode_img = tf.image.random_flip_up_down(decode_img)


    data = dict()
    data['img']=decode_img
    data['label']=decode_label
    return data


def read_record(imageTFRecord,num_parallel_reads=8,shuffle_buffer_size=8,batch_size=16,epoch=10):
    tfrecordFiles = tf.data.Dataset.list_files(imageTFRecord)
    dataset = tfrecordFiles.interleave(tf.data.TFRecordDataset,cycle_length=num_parallel_reads,num_parallel_calls=tf.data.experimental.AUTOTUNE)


    image_feature_description = {
        'image/image_id_raw':tf.io.FixedLenFeature((),tf.string),
        'image/image_dl_raw':tf.io.FixedLenFeature((),tf.string),
        'image/image_nm_raw': tf.io.FixedLenFeature((), tf.string),
        'image/image_nt_raw': tf.io.FixedLenFeature((), tf.string),
        'image/format': tf.io.FixedLenFeature((), tf.string),

        'image/id_height': tf.io.FixedLenFeature([], tf.int64),
        'image/id_width': tf.io.FixedLenFeature([], tf.int64),
        'image/dl_height': tf.io.FixedLenFeature([], tf.int64),
        'image/dl_width': tf.io.FixedLenFeature([], tf.int64),
    }

    def _partse_image_function(example_proto):
        data_features =tf.io.parse_single_example(example_proto,image_feature_description)
        decode_img_id_raw = tf.io.decode_raw(data_features['image/image_id_raw'],tf.uint8)
        decode_img_dl_raw = tf.io.decode_raw(data_features['image/image_dl_raw'], tf.uint8)
        decode_img_nm_raw = tf.io.decode_raw(data_features['image/image_nm_raw'], tf.uint8)
        decode_img_nt_raw = tf.io.decode_raw(data_features['image/image_nt_raw'], tf.uint8)

        data=dict()
        data['id'] = decode_img_id_raw
        data['dl'] = decode_img_dl_raw
        data['nm'] = decode_img_nm_raw
        data['nt'] = decode_img_nt_raw
        return data


    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(_partse_image_function,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def read_record_single_cls(imageTFRecord,num_parallel_reads=8,shuffle_buffer_size=8,batch_size=16,epoch=10):
    tfrecordFiles = tf.data.Dataset.list_files(imageTFRecord)
    dataset = tfrecordFiles.interleave(tf.data.TFRecordDataset,cycle_length=num_parallel_reads,num_parallel_calls=tf.data.experimental.AUTOTUNE)


    image_feature_description = {
        'image/image_id_raw':tf.io.FixedLenFeature((),tf.string),
        'image/lable':tf.io.FixedLenFeature((),tf.int64),
        'image/format': tf.io.FixedLenFeature((), tf.string),

        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
    }

    def _partse_image_function(example_proto):
        data_features =tf.io.parse_single_example(example_proto,image_feature_description)
        decode_img_raw = tf.io.decode_raw(data_features['image/image_raw'],tf.uint8)
        decode_img_label = tf.io.decode_raw(data_features['image/label'], tf.uint8)


        data=dict()
        data['img'] = decode_img_raw
        data['label'] = decode_img_label

        return data


    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(_partse_image_function,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=batch_size,drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.prefetch(buffer_size=batch_size)

    return dataset

def get_img_list(target_task,base_folder_id='z:/asdasdas/sorted_id',base_folder_dl='z:/12313/sorted_dl/',base_folder_natural='z:/1231231/card_checker/data2/negative'):
    if target_task is 'card_checker':
        list_images_id = os.listdir(base_folder_id)
        list_images_dl = os.listdir(base_folder_dl)
        list_bg_natural = os.listdir(base_folder_natural)

        target_images_id = [os.path.join(base_folder_id,img) for img in list_images_id if img.endswith('.jpg') or img.endswith('.png')]
        target_images_dl = [os.path.join(base_folder_dl, img) for img in list_images_id if img.endswith('.jpg') or img.endswith('.png')]
        target_bg_natural = [os.path.join(base_folder_natural, img) for img in list_images_id if img.endswith('.jpg') or img.endswith('.png')]
    elif target_task is 'NonID_checker':
        raise NotImplementedError
    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default='read',type=str,help='one of [write, read]')
    parser.add_argument("--target_task", default='card_checker', type=str, help='one of [card_checler, NonID_checker, condition_checker]')
    config = parser.parse_args()
    if config.mode is 'read':
        read_record_single_cls('images.tfrecords')
    elif config.mode is 'read':
        get_img_list(config.target_task,base_folder_id='z:/asdasdas/sorted_id',base_folder_dl='z:/12313/sorted_dl/',base_folder_natural='z:/1231231/card_checker/data2/negative')
    else:
        raise NotImplementedError














