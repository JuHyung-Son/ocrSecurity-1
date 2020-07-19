import cv2
import detection
import numpy as np
import tools
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import pipeline
import datasets
import os
import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='model', type=str)
args = parser.parse_args()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)


def get_ocr_detector_dataset(type):
    training_gt_dir = f'../ocr_dataset/{type}/loc_gt'
    training_images_dir = f'../ocr_dataset/{type}/images'
    dataset = []
    for gt_filepath in glob.glob(os.path.join(training_gt_dir, '*.txt')):
        image_id = os.path.split(gt_filepath)[1].split('_')[0]
        image_path = os.path.join(training_images_dir, image_id + '.png')
        if not os.path.exists(image_path):
            print()
            print(image_id)
            print()
        lines = []
        with open(gt_filepath, 'r') as f:
            current_line = []
            for row in f.read().split('\n'):
                if row == '':
                    lines.append(current_line)
                    current_line = []
                else:
                    row = row.split(' ')[5:]
                    character = row[-1][1:-1]
                    if character == '':
                        continue
                    x1, y1, x2, y2 = map(int, row[:4])
                    current_line.append((np.array([[x1, y1], [x2, y1], [x2, y2], [x1,
                                                                                  y2]]), character))
        # Some lines only have illegible characters and if skip_illegible is True,
        # then these lines will be blank.
        lines = [line for line in lines if line]
        dataset.append((image_path, lines, 1))
    return dataset


def output(img_path, pipe, out_path):
    img_path = img_path[0]
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    predictions = pipe.recognize(images=[img])[0]
    drawn = tools.drawBoxes(
        image=img, boxes=predictions, boxes_format='predictions'
    )
    print(
        'Predicted:', [text for text, box in predictions]
    )
    cv2.imwrite(out_path, drawn)

validation = get_ocr_detector_dataset('test')

generator_kwargs = {'width': 640, 'height': 640}

validation_image_generator = datasets.get_detector_image_generator(
    labels=validation,
    **generator_kwargs
)

detector = detection.Detector()
detector.model.load_weights(f'model/{args.model}.h5')
pipe = pipeline.Pipeline(detector=detector)
for i, img in enumerate(validation):
    print('sdfsdf')
    print(img[0])
    output(img, pipe, f'output_{i}.png')