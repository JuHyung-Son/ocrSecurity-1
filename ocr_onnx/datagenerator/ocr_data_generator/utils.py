import os
import cv2
import json
import glob
import random
import numpy as np

from tqdm import tqdm
from collections import OrderedDict
from icrawler.builtin import BingImageCrawler

IMAGE_HEIGHT, IMAGE_WIDTH = 512, 512

def crawl_image(word):
    bing_crawler = BingImageCrawler(parser_threads=1, downloader_threads=4,
                                    storage={'root_dir': '/Users/wkh/wkh_dabeeo_dev/visual_studio_code/data_processor/bingtest'})
    bing_crawler.session.verify = True
    bing_crawler.crawl(keyword=word, max_num=1000,
                        #  date_min=None, date_max=None,
                        min_size=(10,10), max_size=None)

def create_blank_image(width, height):
    blank_img = 255* np.ones(shape=[height, width, 3], dtype=np.uint8)
    return blank_img

def image_brightness_control(img):
    controled_img = img
    rb = random.randint(50, 150)
    M = np.ones(img.shape, dtype = "uint8") * rb

    plus_minus = random.randint(0, 1)
    if plus_minus is 0:
        controled_img = cv2.add(img, M)
    else:
        controled_img = cv2.subtract(img, M)

    return controled_img

def image_brightness_control_v2(img):
    controled_img = img
    rb = random.randint(50, 150)
    updown = random.randint(0, 1)
    if updown is 0:rb = 50
    else: rb = 150
    M = np.ones(img.shape, dtype = "uint8") * rb

    step = 150 - 50
    step /= IMAGE_HEIGHT

    for i in range(IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            if updown is 0:
                M[i][j][0] = rb + i * step
                M[i][j][1] = rb + i * step
                M[i][j][2] = rb + i * step
            else:
                M[i][j][0] = rb - i * step
                M[i][j][1] = rb - i * step
                M[i][j][2] = rb - i * step

    plus_minus = random.randint(0, 1)
    if plus_minus is 0:
        controled_img = cv2.add(img, M)
    else:
        controled_img = cv2.subtract(img, M)

    return controled_img

def bit_operation(background_image, logo, hpos, vpos):
    rows, cols, _ = logo.shape
    roi = background_image[vpos:rows+vpos, hpos:cols+hpos]

    gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_logo, 100, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    bg_img = cv2.bitwise_and(roi, roi, mask=mask_inv)
    logo_img = cv2.bitwise_and(logo, logo, mask=mask)

    dst = cv2.add(bg_img, logo_img)

    background_image[vpos:rows+vpos, hpos:cols+hpos] = dst

    return background_image

def extract_annotation(image_folder):
    custom_info = OrderedDict()
    custom_info["info"] = OrderedDict()
    custom_info["license"] = OrderedDict()
    custom_info["images"] = []
    custom_info["annotations"] = []

    json.dumps(custom_info, ensure_ascii=False, indent="\t")

    with open("printed_data_info.json", "r") as info:
        data_info = json.load(info)

    for i in tqdm(range(len(data_info['images']))):
        if os.path.isfile(image_folder+ "/" + str(data_info['images'][i]['file_name'])):
            custom_info["images"].append(data_info['images'][i])
            custom_info["annotations"].append(data_info['annotations'][i])

    with open(image_folder + ".json", 'w', encoding="utf-8") as make_file:
        json.dump(custom_info, make_file, ensure_ascii=False, indent="\t")
        make_file.close()

def create_background_image(num_stamp, stamp_valid = True):
    if os.path.isdir("train_image_background") is False:
        os.mkdir("train_image_background")

    stamp_list = glob.glob("background_image/stamp/*.*")
    background_list = glob.glob("background_image/background/*.*")

    for bl in background_list:
        bg_image = cv2.imread(bl)
        bg_image = cv2.resize(bg_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        if stamp_valid:
            for i in range(num_stamp):
                idx = random.randint(0, len(stamp_list) - 1)
                stamp_image = cv2.imread(stamp_list[idx])
                size = random.randint(100, 250)
                resize_stamp = cv2.resize(stamp_image, (size, size))
                pos_x_rand = random.randint(0, IMAGE_WIDTH - size)
                pos_y_rand = random.randint(0, IMAGE_HEIGHT - size)
                bg_image = bit_operation(bg_image, resize_stamp, pos_x_rand, pos_y_rand)

        # millis = int(round(time.time() * 1000))
        # file_name = str(millis) + ".jpg"
        # cv2.imwrite("train_image_background/" + file_name, bg_image)

"""
create training data and annotation file
"""
# with open("01_printed_word_images.json", "r") as info:
#     word_info = json.load(info)
# create_ocr_traning_data_word(6, word_info, "01_printed_word_images")

'''
create background image
'''
# for i in tqdm(range(10)): # image count
#     stamp_count = random.randint(3, 5)
#     create_background_image(stamp_count, True)

''' 
extract each annotation from original printed_data
'''
# extract_annotation("01_printed_syllable_images")