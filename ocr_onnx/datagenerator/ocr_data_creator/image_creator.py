import os
import cv2
import json
import glob
import time
import random
import argparse
import numpy as np

from tqdm import tqdm
from collections import OrderedDict

from utils import create_blank_image, image_brightness_control, bit_operation, image_brightness_control_v2

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

def create_ocr_traning_data_word(num_image, min_word=5, max_word=35, background=0.1, sentence_sparse=0.1, sentence_sparse2=0.3,
                                alphabet_number=0.1, spacing=0.3, brightness=0.2):

    if os.path.isdir("train_image") is False:
        os.mkdir("train_image")

    if os.path.isdir("train_label") is False:
        os.mkdir("train_label")

    background_image_list = glob.glob("background_template/*.*")
    background_image_count = len(background_image_list)

    for i in tqdm(range(num_image)):
        sentence_count = random.randint(10, 15) # parameter

        millis = int(round(time.time() * 1000))

        if os.path.isfile(str(millis) + ".json") is False:
            custom_info = OrderedDict()
            custom_info["info"] = OrderedDict()
            custom_info["license"] = OrderedDict()
            custom_info["images"] = []
            custom_info["annotations"] = []
            json.dumps(custom_info, ensure_ascii=False, indent="\t")
        else:
            with open(str(millis) + ".json", "r") as info:
                custom_info = json.load(info)

        bg_img = create_blank_image(IMAGE_WIDTH, IMAGE_HEIGHT)

        background_ratio = random.randint(0, 99)
        if background_ratio < background * 100:
            bg_idx = random.randint(0, background_image_count - 1)
            bg_img = cv2.imread(background_image_list[bg_idx])
            r,c,_ = bg_img.shape
            if r > 1024:
                if c > 512:
                    bg_img = cv2.resize(bg_img, (IMAGE_WIDTH, 1024)) # wh
                else:
                    bg_img = cv2.resize(bg_img, (c, 1024))
            else:
                if c > 512:
                    bg_img = cv2.resize(bg_img, (IMAGE_WIDTH, r))
                else:
                    bg_img = cv2.resize(bg_img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        sparse_ratio = random.randint(0, 99)
        sparse_flag = False
        if sparse_ratio < sentence_sparse * 100:
            sparse_flag = True
            sentence_count = random.randint(25, 30)

        annotations = OrderedDict()
        annotations["id"] = str(millis)
        annotations["text"] = []
        annotations["bbox"] = []

        div_h = int(IMAGE_HEIGHT / sentence_count)

        for sentence_repeat in range(sentence_count):
            word_count = random.randint(min_word, max_word)
            sentence_height = random.randint(int(div_h * 0.5), int(div_h * 0.8))

            if sparse_flag is True:
                sentence_height = random.randint(int(div_h * 0.8), int(div_h * 0.9))
                word_count = random.randint(max_word - 10, max_word)

            start_position_x = random.randint(10, int(IMAGE_WIDTH / 3))
            start_position_y = random.randint(sentence_repeat * div_h, (sentence_repeat + 1) * div_h - sentence_height)

            sparse_ratio2 = random.randint(0, 99)
            if sparse_ratio2 < sentence_sparse2 * 100 and sparse_flag == True:
                continue

            for repeat in range(word_count):
                word_folder = ""
                kor_alpha_num_ratio = random.randint(0, 99)
                if kor_alpha_num_ratio < alphabet_number * 100:
                    word_folder = "alphabet_number"
                else:
                    word_folder = "selected_syllable"

                alphabet_number_folder_list = glob.glob(word_folder + "/*")
                idx = random.randint(0, len(alphabet_number_folder_list) - 1)
                word_txt = alphabet_number_folder_list[idx].split('/')[1]

                text_count = glob.glob(word_folder + "/" + word_txt + "/*.*")
                word_idx = random.randint(0, len(text_count) - 1)
                word_img = cv2.imread(text_count[word_idx])

                text_dict = OrderedDict()
                text_dict["text_id"] = str(millis) + "-" + str(sentence_repeat) + "-" + str(repeat)
                text_dict["contents"] = word_txt

                bbox_dict = OrderedDict()
                bbox_dict["bbox_id"] = str(millis) + "-" + str(sentence_repeat) + "-" + str(repeat)

                rows, cols, _ = word_img.shape
                ncols = cols
                if rows > sentence_height:
                    ratio = int(rows / sentence_height)
                    ncols = int(cols / ratio)

                resized_word_img = cv2.resize(word_img, (ncols, sentence_height))

                blank_flag = random.randint(0, 99)

                if blank_flag < spacing * 100:
                    resized_word_img = create_blank_image(ncols, sentence_height)

                if start_position_x + ncols > IMAGE_WIDTH : break

                bg_img = bit_operation(bg_img, resized_word_img, start_position_x, start_position_y)

                bbox = [int((2 * start_position_x + ncols) / 2), int((start_position_y + sentence_height) / 2), ncols, sentence_height]

                if blank_flag >= spacing * 100:
                    annotations["text"].append(text_dict)

                    bbox_dict["contents"] = bbox
                    annotations["bbox"].append(bbox_dict)
                start_position_x += ncols
        
        file_name = str(millis) + ".jpg"

        bright_ratio = random.randint(0, 99)
        brigt_ran = brightness * 100 / 2
        if bright_ratio >= 0 and bright_ratio < brigt_ran:
            bg_img = image_brightness_control(bg_img)
        elif bright_ratio >= brigt_ran and bright_ratio < brigt_ran * 2:
            bg_img = image_brightness_control_v2(bg_img)

        cv2.imwrite("train_image/" + file_name, bg_img)

        images = OrderedDict()
        images["id"] = str(millis)
        images["width"] = IMAGE_WIDTH
        images["height"] = IMAGE_HEIGHT
        images["file_name"] = file_name
        images["data_captured"] = time.time()

        custom_info["images"].append(images)
        custom_info["annotations"].append(annotations)

        with open("train_label/" + str(millis) + ".json", 'w', encoding="utf-8") as make_file:
            json.dump(custom_info, make_file, ensure_ascii=False, indent="\t")
            make_file.close()

def main()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_image', type=int, default=10)
    parser.add_argument('--background_ratio', type=float, default=0.1)
    parser.add_argument('--min_word', type=int, default=5)
    parser.add_argument('--max_word', type=int, default=35)
    parser.add_argument('--sparse_ratio', type=float, default=0.1)
    parser.add_argument('--spacing_ratio', type=float, default=0.3)
    parser.add_argument('--alphabet_ratio', type=float, default=0.1)

    args = parser.parse_args()

    NUM_TRAINING_IMAGE = args.num_image

    create_ocr_traning_data_word(NUM_TRAINING_IMAGE)

if __name__ == "__main__":
    main()