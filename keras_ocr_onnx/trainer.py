import time, datetime, argparse, cv2, os
from models.model_train_recognizer import Model_Train
from utils.utils import *
from datagenerator.genGenerator import read_record_single_cls, apply_aug_single_cls
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

"""
===========================================================
                       configuration
===========================================================
"""

start = time.time()
time_now=datetime.datetime.now()
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",default=4,type=str)
parser.add_argument("--epoch", default=10000, type=int)
parser.add_argument("--target_size", default=250, type=list,nargs="+",help = "Image size after crop")
parser.add_argument("--batch_size", default=8, type=int,help = "Minibatch size(global)")
parser.add_argument("--data_root_test", default='./dataset/test', type=str,help = "Dir to data root")
parser.add_argument("--image_file", default='./dataset/test/176039.jpg', type=str,help = "Dir to data root")
parser.add_argument("--channels", default=1, type=int,help = "Channel size")
parser.add_argument("--color_map", default="RGB", type=int,help = "Channel mode. [RGB, YCbCr]")
parser.add_argument("--model_tag", default="default", type=str,help = "Exp name to save logs/checkpoints.")
parser.add_argument("--checkpoint_dir", default="outputs/checkpoints/", type=str,help = "Dir for checkpoints")
parser.add_argument("--summary_dir", default="outputs/summaries/", type=str,help = "Dir for tensorboard logs.")
parser.add_argument("--restore_file", default=None, type=str,help = "file for restoration")
parser.add_argument("--graph_mode", default=False, type=bool,help = "use graph mode for training")
config = parser.parse_args()

def generate_expname_automatically():
    name = "OCR_%s_%02d_@02d_@02d_@-2d_@02d" % (config.model_tag, time_now.month,time_now.day, time_now.hour,
                                                time_now.minute, time_now.second)
    return name

expname = generate_expname_automatically()
config.checkppint_dir += "OCR_" + config.model_tag; check_folder(config.checkpoint_dir)
config.summary_dir += expname ; check_folder(config.summary_dir)
"""
===========================================================
                      prepare dataset
===========================================================
"""
# read dataset

dataset = read_record_single_cls('datagenerator/images.tfrecords',batch_size = config.batch_size)
"""
===========================================================
                      build model
===========================================================
"""
model = Model_Train(config)


for e in range(config.epoch):
    for i, image_features in enumerate(dataset):
        # print(image_features)
        data = apply_aug_single_cls(image_features, config.batch_size)
        #
        log = model.train_step(data)
        print("[epoch {} train step {}] step : {}".format(e,i,log))
    if e % 10 == 0:
        save_path = model.save(e)
    model.train_loss.reset_states()
    model.train_acc.reset_states()

config = parser.parse_args()