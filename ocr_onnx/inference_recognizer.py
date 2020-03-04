import time, datetime, argparse, cv2, os
from models.model_inference_recognizer import Model_Inference
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
# read image
config.image_file1 = '/root/optimization/~.jpg'

ID = 0

if ID is 0:
    img = cv2.imread(config.image_file1)[...,::-1] #convert to rgb
    img = img[27:90,90:368,0]
    img = np.reshape(cv2.resize(img,(320,320)),(320,320,1))
else:
    img = cv2.imread(config.image_file1)[..., ::-1]  # convert to rgb
    img = img[10:40, 20:120, 0]
    img = np.reshape(cv2.resize(img, (320, 320)), (320, 320, 1))

temp_img = img

"""resize image"""
H, W, _ = img.shape
H,W = (250,int(W*250/H)) if H<W else (int(H*250/W),250)
img = cv2.resize(img,(H,W))

"""reshape image"""
img = np.expand_dims(img,axis=0)
img = np.expand_dims(img,axis=3)
img = normalize(img)
print(img.shape)

"""
===========================================================
                      build model
===========================================================
"""
model = Model_Inference(config,target_image=img)
model.restore()


"""
===========================================================
                         inference
===========================================================
"""
output = model.inference()
print(output)

cv2.imshow("img",temp_img)
cv2.waitKey(0)