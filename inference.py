import time, datetime, argparse, cv2, os
import ocr7
import matplotlib.pyplot as plt
import time
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
parser.add_argument("--image_file", default='/root/optimization/ocrSecurity/test_image.png', type=str,help = "Dir to data root")
parser.add_argument("--channels", default=1, type=int,help = "Channel size")
parser.add_argument("--color_map", default="RGB", type=str,help = "Channel mode. [RGB, YCbCr]")
parser.add_argument("--model_tag", default="default", type=str,help = "Exp name to save logs/checkpoints.")
parser.add_argument("--checkpoint_dir", default="outputs/checkpoints/", type=str,help = "Dir for checkpoints")
parser.add_argument("--summary_dir", default="outputs/summaries/", type=str,help = "Dir for tensorboard logs.")
parser.add_argument("--restore_file", default=None, type=str,help = "file for restoration")
parser.add_argument("--graph_mode", default=False, type=bool,help = "use graph mode for training")
config = parser.parse_args()

"""
===========================================================
                      build model
===========================================================
"""
image = cv2.imread(config.image_file,cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
images = []
images.append(image)

start = time.time()
pipeline = ocr7.ocr_model.Pipeline()
end = time.time()
print("consumed time : {} sec", end-start)

prediction_groups = pipeline.recognize(images)

"""
===========================================================
                         inference
===========================================================
"""
# fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
# for ax, image, predictions in zip(axs, images, prediction_groups):
ocr7.utils.drawAnnotations(image=images[0], predictions=prediction_groups[0], ax=axs)
output_name=os.path.basename(config.image_file)
fig.savefig('/root/optimization/ocrSecurity/output'+output_name)
