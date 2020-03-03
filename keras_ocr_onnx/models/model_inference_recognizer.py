import sys
sys.append('../') #root path
from models.custom_ocr_recognizer import *
from utils.utils import *
from functools import partial
import numpy as np
import math
import tensorflow as tf

partial_resize = partial(tf.image.resize,method=tf.image.ResizeMethod.BILINEAR,antialias=True)

class Model_Inference():
    def __init__(self,config,target_image):
        self.config = config
        self.target_image = target_image
        self.build_model()
        self.step = tf.Variable(0,dtype=tf.int64)

    def build_model(self):
        """model"""
        self.model = build_model(alphabet=28, height=200, width=200, color=False, filters=(64, 128, 256, 256, 512, 512, 512),
                                 rnn_units=(128, 128), dropout=0.1,rnn_steps_to_discard=2, pool_size=2, stn=True)
        self.model.summary()

    def save(self,epoch):
        self.model.summary()
        self.mode.inputs[0].shape.dims[0]._value = 1
        for index, layer in enumerate(self.model.layers):
            print(index)
            try:
                layer.batch_size = 1
            except:
                pass
            try:
                layer._batch_input_shape = ((1,layer._batch_input_shape[1],layer._batch_input_shape[2],layer._batch_input_shape[3]))
            except:
                pass
            try:
                layer.input_shape[0] = ((1,layer.input_shape[0][1],layer.input_shape[0][2],layer.input_shape[0][3]))
            except:
                pass
            try:
                layer.output_shape[0] = ((1,layer.output_shape[0][1],layer.output_shape[0][2],layer.output_shape[0][3]))
            except:
                layer.output._shape._dims[0]._value = 1

        self.model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        self.model.summary()

        self.model.save(os.path.join(self.config.checkpoint_dir,"generator_scale_{}.h5".format(epoch)))

    def restore(self, N=None):
        self.generator = tf.keras.models.load_model(os.path.join(self.config.checkpoint_dir,"generator_scale.h5"),custom_objects={'InstanceNorm': InstanceNorm})

    @tf.function
    def _inference(self):
        gen_output = self.model(self.target_image)
        return gen_output

    def inference(self,N=0,start_N=None):
        gen_output = self._inference()
        return gen_output

