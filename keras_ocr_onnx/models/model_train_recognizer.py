import sys
from models.custom_ocr_recognizer import *
from utils.utils import *
from functools import partial
import numpy as np
import tensorflow as tf

partial_resize = partial(tf.image.resize,method=tf.image.ResizeMethod.BILINEAR,antialias=True)

# class Model_Inference():
#     def __init__(self,config,target_image):
#         self.config = config
#         self.target_image = target_image
#         self.build_model()
#         self.step = tf.Variable(0,dtype=tf.int64)

class Model_Train():
    def __init__(self,config,target_image):
        self.config = config
        self.step = tf.Variable(0,dtype=tf.int32)
        self.build_model()
        log_dir = os.path.join(config.summary_dir)

        if not os.path.isdir(log_dir):
            os.mkdir(log_dir,0o777)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    def build_model(self):
        """model"""
        self.model = build_model(alphabet=28, height=200, width=200, color=False, filters=(64, 128, 256, 256, 512, 512, 512),
                                 rnn_units=(128, 128), dropout=0.1,rnn_steps_to_discard=2, pool_size=2, stn=True)
        learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
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
    def train_one_step(self,data):
        image=data['img']
        targets=data['label']
        image=tf.cast(image,tf.float32)
        image=image/255.0

        with tf.GradientTape() as tape:
            # Make a prediction
            predictions = self.model(image)
            # Get the error/loss using the Loss_object previously defined
            loss = self.loss(targets,predictions)
        # compute the gradient with respect to the loss
        gradients = tape.gradient(loss,self.model.trainable_variables)
        # Change the weights of the model
        self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        # the metrics are accumulate over time. You don't need to average it yourself.
        self.train_loss(loss)
        self.train_acc(targets,predictions)

        return_dicts = {'loss':self.train_loss}
        return_dicts.update({'acc':self.train_acc})
        return return_dicts

    def train_step(self,data,summary_name='train',log_interval=0):
        """training"""
        result_logs_dict = self.train_one_step(data)
        """log summary"""
        # if summary_name and self.step.numpy() % log_interval ==0:
        with self.train_summary_writer.as_default():
            for key, value in result_logs_dict.items():
                value = value.result().numpy()
                tf.summary.scalar("{}_{}".format(summary_name,key),value,step=self.step)
        log = "loss:{} accuracy:{}".format(result_logs_dict["loss".result().numpy(),result_logs_dict['acc'].result().numpy()])
        return log



















