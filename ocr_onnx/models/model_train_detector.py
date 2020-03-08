import sys
from models.custom_ocr_detector import *
from utils.utils import *
from functools import partial
import numpy as np
import tensorflow as tf
tf.keras.backend.floatx()

partial_resize = partial(tf.image.resize,method=tf.image.ResizeMethod.BILINEAR,antialias=True)

# class Model_Inference():
#     def __init__(self,config,target_image):
#         self.config = config
#         self.target_image = target_image
#         self.build_model()
#         self.step = tf.Variable(0,dtype=tf.int64)

class Model_Train():
    def __init__(self,config):
        self.config = config
        self.step = tf.Variable(0,dtype=tf.int64)
        self.build_model()
        log_dir = os.path.join(config.summary_dir)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir,0o777)
        self.train_summary_writer = tf.summary.create_file_writer(log_dir)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.loss = tf.keras.losses.CategoricalCrossentropy()
        self.train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    def build_model(self):
        """model"""
        self.model = build_model()
        learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.model.summary()

    def save(self,epoch):
        self.model.summary()
        # self.model.inputs[0].shape.dims[0]._value = 1
        # for index, layer in enumerate(self.model.layers):
        #     print(index)
        #     try:
        #         layer.batch_size = 1
        #     except:
        #         pass
        #     try:
        #         layer._batch_input_shape = ((1,layer._batch_input_shape[1],layer._batch_input_shape[2],layer._batch_input_shape[3]))
        #     except:
        #         pass
        #     try:
        #         layer.input_shape[0] = ((1,layer.input_shape[0][1],layer.input_shape[0][2],layer.input_shape[0][3]))
        #     except:
        #         pass
        #     try:
        #         layer.output_shape[0] = ((1,layer.output_shape[0][1],layer.output_shape[0][2],layer.output_shape[0][3]))
        #     except:
        #         layer.output._shape._dims[0]._value = 1

        self.model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        # self.model.summary()

        self.model.save(os.path.join(self.config.checkpoint_dir,"generator_scale_{}.h5".format(epoch)))

    def restore(self, N=None):
        self.generator = tf.keras.models.load_model(os.path.join(self.config.checkpoint_dir,"generator_scale.h5"),custom_objects={'InstanceNorm': InstanceNorm})


    @tf.function
    def train_one_step(self,image,targets):

        image = tf.cast(image, tf.float32)
        image=image/255.0
        # image = tf.cast(image, tf.float32)


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
        # return_dicts.update({'acc':self.train_acc})
        return predictions, loss

    #@tf.function
    def train_step(self,data,summary_name='train',log_interval=0):
        """training"""
        image = data['image']
        targets = data['label']
        targets = tf.keras.utils.to_categorical(targets, 10)
        predictions, loss = self.train_one_step(image,targets)
        self.train_acc(targets, predictions)
        self.train_loss(loss)
        """log summary"""
        # if summary_name and self.step.numpy() % log_interval ==0:
        with self.train_summary_writer.as_default():
            # for key, value in result_logs_dict.items():
            value = self.train_acc.result().numpy()
            # value = tf.reduce_mean(value)
            # value = 1
            if len(value.shape) == 0:
                tf.summary.scalar("train",value,step=self.step)
            elif len(value.shape) in [3, 4]:
                tf.summary.scalar("train", denormalize(value), step=self.step)

        log = "loss:{} accuracy:{}".format(self.train_loss.result().numpy(),self.train_acc.result().numpy())
        return log



















