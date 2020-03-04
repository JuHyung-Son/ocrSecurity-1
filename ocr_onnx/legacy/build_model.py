import typing
import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras2onnx
from util_rec import _transform


def CTCDecoder():
    def decoder(y_pred):
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0]) * tf.keras.backend.cast(
            input_shape[1], 'float32')
        unpadded = tf.keras.backend.ctc_decode(y_pred, input_length)[0][0]
        unpadded_shape = tf.keras.backend.shape(unpadded)
        padded = tf.pad(unpadded,
                        paddings=[[0, 0], [0, input_shape[1] - unpadded_shape[1]]],
                        constant_values=-1)
        return padded

    return tf.keras.layers.Lambda(decoder, name='decode')

def build_model(alphabet, height, width, color, filters, rnn_units, dropout,
                rnn_steps_to_discard, pool_size, stn=True):
    inputs = keras.layers.Input((height, width, 3 if color else 1), batch_size=2)
    x = keras.layers.Permute((2, 1, 3))(inputs)
    # x = keras.layers.Lambda(lambda x: x[:, :, ::-1])(x)
    # x = x[..., ::-1]
    x = keras.layers.Conv2D(filters[0], (3, 3), activation='relu', padding='same', name='conv_1')(x)
    x = keras.layers.Conv2D(filters[1], (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = keras.layers.Conv2D(filters[2], (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = keras.layers.BatchNormalization(name='bn_3')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,  2), name='maxpool_3')(x)
    x = keras.layers.Conv2D(filters[3], (3, 3), activation='relu', padding='same', name='conv_4')(x)
    x = keras.layers.Conv2D(filters[4], (3, 3), activation='relu', padding='same', name='conv_5')(x)
    x = keras.layers.BatchNormalization(name='bn_5')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_5')(x)
    x = keras.layers.Conv2D(filters[5], (3, 3), activation='relu', padding='same', name='conv_6')(x)
    x = keras.layers.Conv2D(filters[6], (3, 3), activation='relu', padding='same', name='conv_7')(x)
    x = keras.layers.BatchNormalization(name='bn_7')(x)

    stn_input_output_shape = (width // pool_size**2, height // pool_size**2, 512)
    stn_input_layer = keras.layers.Input(stn_input_output_shape, batch_size=1)
    locnet_y = keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu')(stn_input_layer)
    locnet_y = keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(locnet_y)
    locnet_y = keras.layers.Flatten()(locnet_y)
    locnet_y = keras.layers.Dense(64, activation='relu')(locnet_y)
    locnet_y = keras.layers.Dense(6, weights=[
        np.zeros((64, 6), dtype=np.float32),
        np.float32([[1, 0, 0], [0, 1, 0]]).flatten()
    ])(locnet_y)
    localization_net = keras.models.Model(inputs=stn_input_layer, outputs=locnet_y)

    # stn 끄기
    x= keras.layers.Lambda(_transform,
                           output_shape=stn_input_output_shape)([x, localization_net(x)])
    x = keras.layers.Reshape(target_shape=(width // pool_size**2,
                                           (height // pool_size ** 2) * 512),
                            name='reshape')(x)
    x = keras.layers.Dense(rnn_units[0], activation='relu', name='fc_9')(x)

    rnn_1_forward = keras.layers.LSTM(rnn_units[0], kernel_initializer='he_normal', return_sequences=True, name='lstm_10')(x)
    rnn_1_back = keras.layers.LSTM(rnn_units[0], kernel_initializer='he_normal', go_backwards=True, return_sequences=True, name='lstm_10_back')(x)
    rnn_1_add = keras.layers.Add()([rnn_1_forward, rnn_1_back])
    rnn_2_forward = keras.layers.LSTM(rnn_units[1],
                                      kernel_initializer="he_normal",
                                      return_sequences=True,
                                      name='lstm_11')(rnn_1_add)
    rnn_2_back = keras.layers.LSTM(rnn_units[1],
                                   kernel_initializer="he_normal",
                                   go_backwards=True,
                                   return_sequences=True,
                                   name='lstm_11_back')(rnn_1_add)
    x = keras.layers.Concatenate()([rnn_2_forward, rnn_2_back])
    backbone = keras.models.Model(inputs=inputs, outputs=x)
    x = keras.layers.Dropout(0.2, name='dropout')(x)
    x = keras.layers.Dense(alphabet+1, kernel_initializer='he_normal', activation='softmax', name='fc_12')(x)
    x = keras.layers.Lambda(lambda x: x[:, 2:])(x)
    model = keras.models.Model(inputs=inputs, outputs=x)
    prediction_model = keras.models.Model(inputs=inputs, outputs=CTCDecoder()(model.output))

    return backbone, model, prediction_model


if __name__ == '__main__':
    import os

    """
        'height': 31,
        'width': 200,
        'color': False,
        'filters': (64, 128, 256, 256, 512, 512, 512),
        'rnn_units': (128, 128),
        'dropout': 0.25,
        'rnn_steps_to_discard': 2,
        'pool_size': 2,
        'stn': True,
    """

    backbone, model, prediction_model = build_model(28, 200, 200, False, (64, 128, 256, 256, 512, 512, 512),
                                                    (128, 128), 0.1, 2, 2, True)
    # model.summary()
    onnx_backbone = keras2onnx.convert_keras(backbone, backbone.name)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    # prediction_model = keras2onnx.convert_keras(prediction_model, prediction_model.name)
    # print(onnx_model.graph)
    keras2onnx.save_model(onnx_backbone, 'backbone.onnx')
    keras2onnx.save_model(onnx_model, 'model.onnx')
    # keras2onnx.save_model(prediction_model, 'prediction.onnx')

    print('ocr2onnx DONE')

    os.system('./onnx2trt backbone.onnx -o model.trt -d 16 -b 4')
    # os.system('./onnx2trt model.onnx -o model.trt -d 16 -b 4')
    # os.system('./onnx2trt prediction_model.onnx -o model.trt -d 16 -b 4')