## Tensorflow record (.tfrecord) File I/O

This is a document of tfrecord with a customized example to our OCR model

※ The TFRecord  format is briefly documented here, and described as the recommended format for feeding data into Tensorflow here and here.
This library facilitates producing data in the TFRecord format directly in node.js. The library is not "official" - it is not part of Tensorflow,
 and it is not implemented by the Tensorflow team.
 
### code

#### install dependencies

```
matplotlib
tensorflow2.0
numpy
```
 
### Usage

the example below covers recommended API usage

#### write images to tfrecord for 'cardchecker' classification model

```
write_record_binary_cls(target_images_id,target_images_dl,target_bg_natural)
```

```
@params
target_images_id : a list of target identification card images
target_images_dl : a list of target driver license card images
target_bg_natural : a list of back ground images
```

```
    with tf.io.TFRecordWriter('images.tfrecords') as writer:
        for dl, id, bg in zip_list:
            img_id = tf.keras.preprocessing.image.load_img(id)
            img_dl = tf.keras.preprocessing.image.load_img(dl)
            img_bg = tf.keras.preprocessing.image.load_img(bg)
            img_id_array = tf.keras.preprocessing.image.img_to_array(img_id)
            img_dl_array = tf.keras.preprocessing.image.img_to_array(img_dl)
            img_bg_array = tf.keras.preprocessing.image.img_to_array(img_bg)

            crop_img_id_array = img_id_array[27:90,90:368,0].astype(np.uint8)
            crop_img_dl_array = img_id_array[10:40, 20:120, 0].astype(np.uint8)
            img_nt_array = img_nt_array[...,0]

            if random.random() > 0.5:
                rand_h_integer = random.randint(90,img_id_array.shape[0]-63)
                rand_w_integer = random.randint(0, img_id_array.shape[1] - 278)
                img_nm_array = img_id_array[rand_h_integer:rand_h_integer+63,rand_w_integer:rand_w_integer+278,0].astype(np.uint8)
            else:
                rand_h_integer = random.randint(40, img_dl_array.shape[0] - 30)
                rand_w_integer = random.randint(0, img_dl_array.shape[1] - 100)
                img_nm_array = img_dl_array[rand_h_integer:rand_h_integer + 63, rand_w_integer:rand_w_integer + 278,0].astype(np.uint8)

            resized_id_array = np.reshape(cv2.resize(crop_img_id_array,(320,320,)),(320,320,1))
            resized_dl_array = np.reshape(cv2.resize(crop_img_dl_array, (320, 320,)), (320, 320, 1))
            resized_nm_array = np.reshape(cv2.resize(img_nm_array, (320, 320,)), (320, 320, 1))
            resized_nt_array = np.reshape(cv2.resize(img_nt_array, (320, 320,)), (320, 320, 1)).astype(np.uint8)

            img_id_bytes = resized_id_array.tostring()
            img_dl_bytes = resized_dl_array.tostring()
            img_nm_bytes = resized_nm_array.tostring()
            img_nt_bytes = resized_nt_array.tostring()
            def image_example(img_id_raw,img_dl_raw,img_nm_raw,img_nt_raw):
                feature={
                    'image/image_id_raw': _bytes_feature(img_id_array),
                    'image/image_dl_raw': _bytes_feature(img_dl_array),
                    'image/image_nm_raw': _bytes_feature(img_nm_array),
                    'image/image_nt_raw': _bytes_feature(img_nt_array),

                    'image/format': _bytes_feature(b'jpg'),

                    'image/id_height': _int64_feature(63),
                    'image/id_width': _int64_feature(278),
                    'image/dl_height': _int64_feature(30),
                    'image/dl_width': _int64_feature(100),
                }

                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                return example_proto.SerializetoString()

            def image_example_single_cls(img_raw,label):
                feature={
                    'image/image_id_raw': _bytes_feature(img_raw),
                    'image/format': _bytes_feature(b'jpg'),
                    'image/label': _int64_feature(label),
                    'image/height': _int64_feature(320),
                    'image/width': _int64_feature(320),
                }

                example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                return example_proto.SerializetoString()

            seed = random.random()
            if seed <=0.25:
                tf_example = image_example_single_cls(img_id_bytes,label=1)
                writer.write(tf_example)
            elif seed <= 0.5:
                tf_example = image_example_single_cls(img_dl_bytes, label=1)
                writer.write(tf_example)
            elif seed <= 0.75:
                tf_example = image_example_single_cls(img_nm_bytes, label=0)
                writer.write(tf_example)
            else:
                tf_example = image_example_single_cls(img_nt_bytes, label=0)
                writer.write(tf_example)

```

#### read and load TFRecord for 'cardchecker' classification model
```
read_record_single_cls('images.tfrecortds')
```

```
@params
images.tfrecords : a target tfrecord file
```

```
def read_record_single_cls(imageTFRecord,num_parallel_reads=8,shuffle_buffer_size=8,batch_size=16,epoch=10):
    tfrecordFiles = tf.data.Dataset.list_files(imageTFRecord)
    dataset = tfrecordFiles.interleave(tf.data.TFRecordDataset,cycle_length=num_parallel_reads,num_parallel_calls=tf.data.experimental.AUTOTUNE)


    image_feature_description = {
        'image/image_id_raw':tf.io.FixedLenFeature((),tf.string),
        'image/lable':tf.io.FixedLenFeature((),tf.int64),
        'image/format': tf.io.FixedLenFeature((), tf.string),

        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
    }

    def _partse_image_function(example_proto):
        data_features =tf.io.parse_single_example(example_proto,image_feature_description)
        decode_img_raw = tf.io.decode_raw(data_features['image/image_raw'],tf.uint8)
        decode_img_label = tf.io.decode_raw(data_features['image/label'], tf.uint8)


        data=dict()
        data['img'] = decode_img_raw
        data['label'] = decode_img_label

        return data


    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(_partse_image_function,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=batch_size,drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.prefetch(buffer_size=batch_size)

    return dataset
```

#### Augment input data after loading data
```
data = apply_aug_single_cls(images_features, batch_size)
```

```
@params
image_features : decoded images data from tfrecord
batch_size : target batch size in the trrain phase

@output
augmented data of batch size
```

```
def apply_aug_single_cls(image_features,batch_size):
    try:
        decode_img = tf.reshape(image_features['img'],(batch_size,320,320,1))
    except:
        print('fail to decode image')

    decode_label = tf.reshape(image_features['label'],(batch_size,1))

    decode_img = tf.image.random_brightness(decode_img,max_delta=0.1)
    decode_img = tf.image.random_flip_left_right(decode_img)
    decode_img = tf.image.random_flip_up_down(decode_img)


    data = dict()
    data['img']=decode_img
    data['label']=decode_label
    return data
```