import tensorflow as tf
import os
import numpy as np
import keras
import cv2
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation


dataset_path = r'Z:\Drive\AI\Project Daguerre\Stage 1 Phoebe\Datasets\GET-C_10'
ckpt_path = r'Z:\Drive\AI\Project Daguerre\Stage 1 Phoebe\Model Checkpoints\GETC\Period'
best_ckpt_path = r'Z:\Drive\AI\Project Daguerre\Stage 1 Phoebe\Model Checkpoints\GETC\Best_only'
tfrecord_path = r'Z:\Drive\AI\Project Daguerre\Stage 1 Phoebe\TFRecord\GETC-50k'
dataset_category = ['Basketball', 'PingPong', 'Running', 'Long jump', 'High jump', 'Baseball', 'Soccer', 'Swimming',
                    'Lacrosse', 'Ice hockey']
tfr_file_name = '04'
EPOCH = 10


def path_to_tfrecord(path, subfolder_list, save_path):
    # Create image tensor list and label list, initialize category label counter
    img_list = []
    y_list = []
    category_counter = 0
    path = path if path.endswith('/') else path + '/'
    # Traverse images in all subfolders and append them to img_list. Also creating label list: y_list.
    for sub_name in subfolder_list:
        sub_path = path + sub_name
        print('Processing sub_path:', sub_path)
        dir_list = os.listdir(sub_path)
        for dirs in range(len(dir_list)):
            # Decode, resize, and tensorize image
            with tf.gfile.FastGFile(sub_path + '/' + str(dirs) + '.jpg', 'rb') as img:
                image_raw_data = img.read()
            with tf.Session() as sess:
                img_data = tf.image.decode_jpeg(contents=image_raw_data, channels=3)
                img_data = tf.image.convert_image_dtype(image=img_data, dtype=tf.float32)
                img_resized = tf.image.resize_images(images=img_data, size=[400, 600], method=0)
                sess.close()
            # Append x, y list
            img_list.append(img_resized)
            y_list.append(category_counter)
            print('Appended No.', dirs, 'in', sub_name)
        category_counter = category_counter + 1
    # Transform lists in to Numpy array
    x_list = np.array(img_list)
    y_list = np.array(y_list)
    # Write x_list, y_list to TFRecord file: example
    writer = tf.python_io.TFRecordWriter(path=save_path + '/' + tfr_file_name)
    _int64_feature = lambda a: tf.train.Feature(int64_list=tf.train.Int64List(value=[a]))
    _bytes_feature = lambda a: tf.train.Feature(bytes_list=tf.train.BytesList(value=[a]))
    for index in range(len(x_list)):
        example = tf.train.Example(features=tf.train.Features(feature={'image': x_list[index],
                                                                       'label': y_list[index]}))
        writer.write(example.SerializeToString())
    writer.close()


def build(path, width, height, depth, classes):
    classes = classes
    input_shape = (width, height, depth)

    def parser(record):
        keys_to_features = {'image': tf.FixedLenFeature([], tf.string), 'label': tf.FixedLenFeature([], tf.int64)}
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.image.decode_jpeg(parsed["image"])
        image = tf.reshape(image, [400, 600, 1])
        label = tf.cast(parsed["label"], tf.int32)
        return {"image_data": image}, label

    dataset = tf.data.TFRecordDataset(path + r'/' + tfr_file_name)
    dataset = dataset.map(parser)

    def iterator(x):
        it = dataset.make_one_shot_iterator().get_next()
        sess = tf.Session()
        value = sess.run(it)
        if x:
            return value[0]
        else:
            return value[1]

    inputs = keras.Input(tensor=iterator(x=True), shape=input_shape)

    x = keras.applications.InceptionV3(include_top=False, weights='imagenet')(inputs)

    x = Flatten()(x)
    x = Dense(units=100000)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(units=10000)(x)
    x = Activation("tanh")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(units=1000)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(units=512)(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Dense(units=64)(x)
    x = Activation("tanh")(x)
    x = BatchNormalization()(x)

    x = Dense(units=classes)(x)
    x = Activation("softmax")(x)

    model = keras.Model(inputs, x, name="model")
    adam = keras.optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'], target_tensors=iterator(x=False))
    model.summary()

    print("Model Construction Complete")
    return model


def train(model, dataset, epoch, batch_size, validation_split):
    cp = tf.keras.callbacks.ModelCheckpoint(filepath=best_ckpt_path, monitor='val_dice_loss', save_best_only=True,
                                            verbose=1)
    model.fit(dataset, epochs=epoch, validation_split=validation_split, callbacks=[cp], batch_size=batch_size)


path_to_tfrecord(path=dataset_path, save_path=tfrecord_path, subfolder_list=dataset_category)
GETC = build(path=tfrecord_path, width=600, height=400, depth=3, classes=len(dataset_category))
train(model=GETC, dataset=tfr_dataset, epoch=EPOCH, batch_size=100, validation_split=0.1)
