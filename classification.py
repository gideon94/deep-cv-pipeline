# file used for mobilenet model training and prdiction

from datetime import datetime

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications import MobileNet, imagenet_utils
from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.layers.core import Activation, Dense
from keras.metrics import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


class CarClassifier():
    def __init__(self, train=False):
        if not train:
            self.session = tf.Session()
            keras.backend.set_session(self.session)
            self.model = keras.models.load_model('my_model')
            self.model._make_predict_function()

    def train(self):
        batch_size = 32
        mobilenet = MobileNet(weights='imagenet', include_top=False)
        new_model = mobilenet.output
        new_model = GlobalAveragePooling2D()(new_model)
        new_model = Dense(128, activation='relu',
                          name='relu_layer_3')(new_model)
        new_model = Dropout(0.25)(new_model)
        predictions = Dense(2, activation='softmax', name='output')(new_model)
        new_model = Model(inputs=mobilenet.input, outputs=predictions)
        #all mobile layers are frozen
        for layer in new_model.layers[:87]:
            layer.trainable = False
        for layer in new_model.layers[87:]:
            layer.trainable = True

        logdir = "logs/scalars/train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        train_data_gen = ImageDataGenerator(validation_split=0.2)
        train_gen = train_data_gen.flow_from_directory(
            './train',
            target_size=(224, 224), color_mode='rgb',
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            subset='training'
        )
        validation_gen = train_data_gen.flow_from_directory(
            './train',
            target_size=(224, 224), color_mode='rgb',
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
            subset='validation'
        )

        new_model.compile(
            optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(new_model.summary())
        new_model.fit_generator(generator=train_gen,
                                steps_per_epoch=train_gen.samples//batch_size,
                                validation_data=validation_gen,
                                validation_steps=validation_gen.samples//batch_size,
                                epochs=10,
                                callbacks=[tensorboard_callback]
                                )
        new_model.save_weights('my_weights')
        new_model.save('my_model')

    def predict(self, img):
        img_expanded = np.expand_dims(img, axis=0)
        session = self.session
        model = self.model
        with session.as_default():
            with session.graph.as_default():
                predictions = model.predict(img_expanded)
        return predictions

    def prepare_image(self, file):
        img_path = ''
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
        return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

    def test(self):
        new_model = keras.models.load_model('my_model')
        test_data_gen = ImageDataGenerator(validation_split=0.2)
        test_gen = test_data_gen.flow_from_directory(
            './test',
            target_size=(224, 224), color_mode='rgb',
            batch_size=32,
            class_mode='categorical',
            shuffle=True,
        )
        loss, accuracy = new_model.evaluate(test_gen)


if __name__ == "__main__":
    # test()
    # predict()
    car_classifier = CarClassifier(True)
    car_classifier.train()
