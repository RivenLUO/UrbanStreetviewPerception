import os
import keras_tuner as kt
from tensorflow import keras
import numpy as np
import csv
import json
from keras import Input, optimizers, regularizers, callbacks, models, Model, Sequential, metrics, backend as K, losses
from keras.applications import vgg19, xception, efficientnet, resnet, inception_resnet_v2, VGG19, Xception, \
    EfficientNetB7, ResNet152V2, InceptionResNetV2

def create_ranking_score(img_size):
    """

    :return:
    """
    # Create feature extractor from VGG19
    feature_extractor = VGG19(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    for layer in feature_extractor.layers[:-4]:
        layer.trainable = False

    # Add dense layers on top of the feature extractor
    input_img = Input(shape=(img_size, img_size, 3), name='input_image')
    base = feature_extractor(input_img)
    base = Flatten(name='Flatten')(base)

    # Block 1
    base = Dense(32, activation='relu', name='Dense_1')(base)
    base = BatchNormalization(name='BN1')(base)
    base = Dropout(0.490, name='Drop_1')(base)

    # Block 2
    base = Dense(128, activation='relu', name='Dense_2')(base)
    base = BatchNormalization(name='BN2')(base)
    base = Dropout(0.368, name='Drop_2')(base)

    # Final dense
    base = Dense(1, name="Dense_Output")(base)
    base_network = Model(inp, base, name='Scoring_model')
    return base_network

def create_meta_network(img_size, weights=None):
    """

    :param img_size:
    :param weights:
    :return:
    """
    # Create the two input branches
    input_left = Input(shape=(img_size, img_size, 3), name='left_input')
    input_right = Input(shape=(img_size, img_size, 3), name='right_input')
    base_network = create_ranking_network(img_size)
    left_score = base_network(input_left)
    right_score = base_network(input_right)

    # Subtract scores
    diff = Subtract()([left_score, right_score])

    # Pass difference through sigmoid function.
    prob = Activation("sigmoid", name="Activation_sigmoid")(diff)
    model = Model(inputs=[input_left, input_right], outputs= prob, name="Meta_Model")

    if weights:
        print('Loading weights ...')
        model.load_weights(weights)

    model.compile(optimizer=Adam(learning_rate=1e-6), loss="binary_crossentropy", metrics=['accuracy'])

    return model