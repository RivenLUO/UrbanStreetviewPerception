import dataset
import os
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, Sequential

data_augmentation = Sequential(
    [
        # layers.RandomTranslation(0.5, 0.5),
        layers.RandomFlip(),
        layers.RandomRotation(0.4),
        layers.RandomZoom((-0.4, 0.4), (-0.4, 0.4)),
        layers.RandomContrast(0.7),
        layers.RandomBrightness(0.4, value_range=[0.0, 1.0])
    ]
)

