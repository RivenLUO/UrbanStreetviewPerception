from keras import Model, Input, layers, Sequential
from keras.applications import VGG19
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Subtract, Activation
from keras.optimizers import Adam, SGD, RMSprop
import random


supported_optimizer_dict = {
    "adam": Adam,
    "sgd": SGD,
    "rmsprop": RMSprop
}


def create_ranking_network(dense_units, dropout_rates, activation, img_size):
    """Create ranking network which gives a score to an image.

    :param dense_units:
    :param dropout_rates:
    :param activation:
    :param img_size:

    :return:
    """

    # Create feature extractor from VGG19
    feature_extractor = VGG19(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    for layer in feature_extractor.layers[:-4]:
        layer.trainable = False

    # Add dense layers on top of the feature extractor
    inputs = Input(shape=(img_size, img_size, 3), name='Input_Image')
    base = feature_extractor(inputs)
    base = Flatten(name='Flatten')(base)

    # Check if the number of dense units and dropout rates are the same
    if len(dense_units) != len(dropout_rates):
        raise ValueError("The number of dense units and dropout rates must be the same!")

    # Loop over the dense units and add the corresponding layers
    for block_num in range(1, len(dense_units) + 1):
        # Dense block
        base = Dense(dense_units[block_num - 1], activation=activation, name=f'Dense_{block_num}')(base)
        base = BatchNormalization(name=f'BN_{block_num}')(base)
        base = Dropout(dropout_rates[block_num - 1], name=f'Drop_{block_num}')(base)

    # # Block 1
    # base = Dense(dense_units_dic.get("dense_1"), activation=activation, name='Dense_1')(base)
    # base = BatchNormalization(name='BN1')(base)
    # base = Dropout(0.4, name='Drop_1')(base)
    #
    # # Block 2
    # base = Dense(dense_units_dic.get("dense_2"), activation=activation, name='Dense_2')(base)
    # base = BatchNormalization(name='BN2')(base)
    # base = Dropout(0.3, name='Drop_2')(base)

    # Final dense
    score = Dense(1, name="Dense_Output_Score")(base)
    ranking_network = Model(inputs, score, name='Ranking_Network')
    return ranking_network


def create_siamese_network(
        dense_units,
        dense_dropout_rates,
        dense_activation,
        activation,
        optimizer_name,
        learning_rate,
        learning_rate_decay,
        img_size=224,
        data_aug=True,
        weights=None
):
    """Create meta network which is used to teach the ranking network.

    :param dense_units:
    :param dense_dropout_rates:
    :param dense_activation:
    :param activation:
    :param optimizer_name:
    :param learning_rate:
    :param learning_rate_decay:
    :param img_size:
    :param data_aug:
    :param weights:

    :return:
    """

    # Create the two input branches
    left_image = Input(shape=(img_size, img_size, 3), name='left_image')
    right_image = Input(shape=(img_size, img_size, 3), name='right_image')

    # Add data augmentation if specified
    if data_aug:
        data_augmentation = Sequential(
            [
                layers.RandomTranslation(0.5, 0.5),
                layers.RandomFlip(),
                layers.RandomRotation(0.5),
                layers.RandomZoom((-0.5, 0.5), (-0.5, 0.5)),
                layers.RandomContrast(0.5),
                # layers.RandomBrightness((-0.5, 0.5))
            ]
        )

        left_image = data_augmentation(left_image)
        right_image = data_augmentation(right_image)

    base_network = create_ranking_network(dense_units, dense_dropout_rates, dense_activation, img_size)
    left_score = base_network(left_image)
    right_score = base_network(right_image)

    # Subtract scores
    diff = Subtract()([left_score, right_score])

    # Pass difference through sigmoid function.
    output_score = Activation(activation=activation, name=f'Activation_{activation}')(diff)
    model = Model(inputs=[left_image, right_image], outputs=output_score, name='Meta_Ranking_Model')

    # Select optimizer
    optimizer_class = supported_optimizer_dict.get(optimizer_name.lower())
    if optimizer_class is None:
        raise ValueError("Unsupported optimizer: " + optimizer_name)

    if learning_rate_decay is None:
        optimizer = optimizer_class(learning_rate=learning_rate)
    else:
        optimizer = optimizer_class(learning_rate=learning_rate, decay=learning_rate_decay)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Load weights if specified
    if weights:
        print('Loading weights ...')
        model.load_weights(weights)
        print('Weights loaded!')

    return model

