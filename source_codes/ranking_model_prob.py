from keras import Model, Input, layers, Sequential
from keras.applications import VGG19
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Subtract, Activation
from keras.optimizers import Adam, SGD, RMSprop


def create_ranking_network(dense_units, dropout_rate, activation, img_size):
    """
    Create  ranking network which gives a score to an image.

    :param img_size:
    :type img_size: int
    :param dense_units:
    :type dense_units: int
    :param dropout_rate:
    :type dropout_rate: float
    :param activation:
    :type activation: str
    :return: ranking network
    :rtype: keras.Model
    """
    # Create feature extractor from VGG19
    feature_extractor = VGG19(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    for layer in feature_extractor.layers[:-4]:
        layer.trainable = False

    # Add dense layers on top of the feature extractor
    inp = Input(shape=(img_size, img_size, 3), name='input_image')
    base = feature_extractor(inp)
    base = Flatten(name='Flatten')(base)

    # Block 1
    base = Dense(dense_units, activation=activation, name='Dense_1')(base)
    base = BatchNormalization(name='BN1')(base)
    base = Dropout(dropout_rate, name='Drop_1')(base)

    # Block 2
    base = Dense(dense_units, activation=activation, name='Dense_2')(base)
    base = BatchNormalization(name='BN2')(base)
    base = Dropout(dropout_rate, name='Drop_2')(base)

    # Final dense
    base = Dense(1, name="Dense_Output")(base)
    base_network = Model(inp, base, name='Scoring_model')
    return base_network


def create_meta_network(
        dense_units,
        dropout_rate,
        learning_rate,
        optimizer,
        activation,
        learning_rate_decay,
        img_size,
        data_aug=True,
        weights=None
):
    """
    Create meta network which is used to teach the ranking network.

    :param img_size: dimension of input images during training.
    :type img_size: tuple(int)
    :param dense_units: number of units in the dense layer
    :type dense_units: int
    :param dropout_rate: dropout rate
    :type dropout_rate: float
    :param learning_rate: learning rate for the optimizer
    :type learning_rate: float
    :param optimizer: optimizer for model training
    :type optimizer: str
    :param activation: activation function for model layers
    :type activation: str
    :param learning_rate_decay: learning rate decay rate
    :type learning_rate_decay: float
    :param weights: path to the weights use for initialization
    :type weights: str
    :return: meta network model
    :rtype: keras.Model
    """
    # Create the two input branches
    input_left = Input(shape=(img_size, img_size, 3), name='left_input')
    input_right = Input(shape=(img_size, img_size, 3), name='right_input')

    if data_aug:
        # Augment data with linear transformation
        data_augmentation = Sequential(
            [
                layers.RandomTranslation(0.2, 0.2, fill_mode="reflect", interpolation="bilinear", ),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.2)
            ]
        )
        input_left = data_augmentation(input_left)
        input_right = data_augmentation(input_right)

    base_network = create_ranking_network(img_size, dense_units, dropout_rate, activation)
    left_score = base_network(input_left)
    right_score = base_network(input_right)

    # Subtract scores
    diff = Subtract()([left_score, right_score])

    # Pass difference through sigmoid function.
    prob = Activation("sigmoid", name="Activation_sigmoid")(diff)
    model = Model(inputs=[input_left, input_right], outputs=prob, name="Meta_Model")

    if weights:
        print('Loading weights ...')
        model.load_weights(weights)

    if optimizer == "adam":
        optimizer = Adam(learning_rate=learning_rate, decay=learning_rate_decay)
    elif optimizer == "sgd":
        optimizer = SGD(learning_rate=learning_rate, decay=learning_rate_decay)
    elif optimizer == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate, decay=learning_rate_decay)
    else:
        raise ValueError("Unsupported optimizer: " + optimizer)

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])

    return model
