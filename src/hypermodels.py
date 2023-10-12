from keras import Model, Input, layers, Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, Subtract, Activation
from keras.applications import VGG19, vgg19
from keras.optimizers import Adam, SGD, RMSprop


def create_simple_MLP(x, num_layers, dense_units_list, dense_activation_list, dropout_rate_list):
    if len(x.shape) > 2:
        x = Flatten(name='Flatten')(x)

    def add_dense_block(x, dense_units, dense_activation, dropout_rate, name):
        # Dense block
        x = Dense(units=dense_units,
                  activation=dense_activation,
                  name=name)(x)

        x = BatchNormalization(name=f'{name}_BN')(x)

        x = Dropout(rate=dropout_rate,
                    name=f'{name}_Drop')(x)
        return x

    # Add dense blocks
    for i in range(num_layers):  # num_dense_layers
        # create a dict of hyperparameters
        dense_name = f'Dense_{i + 1}'
        dense_units = dense_units_list[i]
        dense_activation = dense_activation_list[i]
        dropout_rate = dropout_rate_list[i]
        x = add_dense_block(x, dense_units, dense_activation, dropout_rate, dense_name)

    return x


def create_feature_extractor(backbone, **kwargs):
    backbones = {
        'vgg19': VGG19(weights=kwargs['pretrained_weights'],
                       include_top=False,
                       input_shape=kwargs['input_shape']),
    }

    feature_extractor = backbones[backbone]

    # Freeze layers
    for layer in feature_extractor.layers[:-kwargs['unfreeze_layers']]:  # unfreeze_layers range from 0 to 4
        layer.trainable = False

    return feature_extractor


def create_ranking_network(img_size,
                           feature_extractor,
                           num_dense_layers,
                           dense_units_list,
                           dense_activation_list,
                           dropout_rate_list,
                           score_activation,
                           weights=None,
                           **feature_extractor_kwargs):
    feature_extractor = create_feature_extractor(feature_extractor, input_shape=(img_size, img_size, 3),
                                                 **feature_extractor_kwargs)

    # Add dense layers on top of the feature extractor
    inputs = Input(shape=(img_size, img_size, 3), name='Input_Image')
    x = feature_extractor(inputs)
    x = Flatten(name='Flatten')(x)

    # Create dense blocks
    x = create_simple_MLP(x, num_dense_layers, dense_units_list, dense_activation_list, dropout_rate_list)

    # Final dense
    output = Dense(units=1,
                   activation=score_activation,
                   name="Ranking_Score")(x)

    model = Model(inputs, output, name='Ranking_Network')

    if weights:
        model.load_weights(weights)

    return model


def create_siamese_network(img_size, image_preprocessing_layers, image_augmentation_layers,
                           feature_extractor, num_dense_layers, dense_units_list, dense_activation_list,
                           dropout_rate_list,
                           ranking_score_activation, final_activation,
                           optimizer,
                           learning_rate,
                           weights=None,
                           **feature_extractor_kwargs):
    # Create the two input branches
    image1 = Input(shape=(img_size, img_size, 3), name='Image1_Input')
    image2 = Input(shape=(img_size, img_size, 3), name='Image2_Input')

    # Preprocess input images
    if feature_extractor == 'vgg19':
        # (RGB -> BGR, then zero-center each color channel with respect to the ImageNet dataset, without scaling)
        image1 = vgg19.preprocess_input(image1)
        image2 = vgg19.preprocess_input(image2)

    if not image_preprocessing_layers:
        raise Warning("No image preprocessing layers specified. Model building will possibly fail! \n"
                      "For this input: The sizes of image1 and image2 should be identical! \n"
                      "Consider adding image preprocessing layers with Keras API.")
    else:
        image1 = image_preprocessing_layers(image1)
        image2 = image_preprocessing_layers(image2)

    # Augment input images
    if not image_augmentation_layers:
        raise Warning("No image augmentation layers specified. Model training results would be affected. \n"
                      "Consider adding image augmentation layers with Keras API.")
    else:
        image1 = image_augmentation_layers(image1)
        image2 = image_augmentation_layers(image2)

    img_size = image1.shape[1]

    ranking_network = create_ranking_network(img_size=img_size,
                                             feature_extractor=feature_extractor,
                                             num_dense_layers=num_dense_layers,
                                             dense_units_list=dense_units_list,
                                             dense_activation_list=dense_activation_list,
                                             dropout_rate_list=dropout_rate_list,
                                             score_activation=ranking_score_activation,
                                             weights=None,
                                             **feature_extractor_kwargs)

    score1 = ranking_network(image1)
    score2 = ranking_network(image2)

    # Subtract scores
    diff = Subtract()([score1, score2])

    # Pass difference through activation function.
    final_activations = {
        'softmax': layers.Activation(activation='softmax'),  # Discrete binary classification
        'sigmoid': layers.Activation(activation='sigmoid')  # Continuous binary classification
    }
    output = Activation(activation=final_activations[final_activation],
                        name=f'Final_Activation_{final_activation}')(diff)

    model = Model(inputs=[image1, image2], outputs=output, name='Siamese_Ranking_Network')

    # Compile model
    optimizers = {
        'adam': Adam(learning_rate=learning_rate),
        'sgd': SGD(learning_rate=1e-4),
        'rmsprop': RMSprop(learning_rate=1e-4)
    }
    model.compile(optimizer=optimizers[optimizer],
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    if weights:
        model.load_weights(weights)

    return model

# def build_model(hp):
#     # Hyperparameters of model architecture
#     img_size = 224
#
#     num_dense_layers = hp.Int('num_dense_layers', min_value=2, max_value=3, step=1)
#
#     dense_units_list = [hp.Int(f'dense_units_{i + 1}', min_value=32, max_value=96, step=16, default=64)
#                         for i in range(num_dense_layers)]
#
#     dense_activation_list = [hp.Choice(f'dense_activation_{i + 1}', values=['relu', 'tanh', 'sigmoid', 'linear'],default='relu')
#                              for i in range(num_dense_layers)]
#
#     dropout_rate_list = [hp.Float(f'dropout_rate_{i + 1}', min_value=0.1, max_value=0.5, step=0.1, default=0.3)
#                          for i in range(num_dense_layers)]
#
#     score_activation = hp.Choice('score_activation', values=['relu', 'tanh', 'sigmoid', 'linear'])
#     unfreeze_layers = hp.Int('unfreeze_layers', min_value=0, max_value=4, step=1, default=4)
#     final_activation = hp.Choice('final_activation', values=['tanh', 'sigmoid'], default='sigmoid')
#
#     meta_model = create_meta_network(img_size, num_dense_layers, dense_units_list, dense_activation_list,
#                                      dropout_rate_list,
#                                      score_activation, unfreeze_layers, final_activation, data_aug=True, weights=None)
#
#     optimizers = {
#         'adam': Adam(learning_rate=hp.Choice('learning_rate', values=[1e-4, 1e-5, 1e-6])),
#         'sgd': SGD(learning_rate=hp.Choice('learning_rate', values=[1e-4, 1e-5, 1e-6])),
#         'rmsprop': RMSprop(learning_rate=hp.Choice('learning_rate', values=[1e-4, 1e-5, 1e-6]))
#     }
#
#     meta_model.compile(optimizer=optimizers[hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])],
#                        loss='binary_crossentropy',
#                        metrics=['accuracy'])
#
#     return meta_model
