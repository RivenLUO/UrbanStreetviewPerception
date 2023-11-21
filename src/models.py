from keras import Model, Input, Sequential
from keras.layers import Dense, Flatten, Conv2D, Concatenate, BatchNormalization, Activation, Resizing, Subtract, \
    Dropout, concatenate
from keras.applications import VGG19, vgg19, VGG16, vgg16
from keras_tuner import HyperModel
from keras.optimizers import Adam
from keras import backend as K


class SiameseRankingNetwork:
    "This is a Siamese network for training a ranking network. \
    It is a modified version of RSSCNN (Dubey et al., 2016)."
    def __init__(
            self,
            img_size=224,
            use_backbone='vgg19',
            ft_top=True,
            conv2d_num_layers=3,
            conv2d_num_filters=512,
            dense_num_layers=2,
            dense_num_units=4096,
            learning_rate=1e-5,
            loss_weights=0.5,
            loss_version='diff_sigmoid',
            augmentation_model=None,
    ):
        if not isinstance(augmentation_model, (Model, type(None))):
            raise ValueError(
                "Keyword augmentation_model should be "
                "a Keras `Model` or "
                f"empty. Received {augmentation_model}."
            )

        self.img_size = img_size
        self.use_backbone = use_backbone
        self.ft_top = ft_top
        self.conv2d_num_layers = conv2d_num_layers
        self.conv2d_num_filters = conv2d_num_filters
        self.dense_num_layers = dense_num_layers
        self.dense_num_units = dense_num_units
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.loss_version = loss_version
        self.augmentation_model = augmentation_model

    def build(self):
        input1 = Input(shape=(None, None, 3), name='Image1_Input')
        input2 = Input(shape=(None, None, 3), name='Image2_Input')

        if self.augmentation_model:
            input1 = self.augmentation_model(input1)
            input2 = self.augmentation_model(input2)

        # Backbone
        backbone = None
        preprocess_input = None
        if 'vgg19' == self.use_backbone:
            backbone = VGG19(
                input_shape=(self.img_size, self.img_size, 3),
                weights='imagenet',
                include_top=False,
            )
            if self.ft_top:
                for layer in backbone.layers[:-4]:
                    layer.trainable = False
            preprocess_input = vgg19.preprocess_input

        elif 'vgg16' == self.use_backbone:
            backbone = VGG16(
                input_shape=(self.img_size, self.img_size, 3),
                weights='imagenet',
                include_top=False,
            )
            if self.ft_top:
                for layer in backbone.layers[:-3]:
                    layer.trainable = False
            preprocess_input = vgg16.preprocess_input

        input1 = Resizing(height=self.img_size, width=self.img_size)(input1)
        input2 = Resizing(height=self.img_size, width=self.img_size)(input2)
        input1 = preprocess_input(input1)
        input2 = preprocess_input(input2)

        x1 = backbone(input1)
        x2 = backbone(input2)

        # Pairwise comparison subnet
        x = Concatenate(axis=-1)([x1, x2])
        # "Fusion"
        for _ in range(self.conv2d_num_layers):
            x = conv(x, num_filters=self.conv2d_num_filters)
        x = Flatten()(x)
        classification_output = Dense(units=1, activation='sigmoid', name='classification_output')(x)

        # Ranking subnet
        x1 = Flatten()(x1)
        for _ in range(self.dense_num_layers):
            x1 = dense(x1, units=self.dense_num_units)
        score1 = Dense(units=1, activation='linear')(x1)

        x2 = Flatten()(x2)
        for _ in range(self.dense_num_layers):
            x2 = dense(x2, units=self.dense_num_units)
        score2 = Dense(units=1, activation='linear')(x2)

        ranking_output = concatenate([score1, score2], axis=-1, name='ranking_output')
        if self.loss_version == 'diff_sigmoid':
            diff = Subtract()([score1, score2])
            ranking_output = Dense(units=1, activation='sigmoid', name='ranking_output')(diff)

        model = Model(inputs=[input1, input2],
                      outputs={
                            'classification_output': classification_output,
                            'ranking_output': ranking_output
                      },
                      name=f'ModifiedRSSCNN_{self.use_backbone}')

        # Compile
        optimizer = Adam(learning_rate=self.learning_rate)

        # Loss function
        loss_weights = {
            'classification_output': self.loss_weights,
            'ranking_output': 1 - self.loss_weights
        }

        loss = {
            'classification_output': 'binary_crossentropy',
            'ranking_output': rank_svm_loss
        }
        if self.loss_version == 'diff_sigmoid':
            loss = {
                'classification_output': 'binary_crossentropy',
                'ranking_output': 'binary_crossentropy'
            }

        model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=["accuracy"],
        )

        return model


class HyperSiameseRankingNetwork(HyperModel):
    "A hypermodel for SiameseRankingNetwork."
    def __init__(
            self,
            img_size=224,
            augmentation_model=None,
            **kwargs
    ):
        if not isinstance(
                augmentation_model, (HyperModel, Model, type(None))
        ):
            raise ValueError(
                "Keyword augmentation_model should be "
                "a `HyperModel`, a Keras `Model` or "
                f"empty. Received {augmentation_model}."
            )

        self.img_size = img_size
        self.augmentation_model = augmentation_model
        super().__init__(**kwargs)

    def build(self, hp):
        input1 = Input(shape=(None, None, 3), name='Image1_Input')
        input2 = Input(shape=(None, None, 3), name='Image2_Input')

        if self.augmentation_model:
            if isinstance(self.augmentation_model, HyperModel):
                augmentation_model = self.augmentation_model.build(hp)
                input1 = augmentation_model(input1)
                input2 = augmentation_model(input2)
            elif isinstance(self.augmentation_model, Model):
                input1 = self.augmentation_model(input1)
                input2 = self.augmentation_model(input2)

        # Backbone
        backbone = None
        use_backbone = hp.Choice('backbone', ['vgg19', 'vgg16'], default='vgg19')
        fine_tuning_top = hp.Boolean('fine_tuning_top', default=True)
        if 'vgg19' == use_backbone:
            backbone = VGG19(
                input_shape=(self.img_size, self.img_size, 3),
                weights='imagenet',
                include_top=False,
            )

            if fine_tuning_top:
                for layer in backbone.layers[:-4]:
                    layer.trainable = False

            preprocess_input = vgg19.preprocess_input
            input1 = Resizing(height=self.img_size, width=self.img_size)(input1)
            input2 = Resizing(height=self.img_size, width=self.img_size)(input2)
            input1 = preprocess_input(input1)
            input2 = preprocess_input(input2)

        elif 'vgg16' == use_backbone:
            backbone = VGG16(
                input_shape=(self.img_size, self.img_size, 3),
                weights='imagenet',
                include_top=False,
            )
            if fine_tuning_top:
                for layer in backbone.layers[:-3]:
                    layer.trainable = False

            preprocess_input = vgg16.preprocess_input
            input1 = Resizing(height=self.img_size, width=self.img_size)(input1)
            input2 = Resizing(height=self.img_size, width=self.img_size)(input2)
            input1 = preprocess_input(input1)
            input2 = preprocess_input(input2)

        x1 = backbone(input1)
        x2 = backbone(input2)

        # Pairwise comparison subnet
        x = Concatenate(axis=-1)([x1, x2])
        conv2d_num_layers = hp.Int('conv2d_conv_layers', min_value=2, max_value=4, step=1)
        conv2d_num_filters = hp.Int('conv2d_num_filters', min_value=64, max_value=512, step=2, sampling='log')
        # "Fusion"
        for _ in range(conv2d_num_layers):
            x = conv(x, num_filters=conv2d_num_filters)
        x = Flatten()(x)
        classification_output = Dense(units=1, activation='tanh', name='classification_output')(x)

        # Ranking subnet (DenseNet + Linear)
        dense_num_layers = hp.Int('dense_num_layers', min_value=1, max_value=5, step=1)
        dense_num_units = hp.Int('dense_num_units', min_value=64, max_value=512, step=2, sampling='log')
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        x1 = Flatten()(x1)
        for _ in range(dense_num_layers):
            x1 = dense(x, dense_num_units, batch_norm=True, dropout_rate=dropout_rate)
        score1 = Dense(units=1, activation='linear')(x1)

        x2 = Flatten()(x2)
        for _ in range(dense_num_layers):
            x2 = dense(x2, dense_num_units, batch_norm=True, dropout_rate=dropout_rate)
        score2 = Dense(units=1, activation='linear')(x2)

        loss_version = hp.Choice('loss_version', ['rank_svm', 'diff_sigmoid'], default='diff_sigmoid')

        ranking_output = concatenate([score1, score2], axis=-1, name='ranking_output')
        if loss_version == 'diff_sigmoid':
            diff = Subtract()([score1, score2])
            ranking_output = Dense(units=1, activation='sigmoid', name='ranking_output')(diff)

        model = Model(inputs=[input1, input2],
                      outputs={
                            'classification_output': classification_output,
                            'ranking_output': ranking_output
                      },
                      name=f'ModifiedRSSCNN_{use_backbone}')

        # Compile
        learning_rate = hp.Choice("learning_rate", [1e-3, 1e-4, 1e-5], default=1e-3)
        optimizer = Adam(learning_rate=learning_rate)

        # Loss function
        loss_weights = hp.Float('alpha', min_value=0.2, max_value=0.8, step=0.2)  # Weight of binary classification loss

        loss_weights = {
            'classification_output': loss_weights,
            'ranking_output': 1 - loss_weights
        }

        loss = {
            'classification_output': 'binary_crossentropy',
            'ranking_output': rank_svm_loss
        }
        if loss_version == 'diff_sigmoid':
            loss = {
                'classification_output': 'binary_crossentropy',
                'ranking_output': 'binary_crossentropy'
            }

        model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=["accuracy"],
        )

        return model


def rank_svm_loss(y_true, y_pred):
    # convert labels from {0, 1} to {-1, 1}
    y_true = 2 * y_true - 1
    # hinge loss
    score1 = y_pred[:, 0]
    score2 = y_pred[:, 1]
    loss = \
        K.sum(
            K.square(
                K.maximum(0.0, y_true[:,] * (score1 - score2))
            )
        )
    return loss


def conv(x, num_filters, kernel_size=(3, 3), strides=(1, 1)):
    x = Conv2D(
        num_filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def dense(x, units, batch_norm=True, dropout_rate=0.0):
    x = Dense(units=units)(x)
    x = Activation('relu')(x)
    if batch_norm:
        x = BatchNormalization()(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x



