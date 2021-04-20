import abc
from abc import abstractmethod
import math
from parameters import Parameters
import tensorflow as tf
from tensorflow.keras.layers import *
import json
from dataset import Dataset


metrics = [
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.Accuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.MeanIoU(num_classes=2, name='iou'),
    tf.keras.metrics.BinaryAccuracy(name='bin_accuracy'),
]


class Model:
    __metaclass__ = abc.ABCMeta

    def __init__(self, params):
        self.parameters = params
        self.graph = None
        self.history = None

    @abstractmethod
    def build(self, optimizer):
        pass

    def train(self, dataset : Dataset, save_history=True):
        self.history = self.graph.fit(
            dataset.get_train_ds().__iter__(),
            steps_per_epoch=math.ceil(dataset.train_set_size() / self.parameters.batch_size),
            epochs=self.parameters.epochs,
            validation_data=dataset.get_val_ds().__iter__()
        )

        if save_history:
            with open(f'histories/{self.parameters.name}.json', 'w') as file:
                json.dump(self.history.history, file)

    def test(self, x_test, y_test):
        pass  # Do not implement/use before testing all architectures


class MalwareDetectionModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def build(self, optimizer):
        self.graph = tf.keras.Sequential()

        # Input layer
        self.graph.add(
            tf.keras.layers.Bidirectional(
                self.base_layer(True),
                input_shape=self.parameters.input_dim)
        )

        # Hidden layers
        for i in range(self.parameters.hidden_count):
            self.graph.add(
                tf.keras.layers.Bidirectional(
                    self.base_layer(True if i < self.parameters.hidden_count - 1 else False)))

        # Output layer
        self.graph.add(
            tf.keras.layers.Dense(
                2,
                activation="sigmoid",
            )
        )

        self.graph.compile(
            optimizer=optimizer(learning_rate=self.parameters.learning_rate),
            loss=self.parameters.loss,
            metrics=metrics
        )

    def base_layer(self, return_sequences):
        return tf.keras.layers.GRU(
            self.parameters.units,
            return_sequences=return_sequences,
            dropout=self.parameters.dropout,
            recurrent_dropout=self.parameters.recurrent_dropout,
            kernel_initializer="lecun_uniform",
            recurrent_initializer="lecun_uniform",
            bias_regularizer=tf.keras.regularizers.l1_l2(
                l1=self.parameters.bias_l1,
                l2=self.parameters.bias_l2),
            recurrent_regularizer=tf.keras.regularizers.l1_l2(
                l1=self.parameters.recurrent_l1,
                l2=self.parameters.recurrent_l2)
        )


class UNetModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def build(self, optimizer):
        """
        Model found on:
        https://github.com/AlirezaShamsoshoara/Fire-Detection-UAV-Aerial-Image-Classification-Segmentation-UnmannedAerialVehicle
        This function returns a U-Net Model for this binary fire segmentation images:
        Arxiv Link for U-Net: https://arxiv.org/abs/1505.04597
        """
        inputs = Input((self.parameters.input_dim[0], self.parameters.input_dim[1], 3))
        s = Lambda(lambda x: x / 255)(inputs)

        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        self.graph = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        self.graph.compile(
            optimizer=optimizer(learning_rate=self.parameters.learning_rate),
            loss=self.parameters.loss,
            metrics=metrics
        )
