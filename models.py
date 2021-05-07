import abc
from abc import abstractmethod
from parameters import Parameters
import tensorflow as tf
from tensorflow.keras.layers import *
import json
from dataset import Dataset
import numpy as np
import cv2
from generator import Generator
import os
import seaborn as sns
import matplotlib.pyplot as plt
from deeplabv3plus import DeeplabV3Plus

metrics = [
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.MeanIoU(num_classes=2, name='iou'),
    tf.keras.metrics.BinaryAccuracy(name='bin_accuracy'),
    tf.keras.metrics.MeanSquaredError(name="mse")
]


class Model:
    __metaclass__ = abc.ABCMeta

    def _learning_schedule(self, generator : Generator):
        return tf.keras.optimizers.schedules.ExponentialDecay(
            self.parameters.learning_rate,
            decay_steps=len(generator),
            decay_rate=0.96,
            staircase=True)

    def __init__(self, params : Parameters):
        self.parameters = params
        self.graph = None
        self.history = None

    @abstractmethod
    def _build(self, generator : Generator):
        pass

    def train(self, dataset : Dataset, save_history=True, save_model=True, include_val=False):
        if not include_val:
            self._build(dataset.get_train_gen())
            self.history = self.graph.fit(
                dataset.get_train_gen(), # Train split only
                batch_size=self.parameters.batch_size,
                epochs=self.parameters.epochs,
                validation_data=dataset.get_val_gen()
            )
        else:
            self._build(dataset.get_train_val_gen())
            self.history = self.graph.fit(
                dataset.get_train_val_gen(), # Train + val split
                batch_size=self.parameters.batch_size,
                epochs=self.parameters.epochs,
            )

        if save_history:
            with open(f'histories/{self.parameters.name}.json', 'w') as file:
                json.dump(self.history.history, file)

        if save_model:
            self.graph.save(f"models/{self.parameters.name}")

    def load_trained(self):
        self.graph = tf.keras.models.load_model(f"models/{self.parameters.name}")

    def test(self, dataset : Dataset):
        results = self.graph.evaluate(
            dataset.get_test_gen()
        )

        # Construct dict from result list
        output = {}
        for i, key in enumerate(self.graph.metrics_names):
                output[key] = results[i]

        # Dump it
        with open(f'tests/{self.parameters.name}.json', 'w') as file:
            json.dump(output, file)

    def prediction_test(self, dataset : Dataset):
        ids = [
            87648,
            87586
        ]

        ids.extend([*range(87603, 87638)]) # Picture 002_rgb.png

        results = self.graph.predict(
            dataset.load_specific_ids(ids)
        )

        if not os.path.isdir(f"predictions/{self.parameters.name}"):
            os.mkdir(f"predictions/{self.parameters.name}")

        for i, pred in enumerate(results):
            plt.figure()
            sns.heatmap(pred[:, :, 0])
            plt.tight_layout()
            plt.savefig(f"predictions/{self.parameters.name}/{ids[i]}_heatmap.png")
            plt.close()

        preds_val_t = (results > 0.5).astype(np.uint8)
        for i, pred in enumerate(preds_val_t):
            cv2.imwrite(f"predictions/{self.parameters.name}/{ids[i]}_pred_gt.png", pred)

        
        for i, sample in enumerate(dataset.load_specific_ids(ids)):
            cv2.imwrite(f"predictions/{self.parameters.name}/{ids[i]}_rgb.png", sample)

class UNetModel(Model):
    def __init__(self, params):
        super().__init__(params)

    def _build(self, generator : Generator):
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_schedule(generator)),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=metrics
        )
        


class DeepLabV3Plus(Model):
    def __init__(self, params: Parameters):
        super().__init__(params)

    def _build(self, generator: Generator):
        """
        Model found on:
        https://github.com/lattice-ai/DeepLabV3-Plus
        """
        self.graph = DeeplabV3Plus(
            num_classes=2,
            backbone='resnet50'
        )

        self.graph.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.parameters.learning_rate
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=metrics
        )

        self.graph.build(
            (None, self.parameters.input_dim[0], self.parameters.input_dim[1], 3)
        )

        self.graph.summary()