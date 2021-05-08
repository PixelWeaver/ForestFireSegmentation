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
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50

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
            self._summarize()
            self.history = self.graph.fit(
                dataset.get_train_gen(), # Train split only
                batch_size=self.parameters.batch_size,
                epochs=self.parameters.epochs,
                validation_data=dataset.get_val_gen()
            )
        else:
            self._build(dataset.get_train_val_gen())
            self._summarize()
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

    def _summarize(self):
        input_shape = (1, self.parameters.input_dim[0], self.parameters.input_dim[1], 3)
        input_tensor = tf.random.normal(input_shape)
        self.graph(input_tensor) # Run just one random sample through it to make sure it has been built before printing network summary
        self.graph.summary()

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

    @staticmethod
    def _upsample(tensor, size):
        '''bilinear upsampling'''
        name = tensor.name.split('/')[0] + '_upsample'

        def bilinear_upsample(x, size):
            resized = tf.image.resize(
                images=x, size=size)
            return resized
        y = Lambda(lambda x: bilinear_upsample(x, size),
                output_shape=size, name=name)(tensor)
        return y

    @staticmethod
    def _ASPP(tensor):
        '''atrous spatial pyramid pooling'''
        dims = K.int_shape(tensor)

        y_pool = AveragePooling2D(pool_size=(
            dims[1], dims[2]), name='average_pooling')(tensor)
        y_pool = Conv2D(filters=256, kernel_size=1, padding='same',
                        kernel_initializer='he_normal', name='pool_1x1conv2d', use_bias=False)(y_pool)
        y_pool = BatchNormalization(name=f'bn_1')(y_pool)
        y_pool = Activation('relu', name=f'relu_1')(y_pool)

        y_pool = DeepLabV3Plus._upsample(tensor=y_pool, size=[dims[1], dims[2]])

        y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                    kernel_initializer='he_normal', name='ASPP_conv2d_d1', use_bias=False)(tensor)
        y_1 = BatchNormalization(name=f'bn_2')(y_1)
        y_1 = Activation('relu', name=f'relu_2')(y_1)

        y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same',
                    kernel_initializer='he_normal', name='ASPP_conv2d_d6', use_bias=False)(tensor)
        y_6 = BatchNormalization(name=f'bn_3')(y_6)
        y_6 = Activation('relu', name=f'relu_3')(y_6)

        y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same',
                    kernel_initializer='he_normal', name='ASPP_conv2d_d12', use_bias=False)(tensor)
        y_12 = BatchNormalization(name=f'bn_4')(y_12)
        y_12 = Activation('relu', name=f'relu_4')(y_12)

        y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same',
                    kernel_initializer='he_normal', name='ASPP_conv2d_d18', use_bias=False)(tensor)
        y_18 = BatchNormalization(name=f'bn_5')(y_18)
        y_18 = Activation('relu', name=f'relu_5')(y_18)

        y = concatenate([y_pool, y_1, y_6, y_12, y_18], name='ASPP_concat')

        y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same',
                kernel_initializer='he_normal', name='ASPP_conv2d_final', use_bias=False)(y)
        y = BatchNormalization(name=f'bn_final')(y)
        y = Activation('relu', name=f'relu_final')(y)
        return y

    def _build(self, generator: Generator):
        """
        Model found on:
        https://github.com/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow2.0
        """
        base_model = ResNet50(input_shape=(
            self.parameters.input_dim[0], self.parameters.input_dim[1], 3), weights='imagenet', include_top=False)

        image_features = base_model.get_layer('conv5_block3_out').output
        x_a = DeepLabV3Plus._ASPP(image_features)
        x_a = DeepLabV3Plus._upsample(tensor=x_a, size=[self.parameters.input_dim[0] // 4, self.parameters.input_dim[1] // 4])

        x_b = base_model.get_layer('conv2_block3_out').output
        x_b = Conv2D(filters=48, kernel_size=1, padding='same',
                    kernel_initializer='he_normal', name='low_level_projection', use_bias=False)(x_b)
        x_b = BatchNormalization(name=f'bn_low_level_projection')(x_b)
        x_b = Activation('relu', name='low_level_activation')(x_b)

        x = concatenate([x_a, x_b], name='decoder_concat')

        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                kernel_initializer='he_normal', name='decoder_conv2d_1', use_bias=False)(x)
        x = BatchNormalization(name=f'bn_decoder_1')(x)
        x = Activation('relu', name='activation_decoder_1')(x)

        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu',
                kernel_initializer='he_normal', name='decoder_conv2d_2', use_bias=False)(x)
        x = BatchNormalization(name=f'bn_decoder_2')(x)
        x = Activation('relu', name='activation_decoder_2')(x)
        x = DeepLabV3Plus._upsample(x, [self.parameters.input_dim[0], self.parameters.input_dim[1]])

        x = Conv2D(1, (1, 1), name='output_layer')(x)

        self.graph = tf.keras.Model(inputs=base_model.input, outputs=x, name='DeepLabV3_Plus')

        for layer in self.graph.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = 0.9997
                layer.epsilon = 1e-5
            elif isinstance(layer, tf.keras.layers.Conv2D):
                layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)

        self.graph.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_schedule(generator)),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=[  
                tf.keras.metrics.BinaryAccuracy(name='bin_accuracy'),
                tf.keras.metrics.MeanIoU(num_classes=2, name='iou'),
            ])