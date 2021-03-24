import tensorflow as tf  # ! TensorFlow 2.0 !


class MalwareDetectionModel:
    def __init__(self, input_dim, output_dim, params):
        self.parameters = params
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self):
        graph = tf.keras.Sequential()

        # Input layer
        graph.add(
            tf.keras.layers.Bidirectional(
                self.base_layer(True),
                input_shape=self.input_dim)
        )

        # Hidden layers
        for i in range(self.parameters.hidden_count):
            graph.add(
                tf.keras.layers.Bidirectional(
                    self.base_layer(True if i < self.parameters.hidden_count - 1 else False)))

        # Output layer
        graph.add(
            tf.keras.layers.Dense(
                self.output_dim,
                activation="sigmoid",
            )
        )

        # Oh my, thank you Guido for this nice looking switch statement ! Yikes.
        # Optimizer
        if self.parameters.optimizer == 0:
            optimizer = tf.optimizers.SGD(learning_rate=self.parameters.learning_rate)
        elif self.parameters.optimizer == 1:
            optimizer = tf.optimizers.Adam(learning_rate=self.parameters.learning_rate)
        elif self.parameters.optimizer == 2:
            optimizer = tf.optimizers.Adadelta(learning_rate=self.parameters.learning_rate)
        elif self.parameters.optimizer == 3:
            optimizer = tf.optimizers.RMSprop(learning_rate=self.parameters.learning_rate)
        else:
            raise KeyError("Wrong value for graph parameter optimizer : " + str(self.parameters.optimizer))

        graph.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=['accuracy']
        )

        return graph

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