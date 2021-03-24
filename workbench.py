import os
import tensorflow as tf  # ! Tensorflow 2.0 !
import numpy as np
import time
import gc
from sklearn.metrics import f1_score, accuracy_score
from parameters import Parameters
from dataset import Dataset
from model import Model
from metric_writer import MetricWriter

# Just disables the warning, doesn't enable AVX/FMA
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Workbench:
    def __init__(self):
        self.dataset = Dataset()
        self.parameters = None
        self.best_parameters = None
        self.best_accuracy = None
        self.model = None
        self.metric_writer = MetricWriter()
        if tf.test.is_gpu_available(cuda_only=True):
            print("CUDA GPU acceleration was found !")
        else:
            print("Could not find any CUDA GPU acceleration.")

    def random_search(self, iterations):
        for i in range(iterations):
            print("Random search : iteration", i+1, "/", iterations)
            # Generate random hyperparameters
            self.parameters = Parameters()

            # Initialize the model
            graph = Model(tuple(list(self.dataset.train_set_x.shape)[1:]), 1, self.parameters)
            self.model = graph.build()

            # Run the bloody thing
            self.train_n_test()

            # Clear up the mess
            #del self.model
            #gc.collect()

    def testing_run(self):
        print("Running test run...")
        # Generate random hyperparameters
        self.parameters = Parameters.from_file()

        # Initialize the model
        graph = Model(tuple(list(self.dataset.train_set_x.shape)[1:]), 1, self.parameters)
        self.model = graph.build()

        # Run the bloody thing
        metrics = {}
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0.01, patience=10)

        # Train
        train_start = time.time()
        history = self.model.fit(
            self.dataset.train_set_x, self.dataset.train_set_y,
            batch_size=self.parameters.batch_size,
            epochs=self.parameters.epochs,
            shuffle=True,
            verbose=1
        )
        train_stop = time.time()

        # Test
        test_start = time.time()
        predictions = self.model.predict(
            self.dataset.test_set_x,
            batch_size=self.parameters.batch_size,
            verbose=1
        )

        test_end = time.time()

        # Record important stuff
        metrics["train_acc"] = history.history["accuracy"]
        metrics["predictions"] = [x[0] for x in predictions]
        metrics["test_acc"] = accuracy_score(self.dataset.test_set_y.flatten().tolist(),
                                             [np.round(x) for x in metrics["predictions"]])
        metrics["train_time"] = train_stop - train_start
        metrics["test_time"] = test_end - test_start
        metrics["f1_score"] = f1_score(self.dataset.test_set_y.flatten().tolist(),
                                       [np.round(x) for x in metrics["predictions"]])

        print(metrics)

        # Clear up the mess
        del self.model
        gc.collect()

    def train_n_test(self):
        metrics = {}
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', min_delta=0.01, patience=10)

        # Train
        train_start = time.time()
        history = self.model.fit(
            self.dataset.train_set_x, self.dataset.train_set_y,
            batch_size=self.parameters.batch_size,
            epochs=self.parameters.epochs,
            shuffle=True,
            verbose=0
        )
        train_stop = time.time()

        # Test
        test_start = time.time()
        predictions = self.model.predict(
            self.dataset.val_set_x,
            batch_size=self.parameters.batch_size,
            verbose=0
        )
        test_end = time.time()

        # Record important stuff
        metrics["train_acc"] = history.history["accuracy"]
        metrics["predictions"] = [x[0] for x in predictions]
        metrics["test_acc"] = accuracy_score(self.dataset.val_set_y.flatten().tolist(), [np.round(x) for x in metrics["predictions"]])

        print("    ---> achieved", metrics["test_acc"], "test accuracy")

        metrics["train_time"] = train_stop - train_start
        metrics["test_time"] = test_end - test_start
        metrics["f1_score"] = f1_score(self.dataset.val_set_y.flatten().tolist(), [np.round(x) for x in metrics["predictions"]])

        self.metric_writer.write(metrics, self.parameters)

        # Compare hyperparameters to best performance
        if self.best_accuracy is None or metrics["test_acc"] > self.best_accuracy:
            self.best_accuracy = metrics["test_acc"]
            self.best_parameters = self.parameters

            # Save best hyperparameters to disk
            self.best_parameters.to_file()
            print("    ---> new accuracy record !")
        else:
            print("    ---> best:", self.best_accuracy)