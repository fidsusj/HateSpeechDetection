""" LSTM classifier """

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, GlobalMaxPooling1D, Input
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class LSTMClassifier:
    """ LSTM classifier """

    def fit(self, X_train, y_train, X_val, y_val):
        """ LSTM fit """
        dataset_train = tf.data.Dataset.from_tensor_slices(
            (tf.cast(X_train, tf.string), tf.cast(y_train, tf.int32))
        )
        dataset_train = dataset_train.shuffle(10000).batch(16)

        dataset_val = tf.data.Dataset.from_tensor_slices(
            (tf.cast(X_val, tf.string), tf.cast(y_val, tf.int32))
        )
        dataset_val = dataset_val.batch(16)

        encoder = TextVectorization(
            max_tokens=10000, output_mode="int", output_sequence_length=128
        )
        dataset_train_features = dataset_train.map(lambda features, label: features)
        encoder.adapt(dataset_train_features)

        vocab = np.array(encoder.get_vocabulary())
        embedding_dim = 64
        vocab_length = len(vocab)
        print(vocab_length)

        x_in = Input(shape=(1,), dtype="string")
        x = encoder(x_in)
        x = Embedding(
            input_dim=vocab_length,
            output_dim=embedding_dim,
            embeddings_initializer="uniform",
        )(x)
        x = LSTM(units=32, return_sequences=True)(x)
        x = GlobalMaxPooling1D()(x)
        x_out = Dense(1, activation="sigmoid")(x)

        lstm_model = tf.keras.models.Model(
            inputs=x_in, outputs=x_out, name="lstm_model"
        )
        lstm_model.summary()

        lstm_model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
        )
        history_lstm = lstm_model.fit(
            dataset_train, epochs=4, validation_data=dataset_val
        )
        self.model = lstm_model

        def plot_graphs(history, metric):
            plt.plot(history.history[metric])
            plt.plot(history.history["val_" + metric], "")
            plt.xlabel("Epochs")
            plt.ylabel(metric)
            plt.legend([metric, "val_" + metric])

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plot_graphs(history_lstm, "accuracy")
        plt.ylim(None, 1)
        plt.subplot(1, 2, 2)
        plot_graphs(history_lstm, "loss")
        plt.ylim(0, None)
        plt.show()

    def predict(self, X_test):
        """ predict """
        y_hat = self.model.predict(X_test)
        # TODO: ok like this? continuous -> class labels
        return np.around(y_hat)
