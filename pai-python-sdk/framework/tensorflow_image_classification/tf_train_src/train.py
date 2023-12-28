import tensorflow as tf
import argparse
import gzip
import os
import numpy as np


def load_dataset(data_path):
    image_path = os.path.join(data_path, "images.gz")
    label_path = os.path.join(data_path, "labels.gz")
    with gzip.open(label_path, "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.int8, offset=8)
    with gzip.open(image_path, "rb") as f:
        images = np.frombuffer(f.read(), dtype=np.int8, offset=16).reshape(
            len(labels), 28, 28, 1
        )
    return images, labels


def train(batch_size, epochs, train_data, test_data):

    # Load dataset from input channel 'train' and 'test'.
    train_images, train_labels = load_dataset(train_data)
    test_images, test_labels = load_dataset(test_data)

    # model train
    num_classes = 10
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                8, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        train_images,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        verbose=2,
    )

    # save model

    model_path = os.environ.get("PAI_OUTPUT_MODEL")
    model.save(model_path)

    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--train_data",
        default=os.environ.get("PAI_INPUT_TRAIN"),
        help="Path to train data (default: /ml/input/data/train/)",
    )
    parser.add_argument(
        "--test_data",
        default=os.environ.get("PAI_INPUT_TEST"),
        help="Path to test data (default: /ml/input/data/test/)",
    )

    args = parser.parse_args()

    train(args.batch_size, args.epochs, args.train_data, args.test_data)


if __name__ == "__main__":
    main()
