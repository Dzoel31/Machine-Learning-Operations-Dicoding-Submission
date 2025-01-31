"""Module for training a sentiment analysis model using Twitter data."""

import os
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from sentence_sentiment_transform import (
    transformed_name,
    LABEL_KEY,
    FEATURE_KEY,
)


def gzip_reader_fn(filenames):
    """Loads compressed data

    Args:
        filenames: str, file path
    Returns:
        tf.data.TFRecordDataset, dataset
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(
    file_pattern, tf_transform_output, num_epochs, batch_size=32
) -> tf.data.Dataset:
    """Generates features and labels for training.
    Args:
        file_pattern: str, file pattern
        tf_transform_output: tft.TFTransformOutput, transform output
        num_epochs: int, number of epochs
        batch_size: int, representing the number of consecutive elements of
        returned dataset to combine in a single batch
    Returns:
        A dataset that contains (features, indices) tuple where features
        is a dictionary of Tensors, and indices is a single Tensor of
        label indices.
    """
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


VOCAB_SIZE = 500
SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 32
vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
)


def model_builder():
    """Build machine learning model"""

    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
    )
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    # print(model)
    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()

        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs) -> None:
    """Train the model based on the settings provided"""
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy", mode="max", verbose=1, patience=10
    )
    mc = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor="val_binary_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
    )

    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 32)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 32)
    vectorize_layer.adapt(
        [
            j[0].numpy()[0]
            for j in [i[0][transformed_name(FEATURE_KEY)] for i in list(train_set)]
        ]
    )

    # Build the model
    model = model_builder()

    # Train the model
    model.fit(
        x=train_set,
        validation_data=val_set,
        steps_per_epoch=fn_args.train_steps,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback, es, mc],
        epochs=10,
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }
    model.save(
        fn_args.serving_model_dir,
        save_format="tf",
        signatures=signatures)

    plot_model(
        model,
        to_file="images/model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
    )
