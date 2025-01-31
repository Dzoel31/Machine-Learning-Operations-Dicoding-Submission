"""Transform module for Twitter Sentiment Analysis"""

import tensorflow as tf

LABEL_KEY = "sentiment"
FEATURE_KEY = "sentence"


def transformed_name(key):
    """Renaming transformed features

    Args:
        key: str, feature key

    Returns:
        str, transformed feature key
    """
    return key + "_xf"


def preprocessing_fn(inputs):
    """Implement transormation for the input features

    Args:
        inputs: map from feature keys to raw features.

    Returns:
        outputs: map from feature keys to transformed features.
    """

    outputs = inputs.copy()

    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(
        inputs[FEATURE_KEY])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
