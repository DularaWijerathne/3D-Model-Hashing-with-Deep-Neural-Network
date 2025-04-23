from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from config import HASH_BIT_SIZE
from models.feature_extractor import create_point_cloud_feature_extractor


def create_hash_layer(features, name_suffix=""):
    """
    Create a hash layer that outputs binary-like codes with regularization
    """
    # Add a regularizer to encourage better hash code distribution
    hash_logits = layers.Dense(
        HASH_BIT_SIZE,
        name=f"hash_dense_{name_suffix}",
        kernel_regularizer=keras.regularizers.l2(1e-5),
        bias_regularizer=keras.regularizers.l2(1e-5),
    )(features)

    # Apply batch normalization before activation
    hash_norm = layers.BatchNormalization(name=f"hash_bn_{name_suffix}")(hash_logits)

    # Use tanh activation for binary-like outputs
    hash_codes = layers.Activation("tanh", name=f"hash_codes_{name_suffix}")(hash_norm)

    return hash_codes, hash_logits


def create_hashing_model():
    """
    Create a Siamese network for learning hash codes with enhanced architecture
    """
    # Feature extractor (shared weights)
    feature_extractor = create_point_cloud_feature_extractor()

    # Input for point cloud pairs
    input_a = keras.Input(shape=(None, 3), name="point_cloud_a")
    input_b = keras.Input(shape=(None, 3), name="point_cloud_b")

    # Extract features
    features_a = feature_extractor(input_a)
    features_b = feature_extractor(input_b)

    # Add dropout for regularization
    features_a = layers.Dropout(0.2)(features_a)
    features_b = layers.Dropout(0.2)(features_b)

    # Generate hash codes
    hash_a, hash_logits_a = create_hash_layer(features_a, "a")
    hash_b, hash_logits_b = create_hash_layer(features_b, "b")

    # Concatenate outputs for loss computation
    model_output = layers.Concatenate(name="concatenated_hashes")([hash_a, hash_b])

    # Create model
    model = keras.Model(inputs=[input_a, input_b], outputs=model_output)

    return model
