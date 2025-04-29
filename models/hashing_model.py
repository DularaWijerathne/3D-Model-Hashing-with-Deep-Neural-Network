from tensorflow import keras
from tensorflow.keras import layers, initializers
import tensorflow as tf

from config import HASH_BIT_SIZE
from models.feature_extractor import create_point_cloud_feature_extractor

def create_hash_layer(features, name_suffix=""):
    """
    Create a hash layer with tanh activation
    """
    hash_output = layers.Dense(
        HASH_BIT_SIZE,
        activation='tanh',
        name=f"hash_output_{name_suffix}",
        kernel_initializer=initializers.GlorotUniform(),
        bias_initializer=initializers.Zeros(),
        kernel_regularizer=keras.regularizers.l2(1e-5),
        bias_regularizer=keras.regularizers.l2(1e-5),
    )(features)

    return hash_output

def create_hashing_model():
    """Siamese network with simplified architecture"""
    # Feature extractor (shared weights)
    feature_extractor = create_point_cloud_feature_extractor()

    # Input for point cloud pairs
    input_a = keras.Input(shape=(None, 3), name="point_cloud_a")
    input_b = keras.Input(shape=(None, 3), name="point_cloud_b")

    # Extract features
    features_a = feature_extractor(input_a)
    features_b = feature_extractor(input_b)

    # Generate hash codes
    hash_a = create_hash_layer(features_a, "a")
    hash_b = create_hash_layer(features_b, "b")

    # Concatenate outputs for loss computation
    model_output = layers.Concatenate(name="concatenated_hashes")([hash_a, hash_b])

    # Create model
    model = keras.Model(inputs=[input_a, input_b], outputs=model_output)
    
    return model