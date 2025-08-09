import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import POINT_CLOUD_SIZE, POINT_FEATURES


def self_attention_module(x, num_heads=4):
    # Normalize input to attention (critical!)
    x_norm = layers.LayerNormalization()(x)
    
    # Scale attention logits by sqrt(d_k) to prevent saturation
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1] // num_heads,
    )(x_norm, x_norm)
    
    # Residual + Norm
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    
    # FFN with normalization
    ffn = layers.LayerNormalization()(x)
    ffn = layers.Dense(x.shape[-1] * 2, activation='relu')(ffn)
    ffn = layers.Dense(x.shape[-1])(ffn)

    # Add
    x = layers.Add()([x, ffn])
    
    return x


def create_point_cloud_feature_extractor():

    # Input layer
    inputs = keras.Input(shape=(POINT_CLOUD_SIZE, POINT_FEATURES))

    # Point-wise MLPs
    x = layers.Conv1D(64, 1, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # # Add self-attention after initial feature extraction
    # x = self_attention_module(x, num_heads=4)
    
    # Continue with deeper feature extraction
    x = layers.Conv1D(256, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # # Second self-attention module
    # x = self_attention_module(x, num_heads=8)
    
    # Final feature extraction
    x = layers.Conv1D(512, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Global feature extraction with pooling
    # Combine max and average pooling for better feature capture
    x_max = layers.GlobalMaxPooling1D()(x)
    x_avg = layers.GlobalAveragePooling1D()(x)
    x = layers.Concatenate()([x_max, x_avg])
    
    # Dense layers with skip connection
    x1 = layers.Dense(512, activation='relu')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.3)(x1)  # Add dropout for regularization
    
    x2 = layers.Dense(256, activation='relu')(x1)
    x2 = layers.BatchNormalization()(x2)
    
    # Skip connection
    x = layers.Dense(256, activation=None)(x)
    x = layers.Add()([x, x2])
    x = layers.Activation('relu')(x)

    # Create model
    model = keras.Model(inputs=inputs, outputs=x, name="PointCloudFeatureExtractor")
    return model