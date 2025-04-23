import tensorflow as tf
from tensorflow import keras

from config import BATCH_SIZE, HASH_BIT_SIZE, MARGIN

class HashingLoss(keras.losses.Loss):
    """
    Optimized hashing loss with proper type handling and Hamming distance optimization
    """
    
    def __init__(
        self,
        batch_size=BATCH_SIZE,
        hash_bit_size=HASH_BIT_SIZE,
        margin=MARGIN,
        balance_weight=0.1,
        independence_weight=0.01,
        quantization_weight=0.2,
        name="hashing_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.hash_bit_size = hash_bit_size
        self.margin = tf.cast(margin, tf.float32)  # Ensure margin is float32
        self.balance_weight = balance_weight
        self.independence_weight = independence_weight
        self.quantization_weight = quantization_weight
        self.temperature = 0.1

    def call(self, y_true, y_pred):
        # Ensure consistent types
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Split concatenated hashes
        hash_a, hash_b = tf.split(y_pred, num_or_size_splits=2, axis=1)
        
        # Straight-Through Estimator for binary hashes
        binary_a = self._ste_quantize(hash_a)
        binary_b = self._ste_quantize(hash_b)
        
        # Convert labels to [-1, 1] range
        labels = 2 * y_true - 1  # 1 for similar, -1 for dissimilar
        
        # 1. HAMMING CONTRASTIVE LOSS
        hamming_sim = self._hamming_similarity(binary_a, binary_b)
        
        # Fix type mismatch in maximum operation
        contrastive_loss = tf.reduce_mean(
            tf.where(
                labels > 0,
                1 - hamming_sim,  # Minimize distance for similar pairs
                tf.maximum(tf.cast(0, tf.float32), hamming_sim - self.margin)  # Fixed type
            )
        )
        
        # 2. QUANTIZATION LOSS
        quant_loss = tf.reduce_mean(
            tf.square(tf.abs(hash_a) - 1) + tf.square(tf.abs(hash_b) - 1)
        ) / 2
        
        # 3. BIT BALANCE LOSS
        all_hashes = tf.concat([binary_a, binary_b], axis=0)
        balance_loss = tf.reduce_mean(tf.square(tf.reduce_mean(all_hashes, axis=0)))
        
        # 4. BIT INDEPENDENCE LOSS
        centered = all_hashes - tf.reduce_mean(all_hashes, axis=0, keepdims=True)
        cov = tf.matmul(centered, centered, transpose_a=True) / tf.cast(
            tf.shape(all_hashes)[0], tf.float32
        )
        mask = tf.ones_like(cov) - tf.eye(self.hash_bit_size)
        independence_loss = tf.reduce_mean(tf.square(cov * mask))
        
        # Combined loss
        total_loss = (
            contrastive_loss
            + self.quantization_weight * quant_loss
            + self.balance_weight * balance_loss
            + self.independence_weight * independence_loss
        )
        
        return total_loss

    def _ste_quantize(self, x):
        """Straight-Through Estimator for binary quantization"""
        binary = tf.cast(x > 0, tf.float32)
        return binary + x - tf.stop_gradient(x)

    def _hamming_similarity(self, a, b):
        """Normalized Hamming similarity [0,1] where 1=identical"""
        return 1 - (tf.reduce_sum(tf.abs(a - b), axis=1) / tf.cast(self.hash_bit_size, tf.float32))

    def get_config(self):
        config = super().get_config()
        config.update({
            "hash_bit_size": self.hash_bit_size,
            "margin": float(self.margin),  # Convert to Python float for serialization
            "balance_weight": self.balance_weight,
            "independence_weight": self.independence_weight,
            "quantization_weight": self.quantization_weight
        })
        return config