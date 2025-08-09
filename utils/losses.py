import tensorflow as tf
from tensorflow import keras

from config import HASH_BIT_SIZE, BALANCE_WEIGHT, INDEPENDENCE_WEIGHT, MARGIN

class HashingLoss(keras.losses.Loss):
    """
    Hashing loss using cosine similarity
    """
    
    def __init__(
        self,
        margin=MARGIN,  # Margin for contrastive loss
        balance_weight=BALANCE_WEIGHT,
        independence_weight=INDEPENDENCE_WEIGHT,
        name="hashing_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.margin = tf.cast(margin, tf.float32)
        self.balance_weight = balance_weight
        self.independence_weight = independence_weight

    def call(self, y_true, y_pred):
        # Ensure consistent types
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Split concatenated hashes
        hash_a, hash_b = tf.split(y_pred, num_or_size_splits=2, axis=1)
        
        # Normalize hashes
        hash_a = tf.math.l2_normalize(hash_a, axis=1)
        hash_b = tf.math.l2_normalize(hash_b, axis=1)
        
        # Compute cosine similarities
        similarities = tf.reduce_sum(hash_a * hash_b, axis=1)
        
        # Contrastive loss
        pos_mask = tf.cast(y_true > 0.5, tf.float32)
        neg_mask = 1.0 - pos_mask
        
        # Positive pairs should have high similarity (close to 1)
        pos_loss = pos_mask * (1 - similarities)
        
        # Negative pairs should have similarity below margin
        neg_loss = neg_mask * tf.maximum(similarities - self.margin, 0)
        
        contrastive_loss = tf.reduce_mean(pos_loss + neg_loss)
        
        # Balance loss (entropy-based)
        all_hashes = tf.concat([hash_a, hash_b], axis=0)
        mean_activations = tf.reduce_mean(all_hashes, axis=0)
        balance_loss = tf.reduce_mean(tf.abs(mean_activations))
        
        # Independence loss (covariance-based)
        centered = all_hashes - tf.reduce_mean(all_hashes, axis=0, keepdims=True)
        cov = tf.matmul(tf.transpose(centered), centered) / tf.cast(
            tf.shape(all_hashes)[0], tf.float32
        )
        mask = tf.ones_like(cov) - tf.eye(HASH_BIT_SIZE)
        independence_loss = tf.reduce_mean(tf.square(cov * mask))
        
        # Combined loss
        total_loss = (
            contrastive_loss
            + self.balance_weight * balance_loss
            + self.independence_weight * independence_loss
        )
        
        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "margin": float(self.margin),
            "balance_weight": self.balance_weight,
            "independence_weight": self.independence_weight,
        })
        return config