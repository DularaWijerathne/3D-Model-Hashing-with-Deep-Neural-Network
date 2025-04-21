import tensorflow as tf
from tensorflow import keras

from config import BATCH_SIZE, HASH_BIT_SIZE, MARGIN

class HashingLoss(keras.losses.Loss):
    """
    Robust and simplified hashing loss function with strong numerical stability
    """
    def __init__(self, batch_size=BATCH_SIZE, hash_bit_size=HASH_BIT_SIZE,
                 margin=MARGIN, balance_weight=0.05, independence_weight=0.01,
                 quantization_weight=0.1, name='hashing_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.hash_bit_size = hash_bit_size
        self.margin = margin
        self.balance_weight = balance_weight
        self.independence_weight = independence_weight
        self.quantization_weight = quantization_weight
        self.epsilon = 1e-6  # Small value to prevent division by zero

    def call(self, y_true, y_pred):
        """Calculate the hashing loss with extreme numerical stability"""
        # Ensure all inputs are float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Split concatenated hash codes from the model output
        hash_a, hash_b = tf.split(y_pred, num_or_size_splits=2, axis=1)
        
        # Prevent extreme values that could lead to NaN
        hash_a = tf.clip_by_value(hash_a, -3.0, 3.0)
        hash_b = tf.clip_by_value(hash_b, -3.0, 3.0)
        
        # Simplify labels to -1 and 1 format for contrastive calculations
        sim_labels = 2.0 * y_true - 1.0
        sim_labels = tf.reshape(sim_labels, [-1, 1])  # Ensure shape is [batch_size, 1]
        
        # 1. CONTRASTIVE LOSS - Similarity Preservation
        # Use basic similarity measure (Cosine similarity) for stability
        # Avoid L2 normalization which can be unstable; use normalized inner product instead
        dot_product = tf.reduce_sum(hash_a * hash_b, axis=1, keepdims=True)
        hash_a_sq = tf.reduce_sum(tf.square(hash_a), axis=1, keepdims=True)
        hash_b_sq = tf.reduce_sum(tf.square(hash_b), axis=1, keepdims=True)
        similarity = dot_product / (tf.sqrt(hash_a_sq * hash_b_sq) + self.epsilon)
        
        # Clip similarity to avoid extreme values
        similarity = tf.clip_by_value(similarity, -0.99, 0.99)
        
        # Simple contrastive loss: pull similar pairs together, push dissimilar pairs apart
        # similar pairs: maximize similarity (minimize -similarity)
        # dissimilar pairs: minimize similarity with margin
        similar_loss = (1.0 - similarity) / 2.0
        dissimilar_loss = tf.maximum(0.0, similarity - (-0.5))
        
        # Weight by similarity using multiplication (graph-friendly)
        contrastive_factor = (1.0 + sim_labels) / 2.0  # Convert [-1,1] to [0,1]
        pair_losses = contrastive_factor * similar_loss + (1.0 - contrastive_factor) * dissimilar_loss
        contrastive_loss = tf.reduce_mean(pair_losses)
        
        # 2. QUANTIZATION LOSS - Push values toward binary
        # Simplified quantization loss that avoids potential instabilities
        quant_loss = tf.reduce_mean(tf.square(tf.abs(hash_a) - 1.0)) + tf.reduce_mean(tf.square(tf.abs(hash_b) - 1.0))
        quant_loss = quant_loss / 2.0  # Average the two terms
        
        # 3. BALANCE LOSS - Ensure each bit has 50% chance of being 1/-1
        # Simplified bit balance loss with clipping
        hash_batch = tf.concat([hash_a, hash_b], axis=0)
        bit_mean = tf.reduce_mean(hash_batch, axis=0)
        bit_mean = tf.clip_by_value(bit_mean, -1.0, 1.0)  # Prevent extreme gradients
        balance_loss = tf.reduce_mean(tf.square(bit_mean))
        
        # 4. INDEPENDENCE LOSS - Ensure different bits are uncorrelated
        # Simplified to avoid complex matrix operations that could cause instability
        hash_centered = hash_batch - tf.reduce_mean(hash_batch, axis=0, keepdims=True)
        hash_centered = tf.clip_by_value(hash_centered, -3.0, 3.0)
        
        # Compute batch size as float
        batch_size_f = tf.cast(tf.shape(hash_batch)[0], tf.float32)
        
        # Start with normalized hash codes for better numerical stability
        normalized_hash = hash_centered / (tf.sqrt(tf.reduce_sum(tf.square(hash_centered), axis=0, keepdims=True)) + self.epsilon)
        
        # Compute correlation matrix
        corr = tf.matmul(tf.transpose(normalized_hash), normalized_hash) / batch_size_f
        
        # Remove diagonal (we don't care about self-correlation)
        mask = tf.ones_like(corr) - tf.eye(tf.shape(corr)[0])
        corr_no_diag = corr * mask
        
        # Simple sum of squared correlation terms
        independence_loss = tf.reduce_mean(tf.square(corr_no_diag))
        
        # FINAL COMBINED LOSS
        # Very conservative weighting to avoid any component overwhelming the others
        # Use much smaller weights for auxiliary losses
        total_loss = (
            contrastive_loss + 
            self.quantization_weight * quant_loss +
            self.balance_weight * balance_loss + 
            self.independence_weight * independence_loss
        )
        
        # Add robust NaN detection and handling
        # This uses tf.cond which is compatible with graph mode
        safe_loss = tf.cond(
            tf.math.is_finite(total_loss),
            lambda: total_loss,
            lambda: tf.constant(0.1, dtype=tf.float32)  # Small constant fallback
        )
        
        return safe_loss
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'batch_size': self.batch_size,
            'hash_bit_size': self.hash_bit_size,
            'margin': self.margin,
            'balance_weight': self.balance_weight,
            'independence_weight': self.independence_weight,
            'quantization_weight': self.quantization_weight,
        })
        return config