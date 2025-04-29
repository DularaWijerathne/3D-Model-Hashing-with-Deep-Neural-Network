import numpy as np
import tensorflow as tf
from tensorflow import keras
from data.preprocessing import augment_point_cloud


@keras.utils.register_keras_serializable(package="utils.metrics")
class HashingAccuracy(tf.keras.metrics.Metric):
    def __init__(self, hash_bit_size, similarity_threshold, name="hashing_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.hash_bit_size = hash_bit_size
        self.threshold = similarity_threshold
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        hash_a, hash_b = tf.split(y_pred, num_or_size_splits=2, axis=1)
        
        # Normalize hashes for cosine similarity
        hash_a = tf.math.l2_normalize(hash_a, axis=1)
        hash_b = tf.math.l2_normalize(hash_b, axis=1)
        
        # Compute cosine similarity
        similarity = tf.reduce_sum(hash_a * hash_b, axis=1)
        predictions = tf.cast(similarity > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        self.true_positives.assign_add(tf.reduce_sum(predictions * y_true))
        self.false_positives.assign_add(tf.reduce_sum(predictions * (1 - y_true)))
        self.false_negatives.assign_add(tf.reduce_sum((1 - predictions) * y_true))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return f1

    def get_config(self):
        config = super().get_config()
        config.update({
            "hash_bit_size": self.hash_bit_size,
            "similarity_threshold": self.threshold
        })
        return config


class HashingMetrics(keras.callbacks.Callback):
    def __init__(
        self,
        val_pairs,
        rot_inv_pairs,
        hash_bit_size,
        similarity_threshold,
        model=None,
        eval_frequency=5,
        test_rotations=True,
    ):
        super().__init__()
        self.val_pairs = val_pairs
        self.rot_inv_pairs = rot_inv_pairs
        self.hash_bit_size = hash_bit_size
        self.similarity_threshold = similarity_threshold
        self._model = model
        self.eval_frequency = eval_frequency
        self.test_rotations = test_rotations
        self.metrics_history = {
            "epoch": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "rotation_precision": [],
            "rotation_recall": [],
            "rotation_f1": [],
            "invariance_score": [],
        }

    def set_model(self, model):
        """Override to ensure model reference is set"""
        self._model = model
        super().set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.eval_frequency != 0:
            return

        # Compute metrics using model outputs directly
        precision, recall, f1_score = self.compute_metrics()

        # Store and print results
        self._update_history(epoch, precision, recall, f1_score)

        # Test rotation invariance if enabled
        if self.test_rotations:
            rot_metrics = self.test_rotation_invariance()
            self._update_rotation_history(*rot_metrics)

        # Add metrics to logs
        if logs is not None:
            self._update_logs(logs, precision, recall, f1_score)

    def compute_metrics(self, pairs=None):
        """Compute metrics using cosine similarity"""
        if pairs is None:
            pairs = self.val_pairs[:min(500, len(self.val_pairs))]

        # Prepare batch
        pc1 = np.array([pair[0][0] for pair in pairs])
        pc2 = np.array([pair[0][1] for pair in pairs])
        labels = np.array([pair[1] for pair in pairs])

        # Get model outputs
        outputs = self._model.predict([pc1, pc2], verbose=0)
        
        # Split outputs
        hash_a, hash_b = np.split(outputs, 2, axis=1)
        
        # Normalize for cosine similarity
        norm_a = np.linalg.norm(hash_a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(hash_b, axis=1, keepdims=True)
        norm_a = np.where(norm_a == 0, 1, norm_a)  # Avoid division by zero
        norm_b = np.where(norm_b == 0, 1, norm_b)  # Avoid division by zero
        
        hash_a_norm = hash_a / norm_a
        hash_b_norm = hash_b / norm_b
        
        # Compute cosine similarities
        similarities = np.sum(hash_a_norm * hash_b_norm, axis=1)
        
        # Compute predictions
        predictions = similarities > self.similarity_threshold

        # Calculate metrics
        true_pos = np.sum(predictions & (labels == 1))
        false_pos = np.sum(predictions & (labels == 0))
        false_neg = np.sum(~predictions & (labels == 1))

        precision = true_pos / (true_pos + false_pos + 1e-8)
        recall = true_pos / (true_pos + false_neg + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return precision, recall, f1

    def test_rotation_invariance(self):
        """Test rotation invariance using cosine similarity"""
        test_pairs = self.rot_inv_pairs[:min(100, len(self.rot_inv_pairs))]

        # Original metrics
        orig_precision, orig_recall, orig_f1 = self.compute_metrics(test_pairs)

        # Create rotated versions
        rotated_pairs = []
        for pc_pair, label in test_pairs:
            rotated_pc = augment_point_cloud(pc_pair[0], rotate=True, jitter_sigma=0)
            rotated_pairs.append(([rotated_pc, pc_pair[1]], label))

        # Compute rotated metrics
        rot_precision, rot_recall, rot_f1 = self.compute_metrics(rotated_pairs)

        # Calculate invariance score
        invariance_score = min(1, rot_f1 / (orig_f1 + 1e-8))

        return rot_precision, rot_recall, rot_f1, invariance_score
    
    def _update_history(self, epoch, precision, recall, f1):
        """Update metrics history"""
        self.metrics_history["epoch"].append(epoch + 1)
        self.metrics_history["precision"].append(precision)
        self.metrics_history["recall"].append(recall)
        self.metrics_history["f1_score"].append(f1)
        print(
            f"\nEpoch {epoch+1}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

    def _update_rotation_history(
        self, rot_precision, rot_recall, rot_f1, invariance_score
    ):
        """Update rotation metrics history"""
        self.metrics_history["rotation_precision"].append(rot_precision)
        self.metrics_history["rotation_recall"].append(rot_recall)
        self.metrics_history["rotation_f1"].append(rot_f1)
        self.metrics_history["invariance_score"].append(invariance_score)
        print(f"Rotation Test: Precision={rot_precision:.4f}, Recall={rot_recall:.4f}, F1={rot_f1:.4f}")
        print(f"Rotation Invariance Score (closer to 1.0 is better): {invariance_score:.4f}\n")

    def _update_logs(self, logs, precision, recall, f1):
        """Update training logs"""
        logs["val_precision"] = precision
        logs["val_recall"] = recall
        logs["val_f1"] = f1
        if self.test_rotations:
            logs["rotation_invariance"] = self.metrics_history["invariance_score"][-1]