import numpy as np
import tensorflow as tf
from tensorflow import keras
from data.preprocessing import augment_point_cloud


@keras.utils.register_keras_serializable(package="utils.metrics")
class HashingAccuracy(tf.keras.metrics.Metric):
    """
    Custom accuracy metric for hashing models that measures the similarity
    between hash codes using Hamming distance and compares with ground truth labels.
    """

    def __init__(
        self, hash_bit_size, similarity_threshold, name="hashing_accuracy", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        # if not 0 <= similarity_threshold <= 1:
        #     raise ValueError("similarity_threshold must be between 0 and 1")

        self.hash_bit_size = hash_bit_size
        self.threshold = similarity_threshold
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Split concatenated hash codes
        hash_a, hash_b = tf.split(y_pred, num_or_size_splits=2, axis=1)

        # Binarize hash codes (ensure they are 0 or 1)
        hash_a_bin = tf.cast(hash_a > 0, tf.float32)
        hash_b_bin = tf.cast(hash_b > 0, tf.float32)

        # Compute Hamming similarity (fraction of matching bits)
        matching_bits = tf.reduce_sum(
            tf.cast(hash_a_bin == hash_b_bin, tf.float32), axis=1
        )
        similarity = matching_bits / self.hash_bit_size

        # Threshold to make binary prediction
        predictions = tf.cast(similarity > self.threshold, tf.float32)

        # Cast y_true to float32 for operations
        y_true = tf.cast(y_true, tf.float32)

        # Update confusion matrix weights
        self.true_positives.assign_add(tf.reduce_sum(predictions * y_true))
        self.false_positives.assign_add(tf.reduce_sum(predictions * (1 - y_true)))
        self.true_negatives.assign_add(tf.reduce_sum((1 - predictions) * (1 - y_true)))
        self.false_negatives.assign_add(tf.reduce_sum((1 - predictions) * y_true))

    def result(self):
        # Calculate accuracy from confusion matrix
        numerator = self.true_positives + self.true_negatives
        denominator = (
            self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives
        )
        return numerator / (denominator + tf.keras.backend.epsilon())

    def reset_states(self):
        # Reset all state variables
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_negatives.assign(0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hash_bit_size": self.hash_bit_size,
                "similarity_threshold": self.threshold,
            }
        )
        return config


class HashingMetrics(keras.callbacks.Callback):
    """
    Enhanced callback to compute hashing performance metrics during training
    using threshold-based approach with Hamming distance
    """

    def __init__(
        self,
        val_pairs,
        hash_bit_size,
        similarity_threshold,
        feature_extractor=None,
        hash_layer=None,
        eval_frequency=5,
        test_rotations=True,
    ):
        super().__init__()
        # if not 0 <= similarity_threshold <= 1:
        #     raise ValueError("similarity_threshold must be between 0 and 1")

        self.val_pairs = val_pairs
        self.hash_bit_size = hash_bit_size
        self.similarity_threshold = similarity_threshold
        self.feature_extractor = feature_extractor
        self.hash_layer = hash_layer
        self.eval_frequency = eval_frequency
        self.test_rotations = test_rotations
        self._model = None

        # Initialize metrics history
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
        """Override the set_model method to store the model reference"""
        super().set_model(model)
        self._model = model

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.eval_frequency != 0:
            return

        # Initialize feature extractor and hash layer if not set
        self._initialize_layers()

        # Compute metrics
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

    def _initialize_layers(self):
        """Initialize feature extractor and hash layer if needed"""
        if self.feature_extractor is None and self._model is not None:
            for layer in self._model.layers:
                if (
                    isinstance(layer, keras.Model)
                    and "PointCloudFeatureExtractor" in layer.name
                ):
                    self.feature_extractor = layer
                    break
            if self.feature_extractor is None and len(self._model.layers) > 2:
                self.feature_extractor = self._model.layers[2]

        if self.hash_layer is None and self._model is not None:
            for layer in self._model.layers:
                if isinstance(layer, keras.layers.Dense) and "hash_dense" in layer.name:
                    self.hash_layer = layer
                    break

    def compute_binary_hash(self, point_cloud):
        """Convert a point cloud to binary hash code"""
        pc = np.expand_dims(point_cloud, axis=0)
        features = self.feature_extractor.predict(pc, verbose=0)

        if self.hash_layer is None:
            raise ValueError("Hash layer not initialized")

        hash_values = tf.nn.tanh(self.hash_layer(features))
        return tf.cast(hash_values > 0, tf.int32).numpy()[0]

    def compute_metrics(self, pairs=None):
        """Batch compute metrics for efficiency"""
        if pairs is None:
            pairs = self.val_pairs[: min(500, len(self.val_pairs))]

        # Pre-compute all hashes
        hashes = [
            (
                self.compute_binary_hash(pair[0][0]),
                self.compute_binary_hash(pair[0][1]),
                pair[1],  # label
            )
            for pair in pairs
        ]

        # Vectorized similarity calculation
        similarities = np.array(
            [self._hamming_similarity(h1, h2) for h1, h2, _ in hashes]
        )
        labels = np.array([label for _, _, label in hashes])

        # Compute predictions
        predictions = similarities > self.similarity_threshold

        # Calculate metrics
        true_pos = np.sum(predictions & labels)
        false_pos = np.sum(predictions & ~labels)
        false_neg = np.sum(~predictions & labels)

        precision = true_pos / (true_pos + false_pos + 1e-8)
        recall = true_pos / (true_pos + false_neg + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return precision, recall, f1

    def _hamming_similarity(self, hash1, hash2):
        """Compute normalized Hamming similarity"""
        return 1.0 - (np.sum(hash1 != hash2) / self.hash_bit_size)

    def test_rotation_invariance(self):
        """Test model's rotation invariance"""
        test_pairs = self.val_pairs[: min(100, len(self.val_pairs))]

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
        invariance_score = rot_f1 / (orig_f1 + 1e-8)

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
        print(f"Rotation Test: Precision={rot_precision:.4f}, Recall={rot_recall:.4f}")
        print(f"Rotation Invariance Score (closer to 1.0 is better): {invariance_score:.4f}")

    def _update_logs(self, logs, precision, recall, f1):
        """Update training logs"""
        logs["val_precision"] = precision
        logs["val_recall"] = recall
        logs["val_f1"] = f1
        if self.test_rotations:
            logs["rotation_invariance"] = self.metrics_history["invariance_score"][-1]
