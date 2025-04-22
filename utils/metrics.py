import numpy as np
import tensorflow as tf
from tensorflow import keras
from data.preprocessing import augment_point_cloud


@keras.utils.register_keras_serializable(package="utils.metrics")
class HashingAccuracy(tf.keras.metrics.Metric):
    """
    Custom accuracy metric for hashing models that properly measures the similarity
    between hash codes and compares with ground truth labels.
    """
    def __init__(self, hash_bit_size, similarity_threshold, name='hashing_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.hash_bit_size = hash_bit_size
        self.threshold = similarity_threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Split concatenated hash codes
        hash_a, hash_b = tf.split(y_pred, num_or_size_splits=2, axis=1)
        
        # Normalize hash codes for more stable similarity calculation
        hash_a_norm = hash_a / (tf.norm(hash_a, axis=1, keepdims=True) + 1e-8)
        hash_b_norm = hash_b / (tf.norm(hash_b, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity (normalized dot product)
        # Range: [-1, 1] where 1 means identical, -1 means opposite
        similarity = tf.reduce_sum(hash_a_norm * hash_b_norm, axis=1)
        
        # Scale to [0, 1] range for threshold comparison
        similarity = (similarity + 1) / 2
        
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
        denominator = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return numerator / (denominator + tf.keras.backend.epsilon())
        
    def reset_states(self):
        # Reset all state variables
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_negatives.assign(0)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'hash_bit_size': self.hash_bit_size,
            'threshold': self.threshold
        })
        return config


class HashingMetrics(keras.callbacks.Callback):
    """
    Enhanced callback to compute hashing performance metrics during training
    using threshold-based approach similar to HashingAccuracy
    """

    def __init__(self, val_pairs, similarity_threshold, feature_extractor=None, hash_layer=None, eval_frequency=5,
                test_rotations=True):
        super().__init__()
        self.val_pairs = val_pairs
        self.feature_extractor = feature_extractor
        self.hash_layer = hash_layer
        self.eval_frequency = eval_frequency
        self.test_rotations = test_rotations
        self.similarity_threshold = similarity_threshold  # Threshold for similarity judgment
        self._model = None  # Use a different name for our internal reference
        self.metrics_history = {
            'epoch': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'rotation_precision': [],
            'rotation_recall': [],
            'rotation_f1': [],
            'invariance_score': []
        }

    def set_model(self, model):
        """Override the set_model method to store the model reference"""
        super().set_model(model)
        self._model = model

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.eval_frequency != 0:  # Only compute every n epochs to save time
            return

        if self.feature_extractor is None and self._model is not None:
            # Find feature extractor by name
            for layer in self._model.layers:
                if isinstance(layer, keras.Model) and "PointCloudFeatureExtractor" in layer.name:
                    self.feature_extractor = layer
                    break

            # If still not found, use the assumed position
            if self.feature_extractor is None and len(self._model.layers) > 2:
                self.feature_extractor = self._model.layers[2]

        if self.hash_layer is None and self._model is not None:
            # Find hash layer in the model
            for layer in self._model.layers:
                if isinstance(layer, keras.layers.Dense) and "hash_dense" in layer.name:
                    self.hash_layer = layer
                    break

        # Compute metrics
        precision, recall, f1_score = self.compute_metrics()
        
        # Store results
        self.metrics_history['epoch'].append(epoch + 1)
        self.metrics_history['precision'].append(precision)
        self.metrics_history['recall'].append(recall)
        self.metrics_history['f1_score'].append(f1_score)
        
        # Print base metrics
        print(f"\nEpoch {epoch+1}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1_score:.4f}")
        
        # Test rotation invariance if enabled
        if self.test_rotations:
            rot_precision, rot_recall, rot_f1, invariance_score = self.test_rotation_invariance()
            self.metrics_history['rotation_precision'].append(rot_precision)
            self.metrics_history['rotation_recall'].append(rot_recall)
            self.metrics_history['rotation_f1'].append(rot_f1)
            self.metrics_history['invariance_score'].append(invariance_score)
            
            print(f"Rotation test: Precision = {rot_precision:.4f}, Recall = {rot_recall:.4f}, F1 = {rot_f1:.4f}")
            print(f"Invariance score: {invariance_score:.2f} (closer to 1.0 is better)\n")
        
        # Add metrics to logs
        if logs is not None:
            logs['val_precision'] = precision
            logs['val_recall'] = recall
            logs['val_f1_score'] = f1_score
            if self.test_rotations:
                logs['rotation_invariance'] = invariance_score

    def compute_binary_hash(self, point_cloud):
        """Convert a point cloud to binary hash code"""
        # Add batch dimension
        pc = np.expand_dims(point_cloud, axis=0)

        # Extract features
        features = self.feature_extractor.predict(pc, verbose=0)

        if self.hash_layer is not None:
            # Use the provided hash layer if available
            hash_layer = self.hash_layer
        elif self._model is not None:
            # Find hash layer in the model
            hash_layer = None
            for layer in self._model.layers:
                if isinstance(layer, keras.layers.Dense) and "hash_dense" in layer.name:
                    hash_layer = layer
                    break
            if hash_layer is None:
                raise ValueError("Hash layer not found")
        else:
            raise ValueError("Either hash_layer must be provided or model must be available")

        hash_values = tf.nn.tanh(hash_layer(features))
        binary_hash = tf.where(hash_values > 0, 1, 0).numpy()

        return binary_hash[0]  # Remove batch dimension

    def compute_hash_similarity(self, hash1, hash2):
        """
        Compute similarity between two hash codes based on normalized dot product,
        scaled to [0, 1] range
        """
        # Ensure hash codes are in correct format
        hash1 = np.asarray(hash1, dtype=np.float32)
        hash2 = np.asarray(hash2, dtype=np.float32)
        
        # Convert from binary (0/1) to bipolar (-1/+1) if needed
        if np.all((hash1 == 0) | (hash1 == 1)):
            hash1 = 2 * hash1 - 1  # Convert 0/1 to -1/+1
        if np.all((hash2 == 0) | (hash2 == 1)):
            hash2 = 2 * hash2 - 1  # Convert 0/1 to -1/+1
            
        # Normalize hash codes
        norm1 = np.linalg.norm(hash1) + 1e-8
        norm2 = np.linalg.norm(hash2) + 1e-8
        hash1_norm = hash1 / norm1
        hash2_norm = hash2 / norm2
        
        # Compute dot product (cosine similarity since vectors are normalized)
        # Range: [-1, 1]
        dot_product = np.sum(hash1_norm * hash2_norm)
        
        # Scale to [0, 1] range
        similarity = (dot_product + 1) / 2
        
        return similarity

    def compute_metrics(self):
        """
        Compute precision and recall using a threshold-based approach,
        similar to HashingAccuracy
        """
        # Extract a subset of validation data for evaluation
        eval_size = min(500, len(self.val_pairs))  # Use at most 500 samples for evaluation
        eval_pairs = self.val_pairs[:eval_size]

        # Initialize confusion matrix counters
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0

        # Process each pair
        for pc_pair, label in eval_pairs:
            # Compute hash codes
            hash1 = self.compute_binary_hash(pc_pair[0])
            hash2 = self.compute_binary_hash(pc_pair[1])
            
            # Compute similarity
            similarity = self.compute_hash_similarity(hash1, hash2)
            
            # Make binary prediction based on threshold
            prediction = 1 if similarity > self.similarity_threshold else 0
            
            # Update confusion matrix
            if prediction == 1 and label == 1:
                true_positives += 1
            elif prediction == 1 and label == 0:
                false_positives += 1
            elif prediction == 0 and label == 1:
                false_negatives += 1
            else:  # prediction == 0 and label == 0
                true_negatives += 1
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        
        # Calculate F1 score
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        return precision, recall, f1_score

    def test_rotation_invariance(self):
        """
        Test rotation invariance using threshold-based metrics
        """
        # Use a smaller subset for rotation testing to save time
        test_size = min(100, len(self.val_pairs))
        test_pairs = self.val_pairs[:test_size]
        
        # Initialize counters for original and rotated versions
        orig_tp = 0
        orig_fp = 0
        orig_fn = 0
        orig_tn = 0
        
        rot_tp = 0
        rot_fp = 0
        rot_fn = 0
        rot_tn = 0
        
        # Process each pair
        for pc_pair, label in test_pairs:
            # Create rotated version of first point cloud
            rotated_pc1 = augment_point_cloud(pc_pair[0], rotate=True, jitter_sigma=0)
            
            # Compute hash codes
            orig_hash1 = self.compute_binary_hash(pc_pair[0])
            rot_hash1 = self.compute_binary_hash(rotated_pc1)
            hash2 = self.compute_binary_hash(pc_pair[1])
            
            # Compute similarities
            orig_similarity = self.compute_hash_similarity(orig_hash1, hash2)
            rot_similarity = self.compute_hash_similarity(rot_hash1, hash2)
            
            # Make predictions
            orig_prediction = 1 if orig_similarity > self.similarity_threshold else 0
            rot_prediction = 1 if rot_similarity > self.similarity_threshold else 0
            
            # Update original confusion matrix
            if orig_prediction == 1 and label == 1:
                orig_tp += 1
            elif orig_prediction == 1 and label == 0:
                orig_fp += 1
            elif orig_prediction == 0 and label == 1:
                orig_fn += 1
            else:  # orig_prediction == 0 and label == 0
                orig_tn += 1
                
            # Update rotated confusion matrix
            if rot_prediction == 1 and label == 1:
                rot_tp += 1
            elif rot_prediction == 1 and label == 0:
                rot_fp += 1
            elif rot_prediction == 0 and label == 1:
                rot_fn += 1
            else:  # rot_prediction == 0 and label == 0
                rot_tn += 1
        
        # Calculate metrics for original
        orig_precision = orig_tp / (orig_tp + orig_fp + 1e-8)
        orig_recall = orig_tp / (orig_tp + orig_fn + 1e-8)
        orig_f1 = 2 * orig_precision * orig_recall / (orig_precision + orig_recall + 1e-8)
        
        # Calculate metrics for rotated
        rot_precision = rot_tp / (rot_tp + rot_fp + 1e-8)
        rot_recall = rot_tp / (rot_tp + rot_fn + 1e-8)
        rot_f1 = 2 * rot_precision * rot_recall / (rot_precision + rot_recall + 1e-8)
        
        # Calculate invariance score as ratio of F1 scores
        invariance_score = rot_f1 / orig_f1 if orig_f1 > 0 else 0.0
        
        return rot_precision, rot_recall, rot_f1, invariance_score