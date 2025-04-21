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
    def __init__(self, hash_bit_size, threshold=0.5, name='hashing_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.hash_bit_size = hash_bit_size
        self.threshold = threshold
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
    - Adds rotation invariance testing
    - Improved mAP calculation
    - More detailed logging of retrieval metrics
    """

    def __init__(self, val_pairs, feature_extractor=None, hash_layer=None, eval_frequency=5,
                test_rotations=True):
        super().__init__()
        self.val_pairs = val_pairs
        self.feature_extractor = feature_extractor
        self.hash_layer = hash_layer
        self.eval_frequency = eval_frequency
        self.test_rotations = test_rotations
        self._model = None  # Use a different name for our internal reference
        self.metrics_history = {
            'epoch': [],
            'mAP': [],
            'precision_at_k': [],
            'recall_at_k': [],
            'rotation_mAP': [],
            'rotation_precision': [],
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
        mAP, precision_at_k, recall_at_k = self.compute_retrieval_metrics()
        
        # Store results
        self.metrics_history['epoch'].append(epoch + 1)
        self.metrics_history['mAP'].append(mAP)
        self.metrics_history['precision_at_k'].append(precision_at_k)
        self.metrics_history['recall_at_k'].append(recall_at_k)
        
        # Print base metrics
        print(f"\nEpoch {epoch+1}: mAP = {mAP:.4f}, Top_100_Precision = {precision_at_k:.4f}, Top_100_Recall = {recall_at_k:.4f}")
        
        # Test rotation invariance if enabled
        if self.test_rotations:
            rotation_mAP, rotation_precision, invariance_score = self.test_rotation_invariance()
            self.metrics_history['rotation_mAP'].append(rotation_mAP)
            self.metrics_history['rotation_precision'].append(rotation_precision)
            self.metrics_history['invariance_score'].append(invariance_score)
            
            print(f"Rotation invariance test: mAP = {rotation_mAP:.4f}, Precision = {rotation_precision:.4f}")
            print(f"Invariance score: {invariance_score:.2f} (closer to 1.0 is better)\n")
        
        # Add metrics to logs
        if logs is not None:
            logs['val_mAP'] = mAP
            logs['val_precision'] = precision_at_k
            logs['val_recall'] = recall_at_k
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

    def compute_retrieval_metrics(self):
        """Compute mAP and Precision@k for retrieval evaluation"""
        # Extract a subset of validation data for evaluation
        eval_size = min(500, len(self.val_pairs))  # Use at most 500 samples for evaluation
        eval_pairs = self.val_pairs[:eval_size]

        # Compute hash codes for all point clouds
        queries = []
        database = []
        relevance = []

        for i, (pc_pair, label) in enumerate(eval_pairs):
            query_hash = self.compute_binary_hash(pc_pair[0])
            db_hash = self.compute_binary_hash(pc_pair[1])

            queries.append(query_hash)
            database.append(db_hash)
            relevance.append(label)

        # Convert to arrays
        queries = np.array(queries)
        database = np.array(database)
        relevance = np.array(relevance)

        # Compute Hamming distances between queries and database items
        distances = []
        for query in queries:
            # Compute Hamming distance (count of differing bits)
            dist = np.sum(query != database, axis=1)
            distances.append(dist)

        distances = np.array(distances)

        # Compute mAP
        mAP = self.compute_mAP(distances, relevance)

        # Compute Precision@k and Recall@k
        k = min(100, len(database))
        precision = self.compute_precision_at_k(distances, relevance, k)
        recall = self.compute_recall_at_k(distances, relevance, k)

        return mAP, precision, recall

    def test_rotation_invariance(self):
        """Test how well the model handles rotated point clouds"""
        # Use a smaller subset for rotation testing to save time
        test_size = min(100, len(self.val_pairs))
        test_pairs = self.val_pairs[:test_size]
        
        # Create rotated versions of the test pairs
        rotated_test_pairs = []
        for (point_clouds, label) in test_pairs:
            # Apply rotation only to the first point cloud
            rotated_pc1 = augment_point_cloud(point_clouds[0], rotate=True, jitter_sigma=0)
            rotated_test_pairs.append(([rotated_pc1, point_clouds[1]], label))
        
        # Compute hash codes for the rotated and original point clouds
        original_queries = []
        rotated_queries = []
        database = []
        relevance = []
        
        for i, ((orig_pair, _), (rot_pair, label)) in enumerate(zip(test_pairs, rotated_test_pairs)):
            original_hash = self.compute_binary_hash(orig_pair[0])
            rotated_hash = self.compute_binary_hash(rot_pair[0])
            db_hash = self.compute_binary_hash(orig_pair[1])
            
            original_queries.append(original_hash)
            rotated_queries.append(rotated_hash)
            database.append(db_hash)
            relevance.append(label)
        
        # Convert to arrays
        original_queries = np.array(original_queries)
        rotated_queries = np.array(rotated_queries)
        database = np.array(database)
        relevance = np.array(relevance)
        
        # Compute distances for original and rotated queries
        original_distances = []
        rotated_distances = []
        
        for orig_query, rot_query in zip(original_queries, rotated_queries):
            # Compute Hamming distances
            orig_dist = np.sum(orig_query != database, axis=1)
            rot_dist = np.sum(rot_query != database, axis=1)
            
            original_distances.append(orig_dist)
            rotated_distances.append(rot_dist)
        
        original_distances = np.array(original_distances)
        rotated_distances = np.array(rotated_distances)
        
        # Compute metrics for original and rotated queries
        original_mAP = self.compute_mAP(original_distances, relevance)
        rotated_mAP = self.compute_mAP(rotated_distances, relevance)
        
        k = min(100, len(database))
        original_precision = self.compute_precision_at_k(original_distances, relevance, k)
        rotated_precision = self.compute_precision_at_k(rotated_distances, relevance, k)
        
        # Compute invariance score (ratio of rotated to original performance)
        invariance_score = rotated_mAP / original_mAP if original_mAP > 0 else 0.0
        
        return rotated_mAP, rotated_precision, invariance_score

    def compute_mAP(self, distances, relevance):
        """
        Compute mean Average Precision with improved handling of edge cases
        """
        num_queries = distances.shape[0]
        if num_queries == 0:
            return 0.0

        # For each query, sort database by distance
        ap_sum = 0.0
        valid_queries = 0

        for i in range(num_queries):
            # Sort database by distance to query (ascending order for Hamming distance)
            sorted_indices = np.argsort(distances[i])
            sorted_relevance = relevance[sorted_indices]

            # Find positions of relevant items
            relevant_indices = np.where(sorted_relevance == 1)[0]

            if len(relevant_indices) == 0:
                continue  # Skip queries with no relevant items

            valid_queries += 1
            
            # Compute precision at each relevant item position
            precisions = []
            for j, idx in enumerate(relevant_indices):
                # Precision = (# relevant items up to position) / (position)
                precision = np.sum(sorted_relevance[:idx + 1]) / (idx + 1)
                precisions.append(precision)

            # Average precision for this query
            ap = np.mean(precisions)
            ap_sum += ap

        # Mean Average Precision (avoid division by zero)
        mAP = ap_sum / valid_queries if valid_queries > 0 else 0.0
        return mAP

    def compute_precision_at_k(self, distances, relevance, k):
        """
        Compute Precision@k (proportion of relevant items in top-k)
        """
        num_queries = distances.shape[0]
        if num_queries == 0 or k <= 0:
            return 0.0

        precision_sum = 0.0

        for i in range(num_queries):
            # Sort database by distance to query (ascending for Hamming)
            sorted_indices = np.argsort(distances[i])
            # Take top k items
            top_k_indices = sorted_indices[:k]
            # Compute precision (proportion of relevant items)
            precision = np.sum(relevance[top_k_indices]) / k
            precision_sum += precision

        # Average precision@k across all queries
        avg_precision = precision_sum / num_queries
        return avg_precision

    def compute_recall_at_k(self, distances, relevance, k):
        """
        Compute Recall@k (proportion of all relevant items found in top-k)
        """
        num_queries = distances.shape[0]
        if num_queries == 0 or k <= 0:
            return 0.0

        recall_sum = 0.0
        valid_queries = 0

        for i in range(num_queries):
            # Sort database by distance to query (ascending for Hamming)
            sorted_indices = np.argsort(distances[i])
            # Take top k items
            top_k_indices = sorted_indices[:k]
            
            # Count all relevant items for this query
            relevant_count = np.sum(relevance == 1)
            if relevant_count == 0:
                continue  # Skip queries with no relevant items
                
            valid_queries += 1
            
            # Compute recall (proportion of all relevant items found)
            recall = np.sum(relevance[top_k_indices]) / relevant_count
            recall_sum += recall

        # Average recall@k across all valid queries
        avg_recall = recall_sum / valid_queries if valid_queries > 0 else 0.0
        return avg_recall