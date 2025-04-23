import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

from config import HASH_BIT_SIZE, GPU_MEMORY_LIMIT, USE_MIXED_PRECISION

from data.preprocessing import normalize_point_cloud
from utils.losses import HashingLoss
from utils.metrics import HashingAccuracy

#  Suppress warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO/WARNING logs (keeps errors)
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'  # Disable XLA autotune warnings


# Add after imports
def configure_gpu_for_inference():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Memory growth needs to be set before GPUs have been initialized
            for gpu in gpus:
                if GPU_MEMORY_LIMIT is not None:
                    # Limit memory growth to specified value in MB
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [
                            tf.config.LogicalDeviceConfiguration(
                                memory_limit=GPU_MEMORY_LIMIT
                            )
                        ],
                    )
                    print(f"GPU memory limited to {GPU_MEMORY_LIMIT} MB for inference")
                else:
                    # Or use memory growth option
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU memory growth enabled for inference")

            # Enable mixed precision if configured
            if USE_MIXED_PRECISION:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"Mixed precision enabled for inference")

            print(f"Inference will use GPU acceleration (found {len(gpus)} GPU(s))")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error for inference: {e}")
            return False
    else:
        print("No GPU found for inference. Running on CPU.")
        return False


# Configure GPU at module load time
has_gpu_for_inference = configure_gpu_for_inference()


class HashingInference:
    """
    Class for inference with the trained hashing model
    """

    def __init__(self, model_path=None, feature_extractor_path=None):
        if model_path is None and feature_extractor_path is None:
            raise ValueError(
                "Either model_path or feature_extractor_path must be provided"
            )

        # Create a wrapper for HashingAccuracy with default parameters
        # This helps with model loading if the instance parameters don't match exactly
        def hashing_accuracy_wrapper(*args, **kwargs):
            return HashingAccuracy(hash_bit_size=HASH_BIT_SIZE, *args, **kwargs)

        # Add custom objects dictionary for model loading
        custom_objects = {
            "HashingLoss": HashingLoss,
            "HashingAccuracy": hashing_accuracy_wrapper,
            "hashing_accuracy": hashing_accuracy_wrapper,
        }

        if feature_extractor_path:
            self.feature_extractor = keras.models.load_model(feature_extractor_path)
            # Load hash layer weights from the full model if available
            if model_path:
                full_model = keras.models.load_model(
                    model_path, custom_objects=custom_objects
                )
                # Extract hash layer weights
                for layer in full_model.layers:
                    if (
                        isinstance(layer, keras.layers.Dense)
                        and "hash_dense" in layer.name
                    ):
                        self.hash_layer = layer
                        break
        else:
            # Load the full model
            self.model = keras.models.load_model(
                model_path, custom_objects=custom_objects
            )
            # Extract feature extractor by name
            for layer in self.model.layers:
                if (
                    isinstance(layer, keras.Model)
                    and "PointCloudFeatureExtractor" in layer.name
                ):
                    self.feature_extractor = layer
                    break

            # If not found, use the assumed position
            if not hasattr(self, "feature_extractor") and len(self.model.layers) > 2:
                self.feature_extractor = self.model.layers[2]

            # Extract hash layer
            for layer in self.model.layers:
                if isinstance(layer, keras.layers.Dense) and "hash_dense" in layer.name:
                    self.hash_layer = layer
                    break

    def compute_hash(self, point_cloud):
        """Compute hash code for a single point cloud"""
        # Preprocess point cloud
        processed_pc = normalize_point_cloud(point_cloud)

        # Add batch dimension
        processed_pc = np.expand_dims(processed_pc, axis=0)

        # Use GPU if available
        device = "/GPU:0" if has_gpu_for_inference else "/CPU:0"
        with tf.device(device):
            # Extract features
            features = self.feature_extractor.predict(processed_pc, verbose=0)

            # Compute hash code
            hash_logits = self.hash_layer(features)
            hash_values = tf.nn.tanh(hash_logits)

            # Convert to binary (0 or 1)
            binary_hash = np.where(hash_values > 0, 1, 0).numpy()

        return binary_hash[0]  # Remove batch dimension

    def compute_similarity(self, hash1, hash2):
        """Compute similarity between two hash codes"""
        # Hamming distance
        hamming_dist = np.sum(hash1 != hash2)

        # Normalized similarity (1 - normalized Hamming distance)
        similarity = 1 - (hamming_dist / HASH_BIT_SIZE)

        return similarity

    def retrieve_similar_models(
        self, query_point_cloud, database_point_clouds, top_k=10
    ):
        """
        Retrieve top-k similar models from a database

        Parameters:
        query_point_cloud: Point cloud to query
        database_point_clouds: Dictionary of {model_name: point_cloud}
        top_k: Number of similar models to retrieve

        Returns:
        List of (model_name, similarity_score) tuples
        """
        # Compute query hash
        query_hash = self.compute_hash(query_point_cloud)

        # Compute hash codes for database
        database_hashes = {}
        for name, pc in database_point_clouds.items():
            database_hashes[name] = self.compute_hash(pc)

        # Compute similarities
        similarities = []
        for name, hash_code in database_hashes.items():
            similarity = self.compute_similarity(query_hash, hash_code)
            similarities.append((name, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return similarities[:top_k]
