import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import trimesh

from config import POINT_CLOUD_SIZE, HASH_BIT_SIZE, GPU_MEMORY_LIMIT, USE_MIXED_PRECISION, SUPPORTED_FORMATS

from data.preprocessing import normalize_point_cloud
from utils.losses import HashingLoss
from utils.metrics import HashingAccuracy

#  Suppress warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO/WARNING logs (keeps errors)
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'  # Disable XLA autotune warnings


# Add after imports
def configure_gpu_for_retrieval():
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
                    print(f"GPU memory limited to {GPU_MEMORY_LIMIT} MB for retrieval")
                else:
                    # Or use memory growth option
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU memory growth enabled for retrieval")

            # Enable mixed precision if configured
            if USE_MIXED_PRECISION:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"Mixed precision enabled for retrieval")

            print(f"retrieval will use GPU acceleration (found {len(gpus)} GPU(s))")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error for retrieval: {e}")
            return False
    else:
        print("No GPU found for retrieval. Running on CPU.")
        return False


# Configure GPU at module load time
has_gpu_for_retrieval = configure_gpu_for_retrieval()


class HashRetrievalSystem:
    """
    Class for retrieval with the trained hashing model
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

            # Extract hash layer - need to get both the dense layer and the binarize layer
            for layer in self.model.layers:
                if isinstance(layer, keras.layers.Dense) and "hash_dense" in layer.name:
                    self.hash_layer = layer
                    break

        self.database = {}  # Format: {hash_str: [{"path": ..., "class": ...}, ...]}

    def compute_hash(self, point_cloud):
        """Compute binary hash code for a single point cloud"""
        # Preprocess point cloud
        processed_pc = normalize_point_cloud(point_cloud)
        processed_pc = np.expand_dims(processed_pc, axis=0)
        
        # Use GPU if available
        device = "/GPU:0" if has_gpu_for_retrieval else "/CPU:0"
        with tf.device(device):
            # Get model output
            dummy_pc = np.zeros_like(processed_pc)  # Create dummy pair
            model_output = self.model.predict([{'point_cloud_a':processed_pc, 'point_cloud_b':dummy_pc}], verbose=0)
            
            # Get first half of output (hash for input point cloud)
            continuous_hash = model_output[0, :HASH_BIT_SIZE]
            
            # Binarize (sign function)
            binary_hash = (continuous_hash > 0).astype(int)
            
        return binary_hash

    def compute_similarity(self, hash1, hash2):
        """Compute similarity between two binary hash codes"""
        if hash1.shape != hash2.shape:
            raise ValueError("Arrays must have the same shape")
        similarity = np.mean(hash1 == hash2)
        return similarity
    
    def load_point_cloud(self, model_path):
        """
        Load and preprocess a 3D model into a standardized point cloud.
        """
        
        if not model_path.lower().endswith(SUPPORTED_FORMATS):
            raise ValueError(f"Unsupported file format. Expected: {SUPPORTED_FORMATS}")

        # Load models and create point cloud
        mesh = trimesh.load_mesh(model_path)
        point_cloud = mesh.sample(POINT_CLOUD_SIZE)
        
        return point_cloud
    
    def add_model(self, model_path, category="other"):
        """
        Add a 3D model to the database.
        
        Args:
            model_path: Path to 3D model file (.obj, .ply, etc.)
            category: Category/class of the model
        """
        try:
            point_cloud = self.load_point_cloud(model_path)
            binary_hash = self.compute_hash(point_cloud)
            hash_str = ' '.join(map(str, binary_hash))

            entry = {
                "path": model_path,
                "class": category,
            }

            # Handle collisions by appending to list
            if hash_str in self.database:
                self.database[hash_str].append(entry)
            else:
                self.database[hash_str] = [entry]
            return True
        
        except Exception as e:
            print(f"Skipping {model_path}: {str(e)}")
            return False


    def create_database(self, root_dir):
        """
        Populate the database by scanning a structured directory of 3D models.
        
        Expected Directory Structure:
        root_dir/
        ├── category_1/
        │   ├── model_1.obj
        │   ├── model_2.ply
        │   └── ...
        ├── category_2/
        │   ├── model_1.obj
        │   └── ...
        └── ...
        
        Args:
            root_dir: Path to the root directory containing category subfolders.
                    Each subfolder should contain 3D model files of that category.
                
        """
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory not found: {root_dir}")
        
        total_added = 0
        total_failed = 0
        
        for category in os.listdir(root_dir):
            category_dir = os.path.join(root_dir, category)
            if os.path.isdir(category_dir):
                print(f"Processing category: {category}")
                category_added = 0
                category_failed = 0
                
                model_paths = [
                    os.path.join(category_dir, f) 
                    for f in os.listdir(category_dir) 
                    if f.lower().endswith(SUPPORTED_FORMATS)
                ]
                
                for model_path in model_paths:
                    success = self.add_model(model_path, category)
                    if success:
                        category_added += 1
                    else:
                        category_failed += 1
                        print(f"Failed to add: {model_path}")

                total_added += category_added
                total_failed += category_failed
                print(f"  Added: {category_added} | Failed: {category_failed}")

        print(f"\nDatabase creation complete!")
        print(f"Total models added: {total_added}")
        print(f"Total failures: {total_failed}")


    def retrieve_similar_models(self, query_model_path, database=None):
        if not database:
            database = self.database

        # Compute query hash
        query_point_cloud = self.load_point_cloud(query_model_path)
        query_hash = self.compute_hash(query_point_cloud)

        # Compute similarities
        similarities = []
        for hash_str in database:
            hash_code = np.fromstring(hash_str, 'int', sep=' ')
            similarity = self.compute_similarity(query_hash, hash_code)
            similarities.append((hash_str, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # sorted models
        sorted_models = []
        for hash_str, similarity in similarities:
            models = database[hash_str]
            sorted_models.extend(models)      

        return sorted_models





