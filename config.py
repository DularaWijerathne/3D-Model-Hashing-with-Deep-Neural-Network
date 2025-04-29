# Model configuration
HASH_BIT_SIZE = 64  # Size of hash code in bits
POINT_CLOUD_SIZE = 1024  # Number of points in each point cloud
POINT_FEATURES = 3  # X, Y, Z coordinates

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0001

# GPU configuration
GPU_MEMORY_LIMIT = None
USE_MIXED_PRECISION = True  # Enable mixed precision training for faster computation

# Data augmentation parameters
ROTATION_AUGMENTATION = True  # Enable random rotation augmentation
JITTER_SIGMA = 0.01  # Standard deviation for point jitter in augmentation

# Loss function weights
BALANCE_WEIGHT = 0.05  # Weight for bit balance constraint
INDEPENDENCE_WEIGHT = 0.05  # Weight for bit independence constraint
QUANTIZATION_WEIGHT = 0.2  # Weight for quantization loss
MARGIN = 0.2  # Margin for contrastive loss

# Evaluation parameters
RETRIEVAL_K = 100  # Top-k for precision and recall computation

# Thresholders
SIMILARITY_THRESHOLD = 0.92
