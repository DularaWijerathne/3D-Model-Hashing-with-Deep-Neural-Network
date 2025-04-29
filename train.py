import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    GPU_MEMORY_LIMIT,
    USE_MIXED_PRECISION,
    HASH_BIT_SIZE,
    SIMILARITY_THRESHOLD,
)
from data.data_loader import load_data, process_data, create_data_generators
from data.preprocessing import prepare_data_for_training, prepare_data_for_testing
from models.hashing_model import create_hashing_model
from utils.losses import HashingLoss
from utils.metrics import HashingMetrics, HashingAccuracy
from utils.visualization import plot_training_history, plot_metrics_history, create_test_summary_visualization

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO/WARNING logs (keeps errors)
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'  # Disable XLA autotune warnings

# GPU configuration
def configure_gpu():
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
                    print(f"GPU memory limited to {GPU_MEMORY_LIMIT} MB")
                else:
                    # Or use memory growth option
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU memory growth enabled")

            print(f"Found {len(gpus)} GPU(s)")

            # Set mixed precision policy if enabled
            if USE_MIXED_PRECISION:
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                print(
                    f"Mixed precision enabled: compute={policy.compute_dtype}, variable={policy.variable_dtype}"
                )

            # Log device placement
            tf.debugging.set_log_device_placement(False)

            # Print GPU information
            for i, gpu in enumerate(gpus):
                gpu_details = tf.config.experimental.get_device_details(gpu)
                print(f"GPU {i}: {gpu_details.get('device_name', 'Unknown GPU')}")

            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPU found for training. Running on CPU.")
        return False

# Configure GPU at module load time
has_gpu = configure_gpu()

def train_hashing_model(train_data, test_data, output_path):
    """Main function to train the hashing model"""
    # Process data to create pairs
    train_dataset = process_data(train_data)
    test_dataset = process_data(test_data)

    # Prepare data for training
    train_pairs, val_pairs, rot_inv_pairs = prepare_data_for_training(train_dataset)

    # Create data generators
    train_gen, val_gen, train_steps, val_steps = create_data_generators(
        train_pairs, val_pairs
    )

    # Enable mixed precision training for faster GPU computation
    if has_gpu:
        print("Enabling mixed precision training for GPU...")
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Compute dtype: {policy.compute_dtype}")
        print(f"Variable dtype: {policy.variable_dtype}")

    # Create model
    with tf.device("/GPU:0" if has_gpu else "/CPU:0"):
        model = create_hashing_model()

    # Compile model with custom loss and accuracy metric
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # For mixed precision, use loss scaling to prevent underflow
    if has_gpu:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=HashingLoss(),
        metrics=[
            HashingAccuracy(
                hash_bit_size=HASH_BIT_SIZE, 
                similarity_threshold=SIMILARITY_THRESHOLD
            )
        ],
    )

    # Model summary
    print("\n")
    model.summary()

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Create callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_path, "model_checkpoint.keras"),
            save_best_only=True,
            monitor="val_loss",
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_path, "logs"), histogram_freq=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5),
        HashingMetrics(
            val_pairs,
            rot_inv_pairs,
            hash_bit_size=HASH_BIT_SIZE,
            similarity_threshold=SIMILARITY_THRESHOLD,
            model=model,
            test_rotations=True,
        ),
    ]

    # Training configuration
    training_config = {
        "steps_per_epoch": train_steps,
        "validation_data": val_gen,
        "validation_steps": val_steps,
        "epochs": EPOCHS,
        "callbacks": callbacks,
    }

    # Train model
    print("\n")
    print(f"Training on {'GPU' if has_gpu else 'CPU'}...")
    history = model.fit(train_gen, **training_config)

    # Save model
    model.save(os.path.join(output_path, "hashing_model.keras"))

    # Also save the feature extractor separately for inference
    feature_extractor = None
    for layer in model.layers:
        if (
            isinstance(layer, keras.Model)
            and "PointCloudFeatureExtractor" in layer.name
        ):
            feature_extractor = layer
            break

    if feature_extractor is None and len(model.layers) > 2:
        feature_extractor = model.layers[2]

    if feature_extractor:
        feature_extractor.save(os.path.join(output_path, "feature_extractor.keras"))

    # Plot training history with enhanced visualization
    plot_training_history(history, output_path)

    # Plot metrics history from the HashingMetrics callback with enhanced visualization
    metrics_callback = None
    for callback in callbacks:
        if isinstance(callback, HashingMetrics):
            metrics_callback = callback
            break
    
    plot_metrics_history(metrics_callback, output_path)

    # Evaluate on test set
    print("\n\nEvaluating on test set...")
    test_pairs = prepare_data_for_testing(test_dataset)
    rot_inv_test_pairs = prepare_data_for_testing(test_dataset, augment=False)

    # Create metrics object for testing
    metrics = HashingMetrics(
        test_pairs[:500],  # Use a subset for evaluation 
        rot_inv_test_pairs,
        hash_bit_size=HASH_BIT_SIZE,
        similarity_threshold=SIMILARITY_THRESHOLD,
        model=model,
        test_rotations=True,
    )

    precision, recall, f1_score = metrics.compute_metrics()
    print(
        f"Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test F1 Score: {f1_score:.4f}"
    )

    # Test rotation invariance
    rot_precision, rot_recall, rot_f1, invariance_score = (
        metrics.test_rotation_invariance()
    )
    print(
        f"Rotation test Precision: {rot_precision:.4f}, Rotation test Recall: {rot_recall:.4f}, Rotation test F1: {rot_f1:.4f}"
    )
    print(
        f"Rotation invariance score: {invariance_score:.2f} (closer to 1.0 is better)"
    )
    
    # Create summary visualization for test results
    create_test_summary_visualization(
        precision, recall, f1_score,  
        invariance_score, output_path
    )

    return model, history

if __name__ == "__main__":
    # Print GPU information
    if has_gpu:
        print("Training will use GPU acceleration")
        # Show TensorFlow GPU info
        print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
    else:
        print("Training will run on CPU (no GPU detected)")

    data_path = "dataset/ModelNet_temp"
    output_path = "output/"

    # Load data
    train_data = load_data(data_path, "train")
    test_data = load_data(data_path, "test")

    # Train model
    model, history = train_hashing_model(train_data, test_data, output_path)