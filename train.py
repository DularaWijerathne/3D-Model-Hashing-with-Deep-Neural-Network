import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, GPU_MEMORY_LIMIT, USE_MIXED_PRECISION, HASH_BIT_SIZE
from data.data_loader import load_data, process_data, create_data_generators
from data.preprocessing import prepare_data_for_training
from models.hashing_model import create_hashing_model
from utils.losses import HashingLoss
from utils.metrics import HashingMetrics, HashingAccuracy
from inference import HashingInference


# GPU configuration
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth needs to be set before GPUs have been initialized
            for gpu in gpus:
                if GPU_MEMORY_LIMIT is not None:
                    # Limit memory growth to specified value in MB
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT)]
                    )
                    print(f"GPU memory limited to {GPU_MEMORY_LIMIT} MB")
                else:
                    # Or use memory growth option
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("GPU memory growth enabled")

            print(f"Found {len(gpus)} GPU(s)")

            # Set mixed precision policy if enabled
            if USE_MIXED_PRECISION:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"Mixed precision enabled: compute={policy.compute_dtype}, variable={policy.variable_dtype}")

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
    """
    Main function to train the hashing model
    """
    # Process data to create pairs
    train_dataset = process_data(train_data)
    test_dataset = process_data(test_data)

    print(f"Created {len(train_dataset)} training pairs and {len(test_dataset)} testing pairs")

    # Prepare data for training
    train_pairs, val_pairs = prepare_data_for_training(train_dataset)

    # Create data generators
    train_gen, val_gen, train_steps, val_steps = create_data_generators(train_pairs, val_pairs)

    # Enable mixed precision training for faster GPU computation
    if has_gpu:
        print("Enabling mixed precision training for GPU...")
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"Compute dtype: {policy.compute_dtype}")
        print(f"Variable dtype: {policy.variable_dtype}")

    # Create model
    with tf.device('/GPU:0' if has_gpu else '/CPU:0'):
        model = create_hashing_model()

    # Compile model with custom loss and accuracy metric
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # For mixed precision, use loss scaling to prevent underflow
    if has_gpu:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=HashingLoss(batch_size=BATCH_SIZE),
        metrics=[HashingAccuracy(hash_bit_size=HASH_BIT_SIZE)]
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
            monitor='val_loss'
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_path, "logs"),
            histogram_freq=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5
        ),
        HashingMetrics(val_pairs, test_rotations=True)
    ]

    # Training configuration
    training_config = {
        'steps_per_epoch': train_steps,
        'validation_data': val_gen,
        'validation_steps': val_steps,
        'epochs': EPOCHS,
        'callbacks': callbacks
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
        if isinstance(layer, keras.Model) and "PointCloudFeatureExtractor" in layer.name:
            feature_extractor = layer
            break

    if feature_extractor is None and len(model.layers) > 2:
        feature_extractor = model.layers[2]

    if feature_extractor:
        feature_extractor.save(os.path.join(output_path, "feature_extractor.keras"))

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 3, 2)
    plt.plot(history.history['hashing_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_hashing_accuracy'], label='Validation Accuracy')
    plt.title('Hashing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot learning rate if available
    if 'lr' in history.history:
        plt.subplot(1, 3, 3)
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "training_history.png"))
    plt.close()

    # Plot metrics history from the HashingMetrics callback
    metrics_callback = None
    for callback in callbacks:
        if isinstance(callback, HashingMetrics):
            metrics_callback = callback
            break
            
    if metrics_callback and metrics_callback.metrics_history['epoch']:
        plt.figure(figsize=(15, 10))
        
        # Plot mAP
        plt.subplot(2, 2, 1)
        plt.plot(metrics_callback.metrics_history['epoch'], 
                 metrics_callback.metrics_history['mAP'], 
                 'o-', label='mAP')
        if metrics_callback.metrics_history.get('rotation_mAP'):
            plt.plot(metrics_callback.metrics_history['epoch'], 
                     metrics_callback.metrics_history['rotation_mAP'], 
                     'o--', label='Rotation mAP')
        plt.title('Mean Average Precision')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Plot Precision@k
        plt.subplot(2, 2, 2)
        plt.plot(metrics_callback.metrics_history['epoch'], 
                 metrics_callback.metrics_history['precision_at_k'], 
                 'o-', label='Precision@100')
        if metrics_callback.metrics_history.get('rotation_precision'):
            plt.plot(metrics_callback.metrics_history['epoch'], 
                     metrics_callback.metrics_history['rotation_precision'], 
                     'o--', label='Rotation Precision@100')
        plt.title('Precision@100')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Plot Recall@k if available
        if 'recall_at_k' in metrics_callback.metrics_history:
            plt.subplot(2, 2, 3)
            plt.plot(metrics_callback.metrics_history['epoch'], 
                     metrics_callback.metrics_history['recall_at_k'], 
                     'o-', label='Recall@100')
            plt.title('Recall@100')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
        
        # Plot Invariance Score if available
        if 'invariance_score' in metrics_callback.metrics_history:
            plt.subplot(2, 2, 4)
            plt.plot(metrics_callback.metrics_history['epoch'], 
                     metrics_callback.metrics_history['invariance_score'], 
                     'o-', label='Invariance Score')
            plt.title('Rotation Invariance Score')
            plt.xlabel('Epoch')
            plt.ylabel('Score (higher is better)')
            plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "retrieval_metrics.png"))
        plt.close()

    # Evaluate on test set
    print("\n\nEvaluating on test set...")
    test_pairs, _ = prepare_data_for_training(test_dataset)
    test_gen, _, test_steps, _ = create_data_generators(test_pairs, [])

    # Create inference object for evaluation
    inference = HashingInference(model_path=os.path.join(output_path, "hashing_model.keras"))

    # Create metrics object for testing
    metrics = HashingMetrics(test_pairs[:500],  # Use a subset for evaluation
                           feature_extractor=inference.feature_extractor,
                           hash_layer=inference.hash_layer,
                           test_rotations=True)

    mAP, precision_at_k, recall_at_k = metrics.compute_retrieval_metrics()
    print(f"Test mAP: {mAP:.4f}, Test Precision@100: {precision_at_k:.4f}, Test Recall@100: {recall_at_k:.4f}")
    
    # Test rotation invariance
    rotation_mAP, rotation_precision, invariance_score = metrics.test_rotation_invariance()
    print(f"Rotation test mAP: {rotation_mAP:.4f}, Rotation test Precision@100: {rotation_precision:.4f}")
    print(f"Rotation invariance score: {invariance_score:.2f} (closer to 1.0 is better)")

    return model, history


if __name__ == "__main__":
    # Print GPU information
    if has_gpu:
        print("Training will use GPU acceleration")
        # Show TensorFlow GPU info
        print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"GPU device name: {tf.test.gpu_device_name()}")
    else:
        print("Training will run on CPU (no GPU detected)")

    data_path = "dataset/ModelNet_temp"
    output_path = "output/"

    # Load data
    train_data = load_data(data_path, "train")
    test_data = load_data(data_path, "test")

    # Train model
    model, history = train_hashing_model(train_data, test_data, output_path)


# # For testing
# if __name__ == "__main__":
#     # Print GPU information
#     if has_gpu:
#         print("Training will use GPU acceleration")
#         # Show TensorFlow GPU info
#         print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
#         print(f"GPU device name: {tf.test.gpu_device_name()}")
#     else:
#         print("Training will run on CPU (no GPU detected)")

#     # Example usage
#     data_path = "dataset/ModelNet10"
#     output_path = "output/"

#     # Load data
#     data = load_data(data_path, "train")
#     items = list(data.items())
#     random.shuffle(items)
#     temp_data = dict(items[:500])

#     temp_train = dict(list(temp_data.items())[:300])
#     temp_test = dict(list(temp_data.items())[300:])

#     # Train model
#     model, history = train_hashing_model(temp_train, temp_test, output_path)