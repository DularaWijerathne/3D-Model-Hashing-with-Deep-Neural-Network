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
from data.preprocessing import prepare_data_for_training
from models.hashing_model import create_hashing_model
from utils.losses import HashingLoss
from utils.metrics import HashingMetrics, HashingAccuracy
from inference import HashingInference

#  Suppress warnings 
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


def plot_training_history(history, output_path):
    """
    Create enhanced visualizations of training history
    """
    # Set a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure with 2 rows of plots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('3D Model Hashing Training History', fontsize=16, y=0.98)
    
    # Plot loss
    ax = axes[0, 0]
    ax.plot(history.history['loss'], 'o-', color='#3498db', label='Training Loss', linewidth=2)
    ax.plot(history.history['val_loss'], 'o-', color='#e74c3c', label='Validation Loss', linewidth=2)
    ax.set_title('Loss Evolution', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10)
    
    # Plot accuracy
    ax = axes[0, 1]
    ax.plot(history.history['hashing_accuracy'], 'o-', color='#2ecc71', label='Training Accuracy', linewidth=2)
    ax.plot(history.history['val_hashing_accuracy'], 'o-', color='#9b59b6', label='Validation Accuracy', linewidth=2)
    ax.set_title('Hashing Accuracy', fontsize=14)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=10)
    
    # Plot learning rate if available
    ax = axes[1, 0]
    if "lr" in history.history:
        ax.plot(history.history["lr"], 'o-', color='#f39c12', linewidth=2)
        ax.set_title('Learning Rate Schedule', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.6)
    else:
        ax.set_visible(False)
    
    # Add model architecture summary (if space needed for additional plots)
    ax = axes[1, 1]
    ax.axis('off')
    model_info = (
        f"Model Configuration:\n"
        f"Hash Code Size: {HASH_BIT_SIZE} bits\n"
        f"Batch Size: {BATCH_SIZE}\n"
        f"Learning Rate: {LEARNING_RATE}\n"
        f"Similarity Threshold: {SIMILARITY_THRESHOLD}"
    )
    ax.text(0.5, 0.5, model_info, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#f8f9fa', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(os.path.join(output_path, "training_history.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_history(metrics_callback, output_path):
    """
    Create enhanced visualizations of hashing metrics history
    """
    if metrics_callback and metrics_callback.metrics_history["epoch"]:
        # Set a clean, modern style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create figure with 2 rows and 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('3D Model Hashing Performance Metrics', fontsize=16, y=0.98)
        
        # Plot Precision
        ax = axes[0, 0]
        ax.plot(
            metrics_callback.metrics_history["epoch"],
            metrics_callback.metrics_history["precision"],
            'o-', color='#3498db', label='Standard Precision', linewidth=2, markersize=8
        )
        if metrics_callback.metrics_history.get("rotation_precision"):
            ax.plot(
                metrics_callback.metrics_history["epoch"],
                metrics_callback.metrics_history["rotation_precision"],
                'o--', color='#e74c3c', label='Rotation Precision', linewidth=2, markersize=6
            )
        ax.set_title('Precision', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=10)
        
        # Plot Recall
        ax = axes[0, 1]
        ax.plot(
            metrics_callback.metrics_history["epoch"],
            metrics_callback.metrics_history["recall"],
            'o-', color='#2ecc71', label='Standard Recall', linewidth=2, markersize=8
        )
        if metrics_callback.metrics_history.get("rotation_recall"):
            ax.plot(
                metrics_callback.metrics_history["epoch"],
                metrics_callback.metrics_history["rotation_recall"],
                'o--', color='#9b59b6', label='Rotation Recall', linewidth=2, markersize=6
            )
        ax.set_title('Recall', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=10)
        
        # Plot F1 Score
        ax = axes[1, 0]
        ax.plot(
            metrics_callback.metrics_history["epoch"],
            metrics_callback.metrics_history["f1_score"],
            'o-', color='#f39c12', label='Standard F1', linewidth=2, markersize=8
        )
        if metrics_callback.metrics_history.get("rotation_f1"):
            ax.plot(
                metrics_callback.metrics_history["epoch"],
                metrics_callback.metrics_history["rotation_f1"],
                'o--', color='#1abc9c', label='Rotation F1', linewidth=2, markersize=6
            )
        ax.set_title('F1 Score', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('F1 Score', fontsize=12)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=10)
        
        # Plot Invariance Score if available
        ax = axes[1, 1]
        if "invariance_score" in metrics_callback.metrics_history:
            ax.plot(
                metrics_callback.metrics_history["epoch"],
                metrics_callback.metrics_history["invariance_score"],
                'o-', color='#8e44ad', label='Rotation Invariance', linewidth=2, markersize=8
            )
            ax.set_title('Rotation Invariance Score', fontsize=14)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Score (higher is better)', fontsize=12)
            
            # Add a target reference line at 1.0
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect Invariance')
            
            # Add shaded regions for interpretation
            ax.axhspan(0.95, 1.05, alpha=0.2, color='green', label='Excellent')
            ax.axhspan(0.8, 0.95, alpha=0.2, color='yellow')
            ax.axhspan(0, 0.8, alpha=0.1, color='red')
            
            # Add text labels for the regions
            ax.text(metrics_callback.metrics_history["epoch"][-1] * 0.9, 1.02, 'Excellent', fontsize=9, color='green')
            ax.text(metrics_callback.metrics_history["epoch"][-1] * 0.9, 0.85, 'Good', fontsize=9, color='orange')
            ax.text(metrics_callback.metrics_history["epoch"][-1] * 0.9, 0.6, 'Needs Improvement', fontsize=9, color='red')
            
            ax.legend(fontsize=10)
        else:
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(output_path, "threshold_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()


def train_hashing_model(train_data, test_data, output_path):
    """
    Main function to train the hashing model
    """
    # Process data to create pairs
    train_dataset = process_data(train_data)
    test_dataset = process_data(test_data)

    print(
        f"Created {len(train_dataset)} training pairs and {len(test_dataset)} testing pairs"
    )

    # Prepare data for training
    train_pairs, val_pairs = prepare_data_for_training(train_dataset)

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
        loss=HashingLoss(batch_size=BATCH_SIZE),
        metrics=[
            HashingAccuracy(
                hash_bit_size=HASH_BIT_SIZE, similarity_threshold=SIMILARITY_THRESHOLD
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
            monitor="val_loss", patience=20, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        HashingMetrics(
            val_pairs,
            hash_bit_size=HASH_BIT_SIZE,
            test_rotations=True,
            similarity_threshold=SIMILARITY_THRESHOLD,
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

    # Create summary visualization of test results
    def create_test_summary_visualization(precision, recall, f1_score, rot_precision, rot_recall, rot_f1, invariance_score):
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle('3D Model Hashing Test Results Summary', fontsize=16, y=0.98)
        
        # Create bar chart comparing standard vs rotation metrics
        metrics = ['Precision', 'Recall', 'F1 Score']
        standard_values = [precision, recall, f1_score]
        rotation_values = [rot_precision, rot_recall, rot_f1]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, standard_values, width, label='Standard', color='#3498db')
        ax1.bar(x + width/2, rotation_values, width, label='Rotation', color='#e74c3c')
        
        ax1.set_title('Performance Metrics Comparison', fontsize=14)
        ax1.set_ylim([0, 1])
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, fontsize=11)
        for i, v in enumerate(standard_values):
            ax1.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
        for i, v in enumerate(rotation_values):
            ax1.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
        ax1.legend(fontsize=10)
        
        # Create invariance score gauge
        ax2.set_title('Rotation Invariance Score', fontsize=14)
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax2.axis('off')
        
        # Create a gauge-like visualization
        theta = np.linspace(0, 180, 100) * np.pi / 180
        
        # Draw colored background zones for gauge
        radius = 0.8
        wedge_linewidth = 2
        
        # Red zone (0-0.6)
        red_end = 0.6 * 180
        ax2.add_patch(plt.matplotlib.patches.Wedge((0, 0), radius, 0, red_end, 
                                                  width=0.3, facecolor='#ffcccc', edgecolor='#e74c3c', 
                                                  linewidth=wedge_linewidth, alpha=0.7))
        
        # Yellow zone (0.6-0.85)
        yellow_end = 0.85 * 180
        ax2.add_patch(plt.matplotlib.patches.Wedge((0, 0), radius, red_end, yellow_end, 
                                                  width=0.3, facecolor='#ffffcc', edgecolor='#f39c12', 
                                                  linewidth=wedge_linewidth, alpha=0.7))
        
        # Green zone (0.85-1.0)
        green_end = 1.0 * 180
        ax2.add_patch(plt.matplotlib.patches.Wedge((0, 0), radius, yellow_end, green_end, 
                                                  width=0.3, facecolor='#ccffcc', edgecolor='#2ecc71', 
                                                  linewidth=wedge_linewidth, alpha=0.7))
        
        # Add labels
        ax2.text(-0.7, -0.2, "0.0", fontsize=10, ha='center')
        ax2.text(0, -0.2, "0.5", fontsize=10, ha='center')
        ax2.text(0.7, -0.2, "1.0", fontsize=10, ha='center')
        
        ax2.text(-0.6, -0.5, "Poor", fontsize=12, ha='center', color='#e74c3c')
        ax2.text(0, -0.5, "Average", fontsize=12, ha='center', color='#f39c12')
        ax2.text(0.6, -0.5, "Excellent", fontsize=12, ha='center', color='#2ecc71')
        
        # Add needle pointer to show value
        needle_value = min(max(invariance_score, 0), 1)  # Clamp value between 0 and 1
        needle_angle = needle_value * 180 * np.pi / 180
        needle_length = 0.6
        x_needle = needle_length * np.cos(needle_angle - np.pi/2)
        y_needle = needle_length * np.sin(needle_angle - np.pi/2)
        ax2.plot([0, x_needle], [0, y_needle], color='black', linewidth=3)
        
        # Add center circle
        ax2.add_patch(plt.matplotlib.patches.Circle((0, 0), 0.05, facecolor='black'))
        
        # Add text with actual score
        ax2.text(0, 0.3, f"Score: {invariance_score:.4f}", fontsize=14, ha='center', 
                weight='bold', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Add interpretation text
        if invariance_score >= 0.95:
            interpretation = "Excellent rotation invariance"
            color = '#2ecc71'
        elif invariance_score >= 0.8:
            interpretation = "Good rotation invariance"
            color = '#f39c12'
        else:
            interpretation = "Limited rotation invariance"
            color = '#e74c3c'
            
        ax2.text(0, -0.85, interpretation, fontsize=12, ha='center', color=color,
                weight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(os.path.join(output_path, "test_results_summary.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # Evaluate on test set
    print("\n\nEvaluating on test set...")
    test_pairs, _ = prepare_data_for_training(test_dataset)
    test_gen, _, test_steps, _ = create_data_generators(test_pairs, [])

    # Create inference object for evaluation
    inference = HashingInference(
        model_path=os.path.join(output_path, "hashing_model.keras")
    )

    # Create metrics object for testing
    metrics = HashingMetrics(
        test_pairs[:500],  # Use a subset for evaluation
        hash_bit_size=HASH_BIT_SIZE,
        feature_extractor=inference.feature_extractor,
        hash_layer=inference.hash_layer,
        test_rotations=True,
        similarity_threshold=SIMILARITY_THRESHOLD,
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
        rot_precision, rot_recall, rot_f1, 
        invariance_score
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