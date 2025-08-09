import os
import numpy as np
import trimesh
from sklearn.model_selection import train_test_split
import tensorflow as tf
from collections import defaultdict
import random

from config import POINT_CLOUD_SIZE, BATCH_SIZE, POINT_FEATURES


def load_data(file_path, dataset_type, point_count=1000):
    """
    Load 3D models from filesystem and convert to point clouds
    """
    if dataset_type not in {"train", "test"}:
        raise ValueError("dataset_type must be 'train' or 'test'")
    dataset = {}
    category_counts = defaultdict(int)
    
    for idx, category in enumerate(os.listdir(file_path)):
        if os.path.isdir(os.path.join(file_path, category)):
            file_dir = os.path.join(file_path, category, dataset_type)
            model_paths = [
                os.path.join(file_dir, file)
                for file in os.listdir(file_dir)
                if file.endswith(".off")
            ]
            model_name = [
                file for file in os.listdir(file_dir) if file.endswith(".off")
            ]
            for path in model_paths:
                try:
                    mesh = trimesh.load_mesh(path)  # Load the 3D model
                    point_cloud = mesh.sample(point_count)  # Convert to point cloud
                    dataset[os.path.basename(path)] = (point_cloud, idx, category)
                    category_counts[category] += 1
                except Exception as e:
                    print(f"Error loading mesh {path}: {e}")
    
    # Print dataset statistics with the dataset type
    print(f"\n{dataset_type.upper()} Dataset Statistics:")
    total_models = sum(category_counts.values())
    print(f"Total models: {total_models}")
    print(f"Categories: {len(category_counts)}")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} models ({count/total_models:.1%})")
    
    return dataset


def process_data(dataset, balance_ratio=1.0, max_pairs=None):
    """
    Create pairs of point clouds with similarity labels, 
    with balanced positive and negative examples
    
    Args:
        dataset: Dictionary of point clouds and their categories
        balance_ratio: Ratio of positive to negative pairs (1.0 = equal number)
        max_pairs: Maximum number of pairs to generate (None = no limit)
    """
    # Group models by category for easier positive pair generation
    category_to_models = defaultdict(list)
    for model_name, (point_cloud, idx, category) in dataset.items():
        category_to_models[category].append((model_name, point_cloud, idx))
    
    # Generate positive pairs (same category)
    positive_pairs = []
    for category, models in category_to_models.items():
        if len(models) < 2:
            continue
        
        # Create positive pairs within each category
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model_i_name, pc_i, idx_i = models[i]
                model_j_name, pc_j, idx_j = models[j]
                positive_pairs.append(([pc_i, pc_j], 1))
    
    # Generate negative pairs (different categories)
    negative_pairs = []
    categories = list(category_to_models.keys())
    
    # Calculate how many negative pairs we need
    if balance_ratio == 1.0:
        num_negative_pairs = len(positive_pairs)
    else:
        num_negative_pairs = int(len(positive_pairs) / balance_ratio)
    
    # Limit the total number of both positive and negative pairs if specified
    if max_pairs:
        num_negative_pairs = min(num_negative_pairs, max_pairs // 2)
        positive_pairs = positive_pairs[:max_pairs - num_negative_pairs]
    
    # Generate the required number of negative pairs
    if len(categories) >= 2:
        pairs_created = 0
        attempts = 0
        max_attempts = num_negative_pairs * 3  # Avoid infinite loops
        
        while pairs_created < num_negative_pairs and attempts < max_attempts:
            # Pick two different categories
            cat_i, cat_j = random.sample(categories, 2)
            
            # Pick random models from each category
            if category_to_models[cat_i] and category_to_models[cat_j]:
                model_i = random.choice(category_to_models[cat_i])
                model_j = random.choice(category_to_models[cat_j])
                
                # Create negative pair
                negative_pairs.append(([model_i[1], model_j[1]], 0))
                pairs_created += 1
            
            attempts += 1
    
    # Combine and shuffle
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    # Print statistics
    pos_count = len(positive_pairs)
    neg_count = len(negative_pairs)
    total_count = len(all_pairs)
    
    print(f"\nCreated {total_count} pairs:")
    print(f"  Positive pairs (same class): {pos_count} ({pos_count/total_count:.2%})")
    print(f"  Negative pairs (different class): {neg_count} ({neg_count/total_count:.2%})")
    print(f"  Positive:Negative ratio = 1:{neg_count/pos_count:.2f}" if pos_count > 0 else "  No positive pairs!")
    
    return all_pairs


def create_data_generators(train_pairs, val_pairs, batch_size=BATCH_SIZE):
    """
    Create data generators for training and validation with proper output signatures
    """

    def generate_batch(pairs, batch_size):
        while True:  # Make the generator infinite for training across epochs
            num_samples = len(pairs)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)  # Shuffle on each epoch

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_pairs = [pairs[i] for i in batch_indices]

                # Separate point clouds and labels
                pc1_batch = np.array([pair[0][0] for pair in batch_pairs])
                pc2_batch = np.array([pair[0][1] for pair in batch_pairs])
                labels_batch = np.array([pair[1] for pair in batch_pairs])

                # Return as a tuple of inputs and labels
                yield (pc1_batch, pc2_batch), labels_batch

    # Define the output signature
    output_signature = (
        (
            tf.TensorSpec(
                shape=(None, POINT_CLOUD_SIZE, POINT_FEATURES), dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(None, POINT_CLOUD_SIZE, POINT_FEATURES), dtype=tf.float32
            ),
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_generator(
        lambda: generate_batch(train_pairs, batch_size),
        output_signature=output_signature,
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: generate_batch(val_pairs, batch_size), output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    train_steps = len(train_pairs) // batch_size
    val_steps = len(val_pairs) // batch_size

    return train_dataset, val_dataset, train_steps, val_steps