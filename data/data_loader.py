import os
import numpy as np
import trimesh
from sklearn.model_selection import train_test_split
import tensorflow as tf

from config import POINT_CLOUD_SIZE, BATCH_SIZE


def load_data(file_path, dataset_type, point_count=1000):
    """
    Load 3D models from filesystem and convert to point clouds
    """
    if dataset_type not in {'train', 'test'}:
        raise ValueError("dataset_type must be 'train' or 'test'")
    dataset = {}
    for idx, category in enumerate(os.listdir(file_path)):
        if os.path.isdir(os.path.join(file_path, category)):
            file_dir = os.path.join(file_path, category, dataset_type)
            model_paths = [os.path.join(file_dir, file) for file in os.listdir(file_dir) if file.endswith(".off")]
            model_name = [file for file in os.listdir(file_dir) if file.endswith(".off")]
            for path in model_paths:
                try:
                    mesh = trimesh.load_mesh(path)  # Load the 3D model
                    point_cloud = mesh.sample(point_count)  # Convert to point cloud
                    dataset[os.path.basename(path)] = (point_cloud, idx)
                except Exception as e:
                    print(f"Error loading mesh {path}: {e}")
    return dataset


def process_data(dataset):
    """
    Create pairs of point clouds with similarity labels
    """
    pairs = []
    model_names = list(dataset.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            item_1 = dataset[model_names[i]]
            item_2 = dataset[model_names[j]]
            if item_1[1] == item_2[1]:
                label = 1
            else:
                label = 0

            pair = ([dataset[model_names[i]][0], dataset[model_names[j]][0]], label)
            pairs.append(pair)

    return pairs


def create_data_generators(train_pairs, val_pairs, batch_size=BATCH_SIZE):
    """
    Create data generators for training and validation with proper output signatures
    """
    from config import POINT_CLOUD_SIZE, POINT_FEATURES

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
            tf.TensorSpec(shape=(None, POINT_CLOUD_SIZE, POINT_FEATURES), dtype=tf.float32),
            tf.TensorSpec(shape=(None, POINT_CLOUD_SIZE, POINT_FEATURES), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_generator(
        lambda: generate_batch(train_pairs, batch_size),
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: generate_batch(val_pairs, batch_size),
        output_signature=output_signature
    ).prefetch(tf.data.AUTOTUNE)

    train_steps = len(train_pairs) // batch_size
    val_steps = len(val_pairs) // batch_size

    return train_dataset, val_dataset, train_steps, val_steps