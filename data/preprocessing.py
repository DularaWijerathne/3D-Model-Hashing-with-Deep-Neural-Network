import numpy as np
from sklearn.model_selection import train_test_split

from config import POINT_CLOUD_SIZE


def normalize_point_cloud(point_cloud):
    """
    Normalize point cloud to have zero mean, unit sphere radius,
    and standard orientation via PCA
    """
    # Ensure we have the right number of points
    if point_cloud.shape[0] != POINT_CLOUD_SIZE:
        if point_cloud.shape[0] > POINT_CLOUD_SIZE:
            # Random sampling
            indices = np.random.choice(
                point_cloud.shape[0], POINT_CLOUD_SIZE, replace=False
            )
            point_cloud = point_cloud[indices]
        else:
            # Padding with repeated points
            padding = np.random.choice(
                point_cloud.shape[0], POINT_CLOUD_SIZE - point_cloud.shape[0]
            )
            point_cloud = np.vstack([point_cloud, point_cloud[padding]])

    # Center the point cloud
    centroid = np.mean(point_cloud, axis=0)
    centered_point_cloud = point_cloud - centroid

    # PCA for orientation normalization
    covariance = np.cov(centered_point_cloud, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Transform point cloud to align with principal components
    aligned_point_cloud = centered_point_cloud @ eigenvectors

    # Scale to unit sphere
    furthest_distance = np.max(np.sqrt(np.sum(aligned_point_cloud**2, axis=1)))
    normalized_point_cloud = aligned_point_cloud / (furthest_distance + 1e-10)

    return normalized_point_cloud


def random_rotation_matrix():
    """
    Generate a random rotation matrix for data augmentation
    """
    # Random rotation matrix - Euler angles method
    angles = np.random.uniform(0, 2 * np.pi, size=3)

    # Rotation matrices around x, y, z axes
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angles[0]), -np.sin(angles[0])],
            [0, np.sin(angles[0]), np.cos(angles[0])],
        ]
    )

    Ry = np.array(
        [
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])],
        ]
    )

    Rz = np.array(
        [
            [np.cos(angles[2]), -np.sin(angles[2]), 0],
            [np.sin(angles[2]), np.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )

    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    return R


def augment_point_cloud(point_cloud, jitter_sigma=0.01, rotate=True):
    """
    Apply data augmentation to point cloud:
    - Random rotation
    - Random jitter
    """
    augmented_pc = point_cloud.copy()

    # Apply random rotation if enabled
    if rotate:
        R = random_rotation_matrix()
        augmented_pc = augmented_pc @ R

    # Add random jitter
    augmented_pc = augmented_pc + np.random.normal(
        0, jitter_sigma, size=augmented_pc.shape
    )

    return augmented_pc


def prepare_data_for_training(pairs, augment=True):
    """
    Convert the list of pairs to format suitable for model training
    with optional data augmentation
    """
    # Normalize point clouds
    normalized_pairs = []
    for point_clouds, label in pairs:
        pc1 = normalize_point_cloud(point_clouds[0])
        pc2 = normalize_point_cloud(point_clouds[1])

        # Apply augmentation if enabled
        if augment:
            if np.random.random() < 0.5:  # 50% chance to augment
                pc1 = augment_point_cloud(pc1)
            if np.random.random() < 0.5:  # 50% chance to augment
                pc2 = augment_point_cloud(pc2)

        normalized_pairs.append(([pc1, pc2], label))

    # Split into train and validation sets
    train_pairs, val_pairs = train_test_split(
        normalized_pairs, test_size=0.2, random_state=42
    )

    return train_pairs, val_pairs
