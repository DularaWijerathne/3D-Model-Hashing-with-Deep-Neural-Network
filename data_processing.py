import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from menuinst.platforms.win_utils.knownfolders import folder_path
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from pyntcloud import PyntCloud


def load_data(file_path, dataset_type):
    if dataset_type not in {'train', 'test'}:
        raise ValueError("dataset_type must be 'train' or 'test'")

    dataset = []
    for idx, category in enumerate(os.listdir(file_path)):
        if os.path.isdir(os.path.join(file_path, category)):
            file_dir = os.path.join(file_path, category, dataset_type)
            model_paths = [os.path.join(file_dir, file) for file in os.listdir(file_dir) if file.endswith(".off")]
            for path in model_paths:
                mesh = trimesh.load_mesh(path) # Load the 3D model
                point_cloud = mesh.sample(1000) # Convert to point cloud
                dataset.append((point_cloud, idx))

    return dataset


def visualize_model(model):
    centroid = np.mean(model, axis=0)

    # Plot the point cloud and centroid
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(model[:, 0], model[:, 1], model[:, 2], s=1, label="Point Cloud")
    ax.scatter(centroid[0], centroid[1], centroid[2], color='red', s=10, label="Centroid")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()



#testing
path = "dataset/ModelNet10"
gg = load_data(path, 'train')
print(gg)

visualize_model(gg[0][0])





