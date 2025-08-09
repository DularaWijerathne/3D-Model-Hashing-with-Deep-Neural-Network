import os
import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history, output_path):
    """Create enhanced visualizations of training history"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a compact figure with adjusted spacing
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('3D Model Hashing Training History', fontsize=14, y=1.02)
    
    # Plot loss
    ax = axes[0]
    ax.plot(history.history['loss'], '-', color='black', label='Training Loss', linewidth=2)
    ax.plot(history.history['val_loss'], '--', color='black', label='Validation Loss', linewidth=2)
    ax.set_title('Loss Evolution', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss', fontsize=10)
    ax.legend(fontsize=9, frameon=True, facecolor='white', edgecolor='black')
    
    # Plot accuracy
    ax = axes[1]
    ax.plot(history.history['hashing_accuracy'], '-', color='black', label='Training Accuracy', linewidth=2)
    ax.plot(history.history['val_hashing_accuracy'], '--', color='black', label='Validation Accuracy', linewidth=2)
    ax.set_title('Hashing Accuracy', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=10)
    ax.set_ylim([0, 1])
    ax.legend(fontsize=9, frameon=True, facecolor='white', edgecolor='black')
    
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_path, "training_history.png"), dpi=600, 
               bbox_inches='tight', facecolor='white', edgecolor='black')
    plt.close()
    

def plot_metrics_history(metrics_callback, output_path):
    """Create enhanced visualizations of hashing metrics history"""
    if metrics_callback and metrics_callback.metrics_history["epoch"]:
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create compact figure with 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('3D Model Hashing Performance Metrics', fontsize=14, y=1.05)
        
        line_styles = {
            'overall': ('-', 'o', '#333333', 1.5),
            'rotation': ('--', 's', '#666666', 1.5),
            'invariance': ('-.', '*', '#444444', 1.5)
        }
        
        # Plot Precision
        ax = axes[0]
        style, marker, color, lw = line_styles['overall']
        ax.plot(
            metrics_callback.metrics_history["epoch"],
            metrics_callback.metrics_history["precision"],
            style, color=color, label='Overall Precision', 
            linewidth=lw, marker=marker, markersize=5
        )
        if metrics_callback.metrics_history.get("rotation_precision"):
            style, marker, color, lw = line_styles['rotation']
            ax.plot(
                metrics_callback.metrics_history["epoch"],
                metrics_callback.metrics_history["rotation_precision"],
                style, color=color, label='Rotation Precision', 
                linewidth=lw, marker=marker, markersize=4
            )
        ax.set_title('Precision', fontsize=12)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=9)
        
        # Plot Recall
        ax = axes[1]
        style, marker, color, lw = line_styles['overall']
        ax.plot(
            metrics_callback.metrics_history["epoch"],
            metrics_callback.metrics_history["recall"],
            style, color=color, label='Overall Recall', 
            linewidth=lw, marker=marker, markersize=5
        )
        if metrics_callback.metrics_history.get("rotation_recall"):
            style, marker, color, lw = line_styles['rotation']
            ax.plot(
                metrics_callback.metrics_history["epoch"],
                metrics_callback.metrics_history["rotation_recall"],
                style, color=color, label='Rotation Recall', 
                linewidth=lw, marker=marker, markersize=4
            )
        ax.set_title('Recall', fontsize=12)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Recall', fontsize=10)
        ax.set_ylim([0.3, 1])
        ax.legend(fontsize=9)
        
        # Plot F1 Score
        ax = axes[2]
        style, marker, color, lw = line_styles['overall']
        ax.plot(
            metrics_callback.metrics_history["epoch"],
            metrics_callback.metrics_history["f1_score"],
            style, color=color, label='Overall F1', 
            linewidth=lw, marker=marker, markersize=5
        )
        if metrics_callback.metrics_history.get("rotation_f1"):
            style, marker, color, lw = line_styles['rotation']
            ax.plot(
                metrics_callback.metrics_history["epoch"],
                metrics_callback.metrics_history["rotation_f1"],
                style, color=color, label='Rotation F1', 
                linewidth=lw, marker=marker, markersize=4
            )
        ax.set_title('F1 Score', fontsize=12)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.set_ylim([0, 1])
        ax.legend(fontsize=9)
        
        plt.tight_layout(pad=2.0)
        plt.savefig(os.path.join(output_path, "threshold_metrics.png"), 
                   dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()

    
def create_test_summary_visualization(precision, recall, f1_score, invariance_score, output_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create compact figure with adjusted layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle('3D Model Hashing Test Results Summary', fontsize=14, y=1.05)
    
    # Bar chart
    metrics = ['Precision', 'Recall', 'F1 Score']
    overall_values = [precision, recall, f1_score]
        
    x = np.arange(len(metrics))
    width = 0.6
        
    ax1.bar(x, overall_values, width, label='Overall', 
            color='#f0f0f0', edgecolor='#888888', hatch='...', linewidth=1)
        
    ax1.set_title('Performance Metrics', fontsize=12)
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('Score', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=10)
    for i, v in enumerate(overall_values):
        ax1.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)
    
    # Gauge visualization
    ax2.set_title('Rotation Invariance Score', fontsize=12)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.axis('off')

    radius = 0.8
    wedge_linewidth = 1

    # Zones
    zone1_end = 0.15 * 180
    ax2.add_patch(plt.matplotlib.patches.Wedge((0, 0), radius, 0, zone1_end, 
                                            width=0.3, facecolor='#f8f8f8', edgecolor='#aaaaaa', 
                                            linewidth=wedge_linewidth))

    zone2_end = 0.4 * 180
    ax2.add_patch(plt.matplotlib.patches.Wedge((0, 0), radius, zone1_end, zone2_end, 
                                            width=0.3, facecolor='#e8e8e8', edgecolor='#999999', 
                                            linewidth=wedge_linewidth))

    zone3_end = 1.0 * 180
    ax2.add_patch(plt.matplotlib.patches.Wedge((0, 0), radius, zone2_end, zone3_end, 
                                            width=0.3, facecolor='#d8d8d8', edgecolor='#888888', 
                                            linewidth=wedge_linewidth))

    # Labels
    ax2.text(-0.7, -0.2, "0.0", fontsize=9, ha='center', color='#555555')
    ax2.text(0, -0.2, "0.5", fontsize=9, ha='center', color='#555555')
    ax2.text(0.7, -0.2, "1.0", fontsize=9, ha='center', color='#555555')

    ax2.text(-0.6, -0.5, "Poor", fontsize=10, ha='center', weight='bold', color='#555555')
    ax2.text(0, -0.5, "Average", fontsize=10, ha='center', weight='bold', color='#555555')
    ax2.text(0.6, -0.5, "Excellent", fontsize=10, ha='center', weight='bold', color='#555555')

    # Needle
    needle_value = min(max(invariance_score, 0), 1)
    needle_angle = (1 - needle_value) * np.pi
    needle_length = 0.6
    x_needle = needle_length * np.cos(needle_angle)
    y_needle = needle_length * np.sin(needle_angle)
    ax2.plot([0, x_needle], [0, y_needle], color='#666666', linewidth=2)

    ax2.add_patch(plt.matplotlib.patches.Circle((0, 0), 0.05, facecolor='#777777'))

    ax2.text(0, 0.3, f"Score: {invariance_score:.4f}", fontsize=12, ha='center', 
            weight='bold', color='#444444', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#dddddd'))

    # Interpretation
    if invariance_score >= 0.95:
        interpretation = "Excellent rotation invariance"
    elif invariance_score >= 0.8:
        interpretation = "Good rotation invariance"
    else:
        interpretation = "Limited rotation invariance"
        
    ax2.text(0, -0.85, interpretation, fontsize=11, ha='center', 
            weight='bold', color='#444444',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='#dddddd'))

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_path, "test_results_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()