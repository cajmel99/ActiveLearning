import os
import pandas as pd
import matplotlib.pyplot as plt

def load_data(folder_name, file_name, header=None):
    """Load csv file in df"""
    df = pd.read_csv(os.path.join(os.getcwd(), folder_name, file_name), header)

    return df

def plot_metrics_from_file(filename):
    # Read the metrics 
    metrics_df = pd.read_csv(filename)
    
    # Plot Precision, Recall, and F1-score for each class over cycles
    cycles = metrics_df['Cycle'].unique()

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    for class_label in metrics_df['Class'].unique():
        class_data = metrics_df[metrics_df['Class'] == class_label]

        axes[0].plot(class_data['Cycle'], class_data['Precision'], label=f"Class {class_label}", marker='o')
        axes[1].plot(class_data['Cycle'], class_data['Recall'], label=f"Class {class_label}", marker='o')
        axes[2].plot(class_data['Cycle'], class_data['F1-score'], label=f"Class {class_label}", marker='o')

    # Labels and titles
    axes[0].set_title('Precision per Cycle')
    axes[0].set_xlabel('Cycle')
    axes[0].set_ylabel('Precision')
    axes[0].legend()

    axes[1].set_title('Recall per Cycle')
    axes[1].set_xlabel('Cycle')
    axes[1].set_ylabel('Recall')
    axes[1].legend()

    axes[2].set_title('F1-Score per Cycle')
    axes[2].set_xlabel('Cycle')
    axes[2].set_ylabel('F1-Score')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # Plot Overall Accuracy per Cycle
    plt.figure(figsize=(8, 5))
    accuracy_data = metrics_df.groupby('Cycle')['Accuracy'].mean()
    plt.plot(accuracy_data.index, accuracy_data, marker='o', label='Accuracy', color='blue')
    plt.title('Overall Accuracy per Cycle')
    plt.xlabel('Cycle')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
