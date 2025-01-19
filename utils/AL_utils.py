import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score


def select_lowest_probabieties(probabilities, df, n_samples, labels):
    """
    Parametres:
     - probabilities: samples probabilities
     - df: DataFrame 
     - n_samples: number of samples
    Return:
     - df_top: DataFrame with samples with the lowest probabilities
     - df_rest: DataFrame contianing rest of samples
    """
    df_with_proba = df.copy()
    df_with_proba['probability'] = None
    df_with_proba['labels'] = None

    df_with_proba['probability'] = probabilities
    df_with_proba['labels'] = labels
    df_with_proba = df_with_proba.sort_values(by='probability', ascending=True)
    df_top = df_with_proba[:n_samples]
    X_top = df_top.drop(columns=['probability', 'labels'])
    y_top = df_top['labels']
    df_rest = df_with_proba[n_samples:]
    X_rest = df_rest.drop(columns=['probability', 'labels'])
    y_rest = df_rest['labels']

    return X_top, y_top, X_rest, y_rest

def save_metrics_to_file(folder_name, filename, cycle, precision, recall, f1, cm, accuracy, X_labelled, y_test):
    # Ensure the folder exists, create it if necessary
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Construct the full path to the file in the specified folder
    full_path = os.path.join(folder_name, filename)
    
    # Create a list to hold the results for this cycle
    results = []
    
    # Append results for each class in the metrics
    for i, class_label in enumerate(np.unique(y_test)):
        results.append({
            'Cycle': cycle,
            'Class': class_label,
            'Precision': precision[i],
            'Recall': recall[i],
            'F1-score': f1[i],
            'Accuracy': accuracy,
            'Confusion Matrix': cm.tolist(),
            'Labelled Samples': X_labelled.shape[0]
        })
    
    # Convert to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    
    # Append to the CSV file (if it exists) or create a new one
    results_df.to_csv(full_path, mode='a', header=not pd.io.common.file_exists(full_path), index=False)
    print(f"Metrics saved for cycle {cycle} to {full_path}")

    
# Function to calculate metrics
def calculate_metrics(y_test, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    return precision, recall, f1, cm, accuracy

def select_samples(probalility, df, n_samples, labels, metric="least_confidence"):
    """
    Select samples for active learning based on different uncertainty metrics.
    
    Parameters:
    - probalility: Array of probabilities for each sample (shape: [n_samples, n_classes]).
    - df: The DataFrame containing the feature data.
    - n_samples: The number of samples to select.
    - labels: The true labels for the unlabelled samples.
    - metric: The uncertainty metric to use. Options are 'least_confidence', 'entropy', or 'margin_sampling'.
    
    Returns:
    - X_top: The feature data for the selected samples.
    - y_top: The labels for the selected samples.
    - X_rest: The feature data for the remaining samples.
    - y_rest: The labels for the remaining samples.
    """
    
    # Create a DataFrame to store the probabilities and labels
    df_with_proba = df.copy()
    df_with_proba['labels'] = labels

    # Compute the uncertainty scores based on the chosen metric
    if metric == "least_confidence":
        # Least Confidence: Select the samples with the lowest top predicted probabilities
        prob_top = np.max(probalility, axis=1)  # Get the top probability for each sample
        df_with_proba['uncertainty'] = prob_top

    elif metric == "entropy":
        # Entropy: Calculate the entropy for each sample based on its class probabilities
        entropy_values = -np.sum(probalility * np.log(probalility + 1e-10), axis=1)  # Adding small epsilon to avoid log(0)
        df_with_proba['uncertainty'] = entropy_values

    elif metric == "margin_sampling":
        # Margin Sampling: Select the samples where the top two predicted probabilities are closest
        top_2_probs = np.partition(probalility, -2, axis=1)[:, -2:]  # Get the top 2 predicted probabilities
        margin = top_2_probs[:, 1] - top_2_probs[:, 0]  # Calculate the difference between the top two
        df_with_proba['uncertainty'] = margin
        
    else:
        raise ValueError(f"Unknown metric: {metric}. Available options are 'least_confidence', 'entropy', 'margin_sampling'.")
    
    # Sort the samples by the uncertainty (ascending order to get the most uncertain samples)
    df_with_proba = df_with_proba.sort_values(by='uncertainty', ascending=True)
    
    # Select the top `n_samples` samples with the highest uncertainty
    df_top = df_with_proba[:n_samples]
    X_top = df_top.drop(columns=['uncertainty', 'labels'])
    y_top = df_top['labels']
    
    # Select the remaining samples
    df_rest = df_with_proba[n_samples:]
    X_rest = df_rest.drop(columns=['uncertainty', 'labels'])
    y_rest = df_rest['labels']
    
    return X_top, y_top, X_rest, y_rest

def calculate_dynamic_class_weights_based_on_model(model, X_labelled, y_labeled):
    """
    Calculate dynamic class weights based on the performance of the model on the current labelled data.
    """
    y_pred = model.predict(X_labelled)
    
    # Find misclassifications (or measure uncertainty)
    misclassifications = (y_pred != y_labeled)
    
    # Calculate the frequency of misclassifications per class
    class_misclassifications = {class_label: np.sum(misclassifications[y_labeled == class_label]) 
                                for class_label in np.unique(y_labeled)}
    
    # Compute class weights as the inverse of misclassifications (more misclassified = higher weight)
    total_misclassifications = sum(class_misclassifications.values())
    class_weights = {class_label: (total_misclassifications / (class_misclassifications[class_label] + 1)) 
                     for class_label in class_misclassifications}
    
    return class_weights

def select_samples_weighted(probalility, df, n_samples, labels, metric, class_weights):
    """
    Select samples for active learning based on different uncertainty metrics and dynamically calculated class weights.

    Parameters:
    - probalility: Array of probabilities for each sample (shape: [n_samples, n_classes]).
    - df: The DataFrame containing the feature data.
    - n_samples: The number of samples to select.
    - labels: The true labels for the unlabelled samples.
    - metric: The uncertainty metric to use. Options are 'least_confidence', 'entropy', or 'margin_sampling'.
    - calculate_class_weights_fn: A function to dynamically calculate class weights.

    Returns:
    - X_top: The feature data for the selected samples.
    - y_top: The labels for the selected samples.
    - X_rest: The feature data for the remaining samples.
    - y_rest: The labels for the remaining samples.
    """
    
    # Create a DataFrame to store the probabilities and labels
    df_with_proba = df.copy()
    df_with_proba['labels'] = labels
    
    # Compute the uncertainty scores based on the chosen metric
    if metric == "least_confidence":
        # Least Confidence: Select the samples with the lowest top predicted probabilities
        prob_top = np.max(probalility, axis=1)  # Get the top probability for each sample
        df_with_proba['uncertainty'] = prob_top

    elif metric == "entropy":
        entropy_values = -np.sum(probalility * np.log(probalility + 1e-10), axis=1)  # Adding small epsilon to avoid log(0)
        df_with_proba['uncertainty'] = entropy_values

    elif metric == "margin_sampling":
        top_2_probs = np.partition(probalility, -2, axis=1)[:, -2:]  # Get the top 2 predicted probabilities
        margin = top_2_probs[:, 1] - top_2_probs[:, 0]  # Calculate the difference between the top two
        df_with_proba['uncertainty'] = margin
    else:
        raise ValueError(f"Unknown metric: {metric}. Available options are 'least_confidence', 'entropy', 'margin_sampling'.")

    # Apply class weighting to the uncertainty values if class weights are provided
    if class_weights:
        df_with_proba['uncertainty'] *= df_with_proba['labels'].map(class_weights)
    
    # Sort the samples by the uncertainty (ascending order to get the most uncertain samples)
    df_with_proba = df_with_proba.sort_values(by='uncertainty', ascending=True)
    
    # Select the top n_samples samples with the highest uncertainty
    df_top = df_with_proba[:n_samples]
    X_top = df_top.drop(columns=['uncertainty', 'labels'])
    y_top = df_top['labels']
    
    # Select the remaining samples
    df_rest = df_with_proba[n_samples:]
    X_rest = df_rest.drop(columns=['uncertainty', 'labels'])
    y_rest = df_rest['labels']
    
    return X_top, y_top, X_rest, y_rest
