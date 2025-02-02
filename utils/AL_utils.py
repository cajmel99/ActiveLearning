import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def select_lowest_probabieties(probabilities, df, n_samples, labels):
    """
    Split df for df with the n_samples lowest probailities and the rest
    """
    df_with_proba = df
    df_with_proba['probability'] = None
    df_with_proba['labels'] = None

    df_with_proba['probability'] = probabilities
    df_with_proba['labels'] = labels
    df_with_proba = df_with_proba.sort_values(by='probability', ascending=False)
    df_top = df_with_proba[:n_samples]
    X_top = df_top.drop(columns=['probability', 'labels'])
    y_top = df_top['labels']
    df_rest = df_with_proba[n_samples:]
    X_rest = df_rest.drop(columns=['probability', 'labels'])
    y_rest = df_rest['labels']

    return X_top, y_top, X_rest, y_rest

def save_metrics_to_file(folder_name, filename, cycle, precision, recall, f1, cm, accuracy, X_labelled, y_test, positive_class):
    """
    Save metrics only for the positive class (to a CSV file.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    full_path = os.path.join(folder_name, filename)
    
    results = []
    unique_classes = np.unique(y_test)
    
    if positive_class in unique_classes:
        positive_class_index = np.where(unique_classes == positive_class)[0][0]
        
        # Append results only for the positive class
        results.append({
            'Cycle': cycle,
            'Class': positive_class,
            'Precision': precision[positive_class_index],
            'Recall': recall[positive_class_index],
            'F1-score': f1[positive_class_index],
            'Accuracy': accuracy,
            'Confusion Matrix': cm.tolist(),
            'Labelled Samples': X_labelled.shape[0]
        })
    
    if results:  
        results_df = pd.DataFrame(results)
        results_df.to_csv(full_path, mode='a', header=not pd.io.common.file_exists(full_path), index=False)
        print(f"Metrics saved for cycle {cycle} to {full_path}")
    else:
        print(f"No metrics to save for the positive class '{positive_class}' in cycle {cycle}.")

    
def calculate_metrics(y_test, y_pred):
    """
    Caluate metrices
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    return precision, recall, f1, cm, accuracy

def select_samples(probalility, df, n_samples, labels, metric="least_confidence"):
    """
    Select the top n_samples samples with the highest uncertainty
    """

    if len(np.unique(labels)) == 1 or metric == "random_sampling":
        print("Only one class present. Selecting samples randomly.")
        df_sampled = df_with_proba.sample(n=n_samples)
        df_rest = df_with_proba.drop(df_sampled.index)

        X_top = df_sampled.drop(columns=['labels'])
        y_top = df_sampled['labels']

        X_rest = df_rest.drop(columns=['labels'])
        y_rest = df_rest['labels']

        return X_top, y_top, X_rest, y_rest
    
    df_with_proba = df.copy()
    df_with_proba['labels'] = labels

    # Compute the uncertainty scores based on the chosen metric
    if metric == "least_confidence":
        # Least Confidence: Select the samples with the lowest top predicted probabilities
        prob_top = np.max(probalility, axis=1)  
        df_with_proba['uncertainty'] = prob_top

    elif metric == "entropy":
        # Entropy: Calculate the entropy for each sample based on its class probabilities
        entropy_values = -np.sum(probalility * np.log(probalility + 1e-10), axis=1) 
        df_with_proba['uncertainty'] = entropy_values

    elif metric == "margin_sampling":
        # Margin Sampling: Select the samples where the top two predicted probabilities are closest
        top_2_probs = np.partition(probalility, -2, axis=1)[:, -2:]  # Get the top 2 predicted probabilities, Calculate the difference between the top two
        margin = top_2_probs[:, 1] - top_2_probs[:, 0]  
        df_with_proba['uncertainty'] = margin
        
    else:
        raise ValueError(f"Unknown metric: {metric}. Available options are 'least_confidence', 'entropy', 'margin_sampling' or 'random_sampling.")
    
    df_with_proba = df_with_proba.sort_values(by='uncertainty', ascending=False)    
    df_top = df_with_proba[:n_samples]
    X_top = df_top.drop(columns=['uncertainty', 'labels'])
    y_top = df_top['labels']
    
    df_rest = df_with_proba[n_samples:]
    X_rest = df_rest.drop(columns=['uncertainty', 'labels'])
    y_rest = df_rest['labels']
    
    return X_top, y_top, X_rest, y_rest


def basic_active_lerning_flow(X_train, y_train, budget, cycle_budget, model, model_name, dataset_name, X_test, y_test, samples_selection_metric='margin_sampling', test_size=0.9):
  """
  
  """
  # Split data to DL and DU
  X_labelled, X_unlabelled, y_labeled, y_unlabelled = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train)

  # Whole budget
  B = budget
  # Budget per cycle
  b = cycle_budget
  # Number of cycle
  c = 0

  results_folder = 'DETAILED_RESULTS'

  while B>0:
      print(dataset_name)
      if np.isnan(y_labeled).sum() > 0:
        print(f"y_labeled contains NaN values: {np.isnan(y_labeled).sum()} NaNs")
        raise ValueError("y_labeled contains NaN values. Please clean the data.")

      model.fit(X_labelled, y_labeled)
      probalilities = model.predict_proba(X_unlabelled)
      # 2. Select samples based on choosen metrics and ask Oracle
      #class_weights = calculate_dynamic_class_weights_based_on_model(model, X_labelled, y_labeled)
      X_lowest_prob, y_lowest_proba, X_rest, y_rest = select_samples(probalility=probalilities, df=X_unlabelled, n_samples=b, labels=y_unlabelled, metric='margin_sampling')

      # 3. Add samples labelled by Oracle to DL
      X_labelled = pd.concat([X_labelled, X_lowest_prob])
      y_labeled = pd.concat([y_labeled, y_lowest_proba])
      # 4. Update DUL
      X_unlabelled = X_rest
      y_unlabelled = y_rest

      # Calculate accuracy for this cycle
      y_pred = model.predict(X_test)

      metrics_filename = f'{model_name}__{samples_selection_metric}__{dataset_name}__classic_AL.csv'

      # Calculate metrics
      precision, recall, f1, cm, accuracy = calculate_metrics(y_test, y_pred)
      
      # Save the metrics to the file
      save_metrics_to_file(results_folder, metrics_filename, c, precision, recall, f1, cm, accuracy, X_labelled, y_test, positive_class=1)
          
      accuracy = accuracy_score(y_test, y_pred)
      print(f"\nOverall Accuracy: {accuracy:.4f}")
      
      # Update cycle and budget
      c +=1
      B -=b

  return metrics_filename

def select_samples(probalility, df, n_samples, labels, metric):
    """
    Select samples for active learning based on different uncertainty metrics and dynamically calculated class weights.
    """
    # Create a DataFrame to store the probabilities and labels
    df_with_proba = df.copy()
    df_with_proba['labels'] = labels.values


    # Check if only one class is present
    if len(np.unique(labels)) == 1 or metric == "random_sampling":
        df_sampled = df_with_proba.sample(n=n_samples, random_state=42)
        df_rest = df_with_proba.drop(df_sampled.index)

        X_top = df_sampled.drop(columns=['labels'])
        y_top = df_sampled['labels']

        X_rest = df_rest.drop(columns=['labels'])
        y_rest = df_rest['labels']

        return X_top, y_top, X_rest, y_rest

    
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
        raise ValueError(f"Unknown metric: {metric}. Available options are 'least_confidence', 'entropy', 'margin_sampling' or 'random_sampling.")

    # Apply class weighting to the uncertainty values if class weights are provided
    # if class_weights:
    #     df_with_proba['uncertainty'] *= df_with_proba['labels'].map(class_weights)
    
    # Sort the samples by the uncertainty (ascending order to get the most uncertain samples)
    df_with_proba = df_with_proba.sort_values(by='uncertainty', ascending=False)
    
    # Select the top n_samples samples with the highest uncertainty
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
    Calculate dynamic class weights as the fraction of misclassified samples in a specific class by the total number of samples in that class.
    """
    y_pred = model.predict(X_labelled)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_labeled, y_pred, labels=np.unique(y_labeled))
    
    # Misclassified samples per class (row sum - diagonal elements)
    misclassified_counts = cm.sum(axis=1) - np.diag(cm)
    
    # Total number of samples per class (row sum)
    total_samples_per_class = cm.sum(axis=1)
    
    # Compute class weights: (misclassified samples in class) / (total samples in class)
    class_weights = {
        label: (misclassified_counts[i] / total_samples_per_class[i]) if total_samples_per_class[i] > 0 else 0
        for i, label in enumerate(np.unique(y_labeled))
    }
    max_weight = max(class_weights.values()) if class_weights else 1
    class_weights = {k: v / max_weight for k, v in class_weights.items()}

    
    return class_weights

def select_samples_weighted(probalility, df, n_samples, labels, metric, class_weights):
    """
    Select samples for active learning based on different uncertainty metrics and dynamically calculated class weights.
    """
    
    df_with_proba = df.copy()
    df_with_proba['labels'] = labels

    if len(np.unique(labels)) == 1 or metric == "random_sampling":
        df_sampled = df_with_proba.sample(n=n_samples)
        df_rest = df_with_proba.drop(df_sampled.index)

        X_top = df_sampled.drop(columns=['labels'])
        y_top = df_sampled['labels']

        X_rest = df_rest.drop(columns=['labels'])
        y_rest = df_rest['labels']

        return X_top, y_top, X_rest, y_rest

    
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
        raise ValueError(f"Unknown metric: {metric}. Available options are 'least_confidence', 'entropy', 'margin_sampling' or 'random_sampling.")

    # Apply class weighting to the uncertainty values
    if class_weights:
        # df_with_proba['uncertainty'] *= df_with_proba['labels'].map(class_weights)
        df_with_proba['uncertainty'] = (df_with_proba['uncertainty'].rank() * 0.5) + (df_with_proba['labels'].map(class_weights).rank() * 0.5)

    
    # Sort the samples by the uncertainty (ascending order to get the most uncertain samples)
    df_with_proba = df_with_proba.sort_values(by='uncertainty', ascending=False)
    
    # Select the top n_samples samples with the highest uncertainty
    df_top = df_with_proba[:n_samples]
    X_top = df_top.drop(columns=['uncertainty', 'labels'])
    y_top = df_top['labels']
    
    df_rest = df_with_proba[n_samples:]
    X_rest = df_rest.drop(columns=['uncertainty', 'labels'])
    y_rest = df_rest['labels']
    
    return X_top, y_top, X_rest, y_rest



def weighted_active_learning(X_train, y_train, budget, cycle_budget, model, model_name, dataset_name, X_test, y_test, samples_selection_metric='margin_sampling', test_size=0.9):
  """
  
  """
  # Split data to DL and DU
  X_labelled, X_unlabelled, y_labeled, y_unlabelled = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train)

  # Whole budget
  B = budget
  # Budget per cycle
  b = cycle_budget
  # Number of cycle
  c = 0

  results_folder = 'DETAILED_RESULTS'

  while B>0:
      print(dataset_name)
      if np.isnan(y_labeled).sum() > 0:
        print(f"y_labeled contains NaN values: {np.isnan(y_labeled).sum()} NaNs")
        raise ValueError("y_labeled contains NaN values. Please clean the data.")

      model.fit(X_labelled, y_labeled)
      probalilities = model.predict_proba(X_unlabelled)
      # 2. Select samples based on choosen metrics and ask Oracle
      class_weights = calculate_dynamic_class_weights_based_on_model(model, X_labelled, y_labeled)
      X_lowest_prob, y_lowest_proba, X_rest, y_rest = select_samples_weighted(probalility=probalilities, df=X_unlabelled, n_samples=b, labels=y_unlabelled, metric='margin_sampling', class_weights=class_weights)

      # 3. Add samples labelled by Oracle to DL
      X_labelled = pd.concat([X_labelled, X_lowest_prob])
      y_labeled = pd.concat([y_labeled, y_lowest_proba])
      # 4. Update DUL
      X_unlabelled = X_rest
      y_unlabelled = y_rest

      # Calculate accuracy for this cycle
      y_pred = model.predict(X_test)

      metrics_filename = f'{model_name}__{samples_selection_metric}__{dataset_name}__custom_classic_AL.csv'

      # Calculate metrics
      precision, recall, f1, cm, accuracy = calculate_metrics(y_test, y_pred)
      
      # Save the metrics to the file
      save_metrics_to_file(results_folder, metrics_filename, c, precision, recall, f1, cm, accuracy, X_labelled, y_test, positive_class=1)
          
      # Print accurracy
      accuracy = accuracy_score(y_test, y_pred)
      print(f"\nOverall Accuracy: {accuracy:.4f}")
      
      # Update cycle and budget
      c +=1
      B -=b

  return metrics_filename