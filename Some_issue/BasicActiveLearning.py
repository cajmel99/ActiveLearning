import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class ActiveLearningPipeline:
    def __init__(self, model, model_name, budget, cycle_budget, samples_selection_metric, results_folder='results'):
        self.model = model
        self.model_name = model_name
        self.budget = budget
        self.cycle_budget = cycle_budget
        self.samples_selection_metric = samples_selection_metric
        self.results_folder = results_folder

        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def load_data(self, folder_name, file_name, header=None):
        """Load csv file in DataFrame"""
        df = pd.read_csv(os.path.join(os.getcwd(), folder_name, file_name), header=header)
        return df

    def preprocess_features(self, X_df):
        """Convert catgorical columns to numerical and normalize them"""
        X_df_copy = X_df.copy()
        for col in X_df_copy.columns:
            if X_df_copy.loc[:, col].dtype == 'object':  
                le = LabelEncoder()
                X_df_copy.loc[:, col] = le.fit_transform(X_df_copy[col])

        scaler = MinMaxScaler()
        X_normalized = scaler.fit_transform(X_df_copy)

        X_normalized = pd.DataFrame(X_normalized, columns=X_df_copy.columns)

        return X_normalized

    def select_samples(self, probabilities, df, n_samples, labels):
        print(len(labels))
        """
        Select samples based on the results from acquisition function
        """
        #df_with_proba = df.copy()
        print(df_with_proba.shape)
        print(labels.unique())
        df['labels'] = labels
    

        if self.samples_selection_metric == "least_confidence":
            prob_top = np.max(probabilities, axis=1)
            df_with_proba['uncertainty'] = 1 - prob_top
        elif self.samples_selection_metric == "entropy":
            entropy_values = -np.sum(probabilities * np.log(probabilities + 1e-10), axis=1)
            df_with_proba['uncertainty'] = entropy_values
        elif self.samples_selection_metric == "margin_sampling":
            top_2_probs = np.partition(probabilities, -2, axis=1)[:, -2:]
            margin = top_2_probs[:, 1] - top_2_probs[:, 0]
            df_with_proba['uncertainty'] = margin
        else:
            raise ValueError(f"Unknown metric: {self.samples_selection_metric}")

        df_with_proba = df_with_proba.sort_values(by='uncertainty', ascending=True)
        df_sampled = df_with_proba[:n_samples]
        df_rest = df_with_proba[n_samples:]

        X_top = df_sampled.drop(columns=['uncertainty', 'labels'])
        y_top = df_sampled['labels']

        X_rest = df_rest.drop(columns=['uncertainty', 'labels'])
        y_rest = df_rest['labels']

        return X_top, y_top, X_rest, y_rest

    def calculate_metrics(self, y_test, y_pred):
        """ Calculate metrics """
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        return precision, recall, f1, cm, accuracy

    def save_metrics_to_file(self, filename, cycle, precision, recall, f1, cm, accuracy, X_labelled, y_test, positive_class):
        """
        Save metrices to results_folder/filename
        """
        full_path = os.path.join(self.results_folder, filename)
        results = []
        unique_classes = np.unique(y_test)

        if positive_class in unique_classes:
            positive_class_index = np.where(unique_classes == positive_class)[0][0]
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
            results_df.to_csv(full_path, mode='a', header=not os.path.exists(full_path), index=False)
            # print(f"Metrics saved for cycle {cycle} to {full_path}")

    def run_active_learning(self, X_train, y_train, X_test, y_test, dataset_name):
        """
        Active learning pipeline
        """
        X_labelled, X_unlabelled, y_labelled, y_unlabelled = train_test_split(
            X_train, y_train, test_size=0.9, stratify=y_train
        )

        X_labelled = X_labelled.to_numpy()
        X_unlabelled = X_unlabelled.to_numpy()

        remaining_budget = self.budget
        cycle = 0

        while remaining_budget > 0:
            self.model.fit(X_labelled, y_labelled)
            probabilities = self.model.predict_proba(X_unlabelled)

            X_top, y_top, X_rest, y_rest = self.select_samples(probabilities, X_unlabelled, self.cycle_budget, y_unlabelled)

            X_labelled = pd.concat([X_labelled, X_top])
            y_labelled = pd.concat([y_labelled, y_top])
            X_unlabelled = X_rest
            y_unlabelled = y_rest

            y_pred = self.model.predict(X_test)
            precision, recall, f1, cm, accuracy = self.calculate_metrics(y_test, y_pred)

            metrics_filename = f'{self.model_name}__{self.samples_selection_metric}__{dataset_name}__classic_AL.csv'
            self.save_metrics_to_file(metrics_filename, cycle, precision, recall, f1, cm, accuracy, X_labelled, y_test, positive_class=1)
            cycle += 1
            remaining_budget -= self.cycle_budget

        return metrics_filename

    def process_results(self, metrics_file_name, dataset, final_results):
        results_df = pd.read_csv(os.path.join('results', metrics_file_name))

        if not os.path.exists('Metrics_summaries'):
            os.makedirs('Metrics_summaries')
        
        last_iteration_number = max(results_df['Cycle'])
        grouped_df = results_df.groupby('Cycle').agg(['mean', 'std'])
        
        grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns]#.values]
        last_iter_stats = grouped_df.loc[last_iteration_number, :]
        stats_dict = last_iter_stats.to_dict()  # Convert Series to dictionary

        # # Append results to final summary
        final_results.append({
            "acquisition_function": self.samples_selection_metric,
            "dataset_name": dataset,
            "model_name": self.model_name,
            "mean_and_stats_for_last_iter": stats_dict,
            "cycle": last_iteration_number,
            "budget": self.budget
         })

if __name__ == "__main__":
    final_results = []
    #imbalanced_datasets_proper = ['abalone-3_vs_11']
    for dataset in imbalanced_datasets_proper:
        # print(f"Processing dataset: {dataset}")

        df = keel_ds.load_data(dataset, imbalanced=True, raw=True)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].str.strip().map({'negative': 0, 'positive': 1})

        skf = StratifiedKFold(n_splits=5)

        for train_index, test_index in skf.split(X, y):
            x_train_fold, x_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

            # Preporcess features for model
            x_train_fold = preprocess_features(x_train_fold)
            x_test_fold = preprocess_features(x_test_fold)
            #x_train_fold = x_train_fold.to_numpy()
            #x_test_fold = x_test_fold.to_numpy()


            pipeline = ActiveLearningPipeline(
                model=DecisionTreeClassifier(criterion='gini', max_depth=100),
                model_name='decision_tree',
                budget=50,
                cycle_budget=10,
                samples_selection_metric='entropy'
            )

            metrics_file_name = pipeline.run_active_learning(
                X_train=x_train_fold,
                y_train=y_train_fold,
                X_test=x_test_fold,
                y_test=y_test_fold,
                dataset_name=dataset
            )

        pipeline.process_results(metrics_file_name, dataset, final_results)

    final_results_df = pd.DataFrame(final_results)
    final_results_df.to_csv("final_results.csv", index=False)
