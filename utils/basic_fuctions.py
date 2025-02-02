import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_data(folder_name, file_name, header=None):
    """Load csv file in df"""
    df = pd.read_csv(os.path.join(os.getcwd(), folder_name, file_name), header)

    return df

def preprocess_features(X_df):
    X_df_copy = X_df.copy()
    for col in X_df_copy.columns:
        if X_df_copy.loc[:, col].dtype == 'object':  
            le = LabelEncoder()
            X_df_copy.loc[:, col] = le.fit_transform(X_df_copy[col])

    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_df_copy)

    X_normalized = pd.DataFrame(X_normalized, columns=X_df_copy.columns)

    return X_normalized
