import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Function 1: Gaussian Noise Addition with Clipping
def gaussian_noise(df, numerical_columns, mean, std_dev, sampling_percentage, clip_ranges):
    df_with_noise = df.copy()
    num_samples = int(len(df) * sampling_percentage)
    selected_indices = np.random.choice(df.index, num_samples, replace=False)

    for col in numerical_columns:
        noise = np.random.normal(mean, std_dev, len(selected_indices))
        min_clip, max_clip = clip_ranges.get(col, (-20, 20))
        clipped_noise = np.clip(noise, min_clip, max_clip)
        df_with_noise.loc[selected_indices, col] = (
            df_with_noise.loc[selected_indices, col] + clipped_noise
        ).astype(df_with_noise[col].dtype)

    return df_with_noise

# Function 2: Categorical Noise Addition
def categorical_noise(df, categorical_columns, sampling_percentage):
    df_with_noise = df.copy()
    num_samples = int(len(df) * sampling_percentage)
    selected_indices = np.random.choice(df.index, num_samples, replace=False)

    for col in categorical_columns:
        for idx in selected_indices:
            unique_values = df[col].unique()
            new_value = np.random.choice(unique_values)
            df_with_noise.loc[idx, col] = new_value

    return df_with_noise

# Function 3: K-Anonymity-based Generalization using K-Means
def k_anonymity(df, k, quasi_identifiers, numerical_sensitive, categorical_sensitive):
    T_Q = df[quasi_identifiers]
    T_N = df[numerical_sensitive]
    T_C = df[categorical_sensitive]

    def k_means_clustering(df, k):
        model = KMeans(n_clusters=max(len(df) // k, 1), random_state=42)
        labels = model.fit_predict(df)
        df = df.copy()
        df["Cluster"] = labels
        return df, model

    def encode_categorical_columns(df):
        label_encoders = {}
        for column in df.select_dtypes(include=["object", "category"]).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
        return df, label_encoders

    def decode_categorical_columns(df, encoders):
        for column, encoder in encoders.items():
            df[column] = encoder.inverse_transform(df[column].astype(int))
        return df

    T_Q_encoded, Q_encoders = encode_categorical_columns(T_Q)
    T_Q_clustered, _ = k_means_clustering(T_Q_encoded, k)
    T_Q_generalized = T_Q_clustered.groupby("Cluster").transform("mean")

    T_N_clustered, _ = k_means_clustering(T_N, k)
    T_N_generalized = T_N_clustered.groupby("Cluster").transform("mean")

    T_C_encoded, C_encoders = encode_categorical_columns(T_C)
    T_C_clustered, _ = k_means_clustering(T_C_encoded, k)
    T_C_generalized = T_C_clustered.groupby("Cluster").transform(lambda x: x.mode()[0])
    T_C_final = decode_categorical_columns(T_C_generalized, C_encoders)

    anonymized_data = pd.concat([T_Q_generalized, T_N_generalized, T_C_final], axis=1)
    return anonymized_data
