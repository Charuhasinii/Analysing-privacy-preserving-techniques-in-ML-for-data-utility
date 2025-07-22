import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Global variables for trained models and encoders
trained_models = {}
encoder = None
scaler = None
categorical_feature_names = []

def train_and_evaluate(data, numerical_columns, categorical_columns, user_input=None):
    global encoder, scaler, categorical_feature_names


    # Standardize column names
    data.columns = data.columns.str.strip().str.replace(" ", "_")
    print("✅ After Cleaning - Dataset Columns:", list(data.columns))

    # If STRESS_LEVELS is missing, assume it's a user input and predict using trained models
    if user_input is not None:
        if not trained_models:
            raise ValueError(" Models not trained yet! Train them first by running /analysing")


        # Standardize column names in user input
        user_input.columns = user_input.columns.str.strip().str.replace(" ", "_")

        # Encode categorical data
        user_input[categorical_columns] = user_input[categorical_columns].astype(str)
        encoded_categorical = encoder.transform(user_input[categorical_columns])
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical,
            columns=categorical_feature_names,
            index=user_input.index
        )

        # Scale numerical data
        user_numerical = scaler.transform(user_input[numerical_columns])
        user_numerical_df = pd.DataFrame(user_numerical, columns=numerical_columns, index=user_input.index)

        # Merge processed categorical and numerical data
        user_processed = pd.concat([user_numerical_df, encoded_categorical_df], axis=1)

        # Make predictions
        lr_prediction = trained_models["lr"].predict(user_processed)[0]
        rf_prediction = trained_models["rf"].predict(user_processed)[0]
        knn_prediction = trained_models["knn"].predict(user_processed)[0]

        return {
            "Logistic Regression Prediction": lr_prediction,
            "Random Forest Prediction": rf_prediction,
            "k-NN Prediction": knn_prediction
        }

    # Otherwise, train models normally
    X = data.drop(columns=["STRESS_LEVELS"])
    y = data["STRESS_LEVELS"]

    # Encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    data[categorical_columns] = data[categorical_columns].astype(str)
    encoded_categorical = encoder.fit_transform(data[categorical_columns])
    categorical_feature_names = encoder.get_feature_names_out(categorical_columns)
    encoded_categorical_df = pd.DataFrame(
        encoded_categorical,
        columns=categorical_feature_names,
        index=data.index
    )

    # Standardize numerical columns
    scaler = StandardScaler()
    X_numerical = scaler.fit_transform(data[numerical_columns])
    X_numerical_df = pd.DataFrame(X_numerical, columns=numerical_columns, index=data.index)

    # Merge processed categorical and numerical data
    X_processed = pd.concat([X_numerical_df, encoded_categorical_df], axis=1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Train models
    lr = LogisticRegression(max_iter=1000, C=0.5)
    rf = RandomForestClassifier(max_depth=10, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    # Store trained models for user predictions
    trained_models["lr"] = lr
    trained_models["rf"] = rf
    trained_models["knn"] = knn

    # Compute accuracy
    return {
        "Logistic Regression Accuracy": accuracy_score(y_test, lr.predict(X_test)),
        "Random Forest Accuracy": accuracy_score(y_test, rf.predict(X_test)),
        "k-NN Accuracy": accuracy_score(y_test, knn.predict(X_test))
    }
