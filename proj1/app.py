from flask import Flask, render_template, request
import pandas as pd
from models import train_and_evaluate
from privacy_methods import gaussian_noise, categorical_noise, k_anonymity

app = Flask(__name__)

# Load dataset
dataset = pd.read_csv("data/sample_data.csv")


# Strip spaces from column names
dataset.columns = dataset.columns.str.strip()

# Define columns
numerical_columns = ["AGE", "ZIPCODE", "SALARY", "BONUS", "LOAN"]
categorical_columns = ["GENDER", "EDUCATION", "JOB"]
quasi_identifiers = ["AGE", "GENDER", "ZIPCODE", "EDUCATION", "JOB"]
numerical_sensitive = ["SALARY", "BONUS", "LOAN"]
categorical_sensitive = ["STRESS_LEVELS"]

@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predictions", methods=["GET", "POST"])
def predictions():
    if request.method == "POST":
        user_data = {
            "AGE": int(request.form["AGE"]),
            "ZIPCODE": int(request.form["ZIPCODE"]),
            "SALARY": float(request.form["SALARY"]),
            "BONUS": float(request.form["BONUS"]),
            "LOAN": float(request.form["LOAN"]),
            "GENDER": request.form["GENDER"],
            "EDUCATION": request.form["EDUCATION"],
            "JOB": request.form["JOB"],
        }

        user_df = pd.DataFrame([user_data])

        # Standardize column names
        user_df.columns = user_df.columns.str.strip().str.replace(" ", "_")

        # Convert categorical columns to match trained model format
        user_df[categorical_columns] = user_df[categorical_columns].astype(str)

        # Get model predictions for user input
        user_prediction = train_and_evaluate(dataset, numerical_columns, categorical_columns, user_input=user_df)

        return render_template("predictions.html", predictions=user_prediction)

    return render_template("predictions.html")

@app.route("/analysing")
def analysing():
    try:
        # Load dataset
        dataset = pd.read_csv("data/sample_data.csv")
        # Load dataset with column name correction
        #print("🔍 Columns in user_df before passing to train_and_evaluate:", list(dataset.columns))
        dataset.columns = dataset.columns.str.strip()  # Remove spaces from column names

        # Apply privacy techniques
        noisy_data_clipped = gaussian_noise(dataset, numerical_columns, 0, 3000, 0.3, {})
        noisy_data_categorical = categorical_noise(dataset, ["EDUCATION", "JOB"], 0.2)
        anonymized_data = k_anonymity(dataset, 3, quasi_identifiers, numerical_sensitive, categorical_sensitive)
    
        # Compute accuracy scores
        original_utility = train_and_evaluate(dataset, numerical_columns, categorical_columns)
        gaussian_noise_utility = train_and_evaluate(noisy_data_clipped, numerical_columns, categorical_columns)
        categorical_Noise_utility = train_and_evaluate(noisy_data_categorical, numerical_columns, categorical_columns)
        K_Anonymity_utility = train_and_evaluate(anonymized_data, numerical_columns, categorical_columns)

        # Pass results to template
        return render_template(
            "analysing.html",
            original=original_utility,
            gaussian=gaussian_noise_utility,
            categorical=categorical_Noise_utility,
            anonymity=K_Anonymity_utility
        )

    except Exception as e:
        return f"Error in analysing(): {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)
