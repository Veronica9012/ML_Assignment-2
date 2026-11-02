import os

import joblib

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



# 1. Download Wine Quality Dataset (UCI)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(url, sep=";")



# Binary target conversion: 1 (Quality >= 6), 0 (Quality < 6)

df["target"] = (df["quality"] >= 6).astype(int)

df = df.drop(columns=["quality"])



X = df.drop(columns=["target"])

y = df["target"]



# 2. Train-Test Split (80-20 ratio)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y

)



# 3. Save test dataset CSV

test_df = X_test.copy()

test_df["target"] = y_test

test_df.to_csv("test_data.csv", index=False)

print("Successfully generated test_data.csv")



# 4. Feature Scaling & Artifact Saving

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)



os.makedirs("models", exist_ok=True)

joblib.dump(scaler, "models/scaler.pkl")



# 5. Train and save all 5 models

models = {

    "logistic_regression": LogisticRegression(random_state=42),

    "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=5),

    "knn": KNeighborsClassifier(n_neighbors=5),

    "naive_bayes": GaussianNB(),

    "support_vector_machine": SVC(probability=True, random_state=42)

}



for name, model in models.items():

    if name in ["logistic_regression", "knn", "support_vector_machine"]:

        model.fit(X_train_scaled, y_train)

    else:

        model.fit(X_train, y_train)

    joblib.dump(model, f"models/{name}.pkl")



print("All 5 models trained and saved in models/ folder.")

