"""Pipeline ML pour la prédiction du churn."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(csv_path="Churn_Modelling.csv", test_size=0.2, random_state=1):
    """Charge le CSV, prétraite les données et sépare train/test."""
    df = pd.read_csv(csv_path)

    # Encodage de Gender : Female/Male -> 0/1
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])

    # Suppression des colonnes inutiles
    df = df.drop(columns=["Surname", "Geography", "RowNumber", "CustomerId"])

    x = df.drop(columns=["Exited"])  # variables explicatives
    y = df["Exited"]  # cible à prédire

    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier et retourne le modèle."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Évalue le modèle : accuracy, matrice de confusion, rapport."""
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy : {accuracy * 100:.2f} %")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return accuracy


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle avec joblib."""
    joblib.dump(model, path)
    print(f"Modèle sauvegardé dans {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé."""
    model = joblib.load(path)
    print(f"Modèle chargé depuis {path}")
    return model
