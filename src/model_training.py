import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def generate_synthetic_dataset(samples=300):
    """
    Create dummy EEG dataset
    0 = Healthy
    1 = Schizophrenia
    """

    np.random.seed(42)

    healthy = np.random.normal(0.5, 0.1, (samples // 2, 5))
    schizophrenia = np.random.normal(0.8, 0.15, (samples // 2, 5))

    X = np.vstack((healthy, schizophrenia))
    y = np.array([0] * (samples // 2) + [1] * (samples // 2))

    return X, y


def train_and_save_model():

    X, y = generate_synthetic_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    os.makedirs("models", exist_ok=True)

    with open("models/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved successfully inside models/")
    

if __name__ == "__main__":
    train_and_save_model()