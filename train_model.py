import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Dataset file name
DATASET_PATH = "ads_dataset.csv"


def train_models():
    """
    Train all machine learning models:
    1. TF-IDF Vectorizer
    2. K-Means Clustering
    3. Regression model for risk score
    4. Neural Network classifier
    """

    print("Loading dataset...")

    # Load dataset
    data = pd.read_csv(DATASET_PATH)

    # Check required columns
    required_columns = ["ad_text", "label", "risk_score"]
    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Missing required column: {column}")

    # Remove empty values
    data = data.dropna()

    # Input and output columns
    X_text = data["ad_text"]
    y_label = data["label"]
    y_risk = data["risk_score"]

    print("Converting text into TF-IDF features...")

    # Convert advertisement text into numerical features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=1000
    )

    X_features = vectorizer.fit_transform(X_text)

    print("Training clustering model...")

    # K-Means clustering model
    clustering_model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    clustering_model.fit(X_features)

    print("Training regression model...")

    # Regression model for risk score prediction
    regression_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    regression_model.fit(X_features, y_risk)

    print("Training neural network model...")

    # Convert Fake/Genuine text labels into numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_label)

    # Split data for checking accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    # Neural Network classifier
    neural_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=800,
        random_state=42
    )

    neural_model.fit(X_train, y_train)

    # Test model accuracy
    y_pred = neural_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Neural Network Model Accuracy: {accuracy * 100:.2f}%")

    print("Saving models...")

    # Save all trained models
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(clustering_model, "clustering_model.pkl")
    joblib.dump(regression_model, "regression_model.pkl")
    joblib.dump(neural_model, "neural_model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")

    print("All models trained and saved successfully!")
    print("Saved files:")
    print("- vectorizer.pkl")
    print("- clustering_model.pkl")
    print("- regression_model.pkl")
    print("- neural_model.pkl")
    print("- label_encoder.pkl")


if __name__ == "__main__":
    train_models()