from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import joblib
import os
import numpy as np

app = Flask(__name__)
app.secret_key = "fake_ad_detection_secret_key"

# Fixed login credentials for mini project
USERNAME = "admin"
PASSWORD = "admin123"

# Model file paths
VECTORIZER_PATH = "vectorizer.pkl"
CLUSTERING_MODEL_PATH = "clustering_model.pkl"
REGRESSION_MODEL_PATH = "regression_model.pkl"
NEURAL_MODEL_PATH = "neural_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"


def load_models():
    vectorizer = joblib.load(VECTORIZER_PATH)
    clustering_model = joblib.load(CLUSTERING_MODEL_PATH)
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
    neural_model = joblib.load(NEURAL_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    return vectorizer, clustering_model, regression_model, neural_model, label_encoder


def get_risk_level(risk_score):
    if risk_score >= 80:
        return "High Risk"
    elif risk_score >= 50:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_risk_explanation(risk_score):
    if risk_score >= 80:
        return "This advertisement contains highly suspicious patterns such as unrealistic offers, urgency words, or misleading claims."
    elif risk_score >= 50:
        return "This advertisement contains some suspicious patterns and should be verified carefully before trusting it."
    else:
        return "This advertisement has fewer suspicious patterns and appears safer compared to high-risk advertisements."


def get_cluster_details(cluster_number):
    cluster_info = {
        0: {
            "name": "Online Job or Money Offer Advertisement",
            "explanation": "This group contains advertisements related to job offers, work-from-home income, fast money, or earning schemes."
        },
        1: {
            "name": "Shopping or Discount Advertisement",
            "explanation": "This group contains advertisements related to product sales, discounts, offers, coupons, and online shopping."
        },
        2: {
            "name": "Loan or Finance Advertisement",
            "explanation": "This group contains advertisements related to instant loans, finance offers, money transfer, investment, or banking claims."
        },
        3: {
            "name": "Prize or Gift Advertisement",
            "explanation": "This group contains advertisements related to lottery prizes, free gifts, lucky draws, rewards, or claim offers."
        }
    }

    return cluster_info.get(cluster_number, {
        "name": f"Advertisement Group {cluster_number}",
        "explanation": "This advertisement belongs to a text-based group identified by the clustering model."
    })


def get_recommendation(prediction, risk_score):
    if prediction == "Fake" and risk_score >= 80:
        return "High risk advertisement. Do not click any links, do not make payments, and do not share personal or banking details."
    elif prediction == "Fake":
        return "This advertisement looks suspicious. Verify the source carefully and avoid sharing personal information."
    elif prediction == "Genuine" and risk_score <= 40:
        return "This advertisement appears genuine. Still, verify seller details before making any purchase."
    else:
        return "This advertisement seems mostly safe, but please check the source and terms before proceeding."


def get_user_action(prediction, risk_score):
    if prediction == "Fake" or risk_score >= 80:
        return "Avoid and Report"
    elif risk_score >= 50:
        return "Verify Before Trusting"
    else:
        return "Safe to Review"


@app.route("/signin", methods=["GET", "POST"])
def signin():
    error = None

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username == USERNAME and password == PASSWORD:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            error = "Invalid username or password"

    return render_template("signin.html", error=error)


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("signin"))


@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("signin"))

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({
            "error": "Please sign in first."
        }), 401

    try:
        data = request.get_json()
        ad_text = data.get("ad_text", "").strip()

        if not ad_text:
            return jsonify({
                "error": "Please enter advertisement text."
            }), 400

        required_files = [
            VECTORIZER_PATH,
            CLUSTERING_MODEL_PATH,
            REGRESSION_MODEL_PATH,
            NEURAL_MODEL_PATH,
            LABEL_ENCODER_PATH
        ]

        for file in required_files:
            if not os.path.exists(file):
                return jsonify({
                    "error": "Model files not found. Please run python train_model.py first."
                }), 500

        vectorizer, clustering_model, regression_model, neural_model, label_encoder = load_models()

        text_features = vectorizer.transform([ad_text])

        # Neural Network Classification
        prediction_encoded = neural_model.predict(text_features)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        if hasattr(neural_model, "predict_proba"):
            probabilities = neural_model.predict_proba(text_features)[0]
            confidence = round(float(np.max(probabilities)) * 100, 2)
        else:
            confidence = 85.0

        neural_analysis = {
            "component": "Neural Network Classification",
            "model": "MLPClassifier",
            "prediction": prediction,
            "confidence": confidence,
            "explanation": "The neural network analyzes TF-IDF text features and classifies the advertisement as Fake or Genuine."
        }

        # Regression Risk Analysis
        risk_score = regression_model.predict(text_features)[0]
        risk_score = round(float(risk_score), 2)
        risk_score = max(0, min(100, risk_score))

        risk_level = get_risk_level(risk_score)
        risk_explanation = get_risk_explanation(risk_score)

        regression_analysis = {
            "component": "Regression Risk Analysis",
            "model": "Random Forest Regressor",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "explanation": risk_explanation
        }

        # Clustering Analysis
        cluster_number = int(clustering_model.predict(text_features)[0])
        cluster_details = get_cluster_details(cluster_number)

        clustering_analysis = {
            "component": "Clustering Analysis",
            "model": "K-Means Clustering",
            "cluster_number": cluster_number,
            "cluster_name": cluster_details["name"],
            "explanation": cluster_details["explanation"]
        }

        # Recommendation System
        recommendation = get_recommendation(prediction, risk_score)
        user_action = get_user_action(prediction, risk_score)

        recommendation_analysis = {
            "component": "Recommendation System",
            "recommendation": recommendation,
            "user_action": user_action,
            "explanation": "The recommendation system gives safety suggestions based on prediction result and risk score."
        }

        return jsonify({
            "final_prediction": prediction,
            "confidence": confidence,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "cluster": cluster_details["name"],
            "recommendation": recommendation,
            "user_action": user_action,

            "neural_analysis": neural_analysis,
            "regression_analysis": regression_analysis,
            "clustering_analysis": clustering_analysis,
            "recommendation_analysis": recommendation_analysis,

            "prediction": prediction,
            "cluster_number": cluster_number,
            "cluster_name": cluster_details["name"],
            "cluster_explanation": cluster_details["explanation"]
        })

    except Exception as e:
        return jsonify({
            "error": f"Something went wrong: {str(e)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)