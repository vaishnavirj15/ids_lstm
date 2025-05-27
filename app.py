from flask import Flask, render_template, request, session, jsonify, redirect, url_for, send_file
import os
import logging
import numpy as np
import pandas as pd
import joblib
import json
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import classification_report
from werkzeug.middleware.proxy_fix import ProxyFix

# === Setup ===
os.makedirs("logs", exist_ok=True)
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.wsgi_app = ProxyFix(app.wsgi_app)
logging.basicConfig(filename="logs/app.log", level=logging.INFO)

# === Load model & scaler safely ===
try:
    model = tf.keras.models.load_model("bidirectional_lstm_classweight.h5", compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
except Exception as e:
    logging.error(f"❌ Model load failed: {e}")
    model = None

try:
    scaler = joblib.load("scaler_ids.pkl")
except Exception as e:
    logging.error(f"❌ Scaler load failed: {e}")
    scaler = None

THRESHOLD = 0.35
FEATURES = [
    'dst_host_srv_serror_rate', 'serror_rate', 'srv_serror_rate', 'logged_in',
    'dst_host_same_srv_rate', 'protocol_type_tcp', 'protocol_type_udp', 'count',
    'src_bytes', 'dst_bytes', 'flag_SF', 'flag_REJ'
]

@app.route('/')
def index():
    return render_template("index.html", features=FEATURES, history=session.get("history", []))

@app.route('/predict', methods=["POST"])
def predict():
    if not model or not scaler:
        error_msg = "❌ Model or scaler not loaded."
        return render_template("index.html", features=FEATURES, error=error_msg)

    try:
        numeric_features = [
            'duration', 'src_bytes', 'dst_bytes', 'count',
            'serror_rate', 'srv_serror_rate', 'dst_host_same_srv_rate',
            'dst_host_srv_serror_rate', 'logged_in'
        ]
        categorical_features = {
            'protocol_type': ['tcp', 'udp', 'icmp'],
            'flag': ['SF', 'REJ', 'S0', 'S1']
        }

        input_vector = dict.fromkeys(FEATURES, 0.0)

        for feature in numeric_features:
            val = request.form.get(feature, 0)
            input_vector[feature] = float(val)

        protocol = request.form.get('protocol_type', 'tcp')
        for proto_val in categorical_features['protocol_type']:
            key = f"protocol_type_{proto_val}"
            input_vector[key] = 1.0 if proto_val == protocol else 0.0

        flag_val = request.form.get('flag', 'SF')
        for fval in categorical_features['flag']:
            key = f"flag_{fval}"
            input_vector[key] = 1.0 if fval == flag_val else 0.0

        values = [input_vector[f] for f in FEATURES]
        X_input = np.array(values).reshape(1, -1)
        X_scaled = scaler.transform(X_input)
        X_lstm = X_scaled.reshape((1, 1, len(FEATURES)))

        pred_prob = model.predict(X_lstm, verbose=0)[0][0]
        result = "Attack" if pred_prob >= THRESHOLD else "Normal"
        confidence = round(pred_prob * 100, 2) if result == "Attack" else round((1 - pred_prob) * 100, 2)
        risk = "High" if pred_prob > 0.8 else "Medium" if pred_prob > 0.5 else "Low"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logging.info(f"{timestamp} | {result} | Conf: {confidence}% | Inputs: {values}")

        entry = {"timestamp": timestamp, "result": result, "confidence": confidence, "risk": risk}
        history = session.get("history", [])
        history.insert(0, entry)
        session["history"] = history[:10]

        return render_template("index.html",
                               features=FEATURES,
                               result=result,
                               confidence=confidence,
                               risk=risk,
                               history=session["history"])
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        error_msg = f"❌ Prediction error: {str(e)}"
        return render_template("index.html",
                               features=FEATURES,
                               history=session.get("history", []),
                               error=error_msg)

@app.route('/clear')
def clear():
    session.clear()
    return redirect(url_for("index"))

@app.route('/model-info')
def model_info():
    metrics_file = "metrics_cache.json"
    try:
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        else:
            raise FileNotFoundError("⚠️ Cached metrics not available. Please generate them locally.")

    except Exception as e:
        metrics = {"error": str(e)}

    return render_template("model_info.html", metrics=metrics)

@app.route('/metrics-chart')
def metrics_chart():
    try:
        with open("metrics_cache.json", 'r') as f:
            metrics = json.load(f)
        labels = ["Accuracy", "Precision", "Recall"]
        values = [float(metrics[k].replace('%','')) for k in ["accuracy", "precision", "recall"]]
        fig, ax = plt.subplots()
        ax.bar(labels, values, color=['green', 'blue', 'orange'])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.set_title("Model Metrics Overview")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download-metrics')
def download_metrics():
    path = "metrics_cache.json"
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "Metrics file not found.", 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
