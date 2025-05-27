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

# === Setup ===
os.makedirs("logs", exist_ok=True)
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
logging.basicConfig(filename="logs/app.log", level=logging.INFO)

# === Load model & scaler ===
model = tf.keras.models.load_model("bidirectional_lstm_classweight.h5", compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

scaler = joblib.load("scaler_ids.pkl")
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
        error_msg = f"❌ Prediction error: {e}"
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
            test_path = "NSL-KDD/KDDTest+.txt"
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test file not found at {test_path}")

            df = pd.read_csv(test_path, header=None)
            df.columns = [
                "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
                "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
                "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                "dst_host_srv_rerror_rate", "label", "difficulty"
            ]

            # Ensure label column is present and clean
            df['label'] = df['label'].astype(str).str.strip().apply(lambda x: 0 if x == 'normal' else 1)

            # One-hot encode categorical variables
            df = pd.get_dummies(df, columns=["protocol_type", "flag"])

            # Ensure all required model input features exist
            for col in FEATURES:
                if col not in df.columns:
                    df[col] = 0.0
            df = df.reindex(columns=FEATURES + ['label'], fill_value=0)

            X = df[FEATURES]
            y = df['label']
            X_scaled = scaler.transform(X)
            X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, len(FEATURES)))
            preds_prob = model.predict(X_lstm, verbose=0).flatten()
            preds = (preds_prob >= THRESHOLD).astype(int)
            report = classification_report(y, preds, output_dict=True)
            false_positive = sum((preds == 1) & (y == 0))
            false_negative = sum((preds == 0) & (y == 1))

            precision = report['1']['precision'] if '1' in report else 0
            recall = report['1']['recall'] if '1' in report else 0

            metrics = {
                "accuracy": f"{report['accuracy']*100:.2f}%",
                "precision": f"{precision*100:.2f}%",
                "recall": f"{recall*100:.2f}%",
                "false_positive_rate": f"{100 * false_positive / max(sum(y==0), 1):.2f}%",
                "false_negative_rate": f"{100 * false_negative / max(sum(y==1), 1):.2f}%",
                "threshold": THRESHOLD,
                "features_used": FEATURES
            }

            with open(metrics_file, "w") as f:
                json.dump(metrics, f)
    except Exception as e:
        metrics = {"error": str(e)}

    return render_template("model_info.html", metrics=metrics)

@app.route('/metrics-chart')
def metrics_chart():
    if not os.path.exists("metrics_cache.json"):
        return "No cached metrics yet.", 404
    with open("metrics_cache.json", 'r') as f:
        metrics = json.load(f)
    labels = ["Accuracy", "Precision", "Recall"]
    values = [float(metrics["accuracy"].replace('%','')),
              float(metrics["precision"].replace('%','')),
              float(metrics["recall"].replace('%',''))]
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['green', 'blue', 'orange'])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percentage")
    ax.set_title("Model Metrics Overview")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/download-metrics')
def download_metrics():
    path = "metrics_cache.json"
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "Metrics file not found.", 404

if __name__ == '__main__':
    app.run(debug=True)
