# 🚨 LSTM-Based Network Intrusion Detection System (IDS)

A sophisticated **Bidirectional LSTM**-based Intrusion Detection System that accurately identifies network attacks using advanced deep learning and optimized feature selection. Designed for real-time web-based deployment with interactive visualization.

---

## 📌 Overview

This project leverages a deep learning model trained on the NSL-KDD dataset to detect anomalies in network traffic. It includes:
- 🧠 A Bidirectional LSTM-based neural network
- 🧪 Custom Focal Loss for class imbalance
- 📉 ROC-AUC of **0.92** — excellent discrimination power
- 🌐 A Flask-based web app for interactive input and risk prediction

---

## 🔧 Technical Highlights

- **🔁 Bidirectional LSTM Architecture**  
  Captures both past and future context for better anomaly detection.

- **🎯 Focal Loss**  
  Handles class imbalance by focusing more on hard-to-classify samples.

- **⚖️ Class Weighting Strategy**  
  Balanced class weights to improve performance on minority attack classes.

- **🧹 Feature Optimization**  
  Only 12 high-impact features used for dimensionality reduction.

- **🛡️ Regularization**  
  Uses L1-L2 penalties, batch normalization, and dropout layers to combat overfitting.

- **🎚️ Threshold Tuning**  
  Custom threshold (0.35) balances false positives and recall for practical use.

---

## 📊 Performance Metrics

| Metric                | Value     |
|-----------------------|-----------|
| **Precision**         | 96.47%    |
| **Accuracy**          | 81.53%    |
| **Recall**            | 70.12%    |
| **False Positive Rate** | 3.39%     |
| **ROC-AUC Score**     | 0.92      |

---

## 🧠 Model Architecture

Bidirectional LSTM (64 units, L1-L2 regularization) → BatchNorm → Dropout(0.3)



↓
Bidirectional LSTM (32 units, L1-L2 regularization) → BatchNorm → Dropout(0.3)



↓
Dense (16 units, ReLU) → Dropout(0.2)



↓
Dense (1 unit, Sigmoid)

---

## 🧬 Features Used

```python```
features = [
    'dst_host_srv_serror_rate', 'serror_rate', 'srv_serror_rate', 'logged_in',
    'dst_host_same_srv_rate', 'protocol_type_tcp', 'protocol_type_udp',
    'count', 'src_bytes', 'dst_bytes', 'flag_SF', 'flag_REJ'
]

---

## 🌐 Web Application
-✔️ Flask-based responsive interface


-✔️ Real-time risk classification (Low/Medium/High)


-✔️ History tracking and visualizations



-✔️ Intuitive layout and input system



---

## 🚀 Getting Started




**1️⃣ Clone the Repository**
-```git clone https://github.com/yourusername/lstm-network-ids.git```



-```cd lstm-network-ids```



**2️⃣ Install Dependencies**
```pip install -r requirements.txt```

Or install manually:

```pip install tensorflow flask numpy pandas scikit-learn matplotlib joblib```




**3️⃣  Download NSL-KDD Dataset for Training**
-```mkdir -p NSL-KDD```



-```wget -O NSL-KDD/KDDTrain+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt```



-```wget -O NSL-KDD/KDDTest+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt```




**4️⃣ Train the Model (Skip if using pretrained)**
-```python lstm_test.py```




**5️⃣ Launch the Web Application**
-```python app.py```

*Then open http://localhost:5000 in your browser.*

---

## 🏗️ Training Strategy
- **Data normalization + one-hot encoding**

- **Focal loss to focus learning on rare/complex cases**

- **Class weights for imbalance correction**

- **Adam optimizer with learning rate decay**

- **Early stopping to avoid overfitting**

---

## 🔮 Future Improvements

🌐 Real-time packet sniffing and live traffic monitoring

🧩 Multi-class classification (attack-type level)

🧪 Adversarial training for robustness

🔗 Integration with SIEM and firewall systems

🧠 Model ensembling with CNNs or Transformers

---

## 🙌 Credits
**Dataset: NSL-KDD by Canadian Institute for Cybersecurity**

**Model: Custom BiLSTM with TensorFlow 2.x**

**Author: Vaishnavi Raj**




