import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib

# === Settings ===
np.random.seed(42)
tf.random.set_seed(42)

# === Load Data ===
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"] 
train_df = pd.read_csv("NSL-KDD/KDDTrain+.txt", header=None)
test_df = pd.read_csv("NSL-KDD/KDDTest+.txt", header=None)
train_df.columns = test_df.columns = columns

train_df['binary_label'] = train_df['label'].apply(lambda x: 0 if x == 'normal' else 1)
test_df['binary_label'] = test_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# One-hot encode categorical features
full_df = pd.concat([train_df, test_df])
full_df = pd.get_dummies(full_df, columns=['protocol_type', 'service', 'flag'], drop_first=False)

train_encoded = full_df.iloc[:len(train_df)]
test_encoded = full_df.iloc[len(train_df):]

# === 12 Optimized Features ===
features = [
    'dst_host_srv_serror_rate', 'serror_rate', 'srv_serror_rate', 'logged_in',
    'dst_host_same_srv_rate', 'protocol_type_tcp', 'protocol_type_udp',
    'count', 'src_bytes', 'dst_bytes', 'flag_SF', 'flag_REJ'
]

X_train = train_encoded[features]
y_train = train_encoded['binary_label']
X_test = test_encoded[features]
y_test = test_encoded['binary_label']

# === Scale Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_lstm = X_train_scaled.reshape(-1, 1, len(features))
X_test_lstm = X_test_scaled.reshape(-1, 1, len(features))

# === Compute Class Weights ===
class_weights_array = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights_array))

# === Focal Loss ===
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt + 1e-7))
    return loss

# === Bidirectional LSTM Model ===
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(1e-5, 1e-5)), input_shape=(1, len(features))),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(32, kernel_regularizer=l1_l2(1e-5, 1e-5))),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=focal_loss(),
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)

# === Callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, min_lr=1e-6)
]

# === Train the Model with Class Weights ===
history = model.fit(
    X_train_lstm, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# === Save Model and Scaler ===
model.save("bidirectional_lstm_classweight.h5")
joblib.dump(scaler, "scaler_ids.pkl")
print("✅ Model and scaler saved.")

# === Evaluate ===
y_pred_proba = model.predict(X_test_lstm).flatten()
threshold = 0.35
y_pred = (y_pred_proba >= threshold).astype(int)

print(f"\n✅ Confusion Matrix (Threshold = {threshold}):\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
print("✅ ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

# === Plot ROC ===
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_proba):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - KDDTest+")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_class_weight.png")
plt.close()
