# train_ml_models.py
# Smart Study Optimizer - ML Model Training (Focus + Break)

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
import joblib

# ===============================================================
# 1. Generate Synthetic Study Behavior Dataset
# ===============================================================

print("📂 Generating synthetic dataset...")

np.random.seed(42)
N = 20000  # Large enough for good training

data = pd.DataFrame({
    "keystrokes": np.random.poisson(lam=50, size=N),
    "mouse_moves": np.random.poisson(lam=60, size=N),
    "idle_time": np.random.uniform(0, 300, N),  # seconds
    "session_duration": np.random.uniform(60, 3600, N),  # seconds (1 min - 1 hour)
    "time_of_day": np.random.choice(["morning", "afternoon", "evening", "night"], size=N)
})

# Encode categorical feature
data["time_of_day_encoded"] = data["time_of_day"].map({
    "morning": 0.9,
    "afternoon": 0.7,
    "evening": 0.5,
    "night": 0.3
})

# ===============================================================
# 2. Simulate Target Variables
# ===============================================================

# Focus Score (0–1)
# - Higher when keystrokes/mouse are active, idle time is low
data["focus_score"] = (
    0.4 * np.tanh((data["keystrokes"] + data["mouse_moves"]) / 100)
    + 0.4 * (1 - np.tanh(data["idle_time"] / 200))
    + 0.2 * data["time_of_day_encoded"]
)
data["focus_score"] = np.clip(data["focus_score"], 0, 1)

# Break Needed (1 = needs break, 0 = continue)
data["needs_break"] = np.where(
    (data["focus_score"] < 0.5) | (data["session_duration"] > 2400), 1, 0
)

# ===============================================================
# 3. Split Data for Training
# ===============================================================

features = ["keystrokes", "mouse_moves", "idle_time", "session_duration", "time_of_day_encoded"]
X = data[features]
y_focus = data["focus_score"]
y_break = data["needs_break"]

X_train, X_test, y_focus_train, y_focus_test = train_test_split(X, y_focus, test_size=0.2, random_state=42)
_, _, y_break_train, y_break_test = train_test_split(X, y_break, test_size=0.2, random_state=42)

# ===============================================================
# 4. Scale Features
# ===============================================================

print("⚙️ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===============================================================
# 5. Train Focus Predictor (Regression)
# ===============================================================

print("\n🎯 Training Focus Predictor (Regression)...")
focus_model = LinearRegression()
focus_model.fit(X_train_scaled, y_focus_train)

y_focus_pred = focus_model.predict(X_test_scaled)
r2 = r2_score(y_focus_test, y_focus_pred)
mse = mean_squared_error(y_focus_test, y_focus_pred)
print(f"✅ Focus Model R²: {r2:.3f}, MSE: {mse:.4f}")

# ===============================================================
# 6. Train Break Predictor (Classification)
# ===============================================================

print("\n🧩 Training Break Predictor (Classification)...")
break_model = LogisticRegression(max_iter=1000)
break_model.fit(X_train_scaled, y_break_train)

y_break_pred = break_model.predict(X_test_scaled)
acc = accuracy_score(y_break_test, y_break_pred)
print(f"✅ Break Model Accuracy: {acc:.3f}\n")

print("Detailed Classification Report:")
print(classification_report(y_break_test, y_break_pred))

# ===============================================================
# 7. Save Models
# ===============================================================

SAVE_DIR = os.path.join("data", "models")
os.makedirs(SAVE_DIR, exist_ok=True)

joblib.dump(focus_model, os.path.join(SAVE_DIR, "focus_predictor.joblib"))
joblib.dump(break_model, os.path.join(SAVE_DIR, "break_predictor.joblib"))
joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.joblib"))

print(f"\n💾 Models saved in: {SAVE_DIR}")
print("✅ Training complete!\n")

# ===============================================================
# 8. (Optional) Save Dataset Snapshot
# ===============================================================

data.to_csv("synthetic_study_data.csv", index=False)
print("📊 Synthetic dataset saved as 'synthetic_study_data.csv'.")
