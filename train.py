import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("=== TRAINING GESTURE MODEL ===")

df = pd.read_csv('data/gesture_data.csv', header=None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

print(f"Total samples: {len(y)}")
print(f"Gestures found: {np.unique(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy * 100:.1f}%")
print("\nDetailed report:")
print(classification_report(y_test, y_pred))

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/gesture_model.pkl')
print("\nModel saved to model/gesture_model.pkl")
print("Training complete!")