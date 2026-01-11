import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("Loading dataset...")
df = pd.read_csv('master_dataset.csv')

# INPUTS: What the robot knows (Command + Wheel Speed)
X = df[['v_cmd', 'v_wheel']]

y = df['slip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the Brain (this may take a moment)...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.5f} (Lower is better)")
print(f"R2 Score: {r2:.5f} (1.0 is perfect, 0.0 is terrible)")
print("-" * 30)

joblib.dump(model, 'slip_detector_model.pkl')
print("Model saved as 'slip_detector_model.pkl'")
