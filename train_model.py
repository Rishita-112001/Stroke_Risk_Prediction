import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.DataFrame({
    'gender': [1, 0, 1],
    'age': [67, 45, 34],
    'hypertension': [0, 1, 0],
    'heart_disease': [1, 0, 0],
    'ever_married': [1, 1, 0],
    'work_type': [0, 1, 2],
    'Residence_type': [1, 0, 1],
    'avg_glucose_level': [228.69, 202.21, 105.92],
    'bmi': [36.6, 27.3, 30.5],
    'smoking_status': [0, 1, 2],
})
labels = [1, 0, 0]

model = RandomForestClassifier()
model.fit(data, labels)

with open("model/stroke_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved to model/stroke_model.pkl")
