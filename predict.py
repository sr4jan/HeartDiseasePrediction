# predict.py
import pandas as pd
import pickle

# Load the saved model and preprocessor
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Load new data (ensure new_data.csv has the same feature columns as heart.csv, excluding 'num' and 'id')
new_data = pd.read_csv('new_data.csv')

if 'id' in new_data.columns:
    new_data = new_data.drop('id', axis=1)

# Preprocess the new data using the saved preprocessor
new_X = preprocessor.transform(new_data)

# Predict with the trained model
predictions = model.predict(new_X)

print("🔮 Predictions for new data:")
print(predictions)
