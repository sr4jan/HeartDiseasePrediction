# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the saved model and preprocessor
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get form inputs
            age = float(request.form.get('age'))
            sex = request.form.get('sex')  # e.g., "Male" or "Female"
            dataset = request.form.get('dataset')  # e.g., "Cleveland"
            cp = request.form.get('cp')  # chest pain type
            trestbps = float(request.form.get('trestbps'))
            chol = float(request.form.get('chol'))
            fbs = request.form.get('fbs')
            restecg = request.form.get('restecg')
            thalch = float(request.form.get('thalch'))
            exang = request.form.get('exang')
            oldpeak = float(request.form.get('oldpeak'))
            slope = request.form.get('slope')
            ca = float(request.form.get('ca'))
            thal = request.form.get('thal')

            # Create a DataFrame for the single input row
            input_data = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                'dataset': [dataset],
                'cp': [cp],
                'trestbps': [trestbps],
                'chol': [chol],
                'fbs': [fbs],
                'restecg': [restecg],
                'thalch': [thalch],
                'exang': [exang],
                'oldpeak': [oldpeak],
                'slope': [slope],
                'ca': [ca],
                'thal': [thal]
            })

            # Preprocess the input (using the preprocessor saved from training)
            processed_data = preprocessor.transform(input_data)
            pred = model.predict(processed_data)
            prediction = 'Heart Disease Detected' if pred[0] == 1 else 'No Heart Disease'
        except Exception as e:
            prediction = f"Error in prediction: {str(e)}"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
