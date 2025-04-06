# Heart Disease Prediction

This is a machine learning project that predicts whether a person is likely to have heart disease based on various medical attributes. The model is trained using scikit-learn and deployed through a simple Flask web application.

## Features

- Train a machine learning model to predict heart disease
- Web interface built with HTML, CSS, and Flask
- Take user input (age, sex, smoker status, blood pressure, etc.)
- Display prediction results

## Project Structure

```
HeartDiseasePrediction/
│
├── app.py                 # Flask backend
├── predict.py             # Model testing script
├── model.pkl              # Saved ML model
├── preprocessor.pkl       # Saved preprocessor
├── templates/
│   └── index.html         # Frontend page
├── static/
│   └── styles.css         # Stylesheet (optional)
├── dataset.csv            # Sample dataset
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .gitignore             # Files to ignore in Git
```

## How to Run Locally

### Clone the Repository

```bash
git clone https://github.com/sr4jan/HeartDiseasePrediction.git
cd HeartDiseasePrediction
```

### Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate    # For Mac/Linux
venv\Scripts\activate       # For Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

### Open in Browser

Visit http://127.0.0.1:5000/ to use the application.

## Technologies Used

- Python
- Flask
- scikit-learn
- HTML, CSS

## License

This project is licensed under the MIT License.

---

Made with ❤️ by Srajan