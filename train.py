# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle

# 1. Load the dataset
data = pd.read_csv('heart.csv')
print("🧩 Columns in dataset:", data.columns.tolist())

# 2. Use 'num' as the target column and convert it to binary:
#    0 remains 0 (no heart disease), any value > 0 becomes 1 (heart disease present)
if 'num' not in data.columns:
    raise Exception("'num' column not found in dataset. Please check your heart.csv")

# Drop 'id' if it exists
X = data.drop(['num', 'id'], axis=1, errors='ignore')
y = data['num'].apply(lambda x: 1 if x > 0 else 0)

# 3. Identify numeric and categorical columns
# Based on your dataset, the following columns are assumed to be numeric:
numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
# And these are categorical:
categorical_cols = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

# 4. Create a preprocessing pipeline for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Fit and transform the feature set
X_processed = preprocessor.fit_transform(X)

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 6. Train the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# 7. Save the trained model and preprocessor
with open('model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("✅ Model and preprocessor saved successfully!")
