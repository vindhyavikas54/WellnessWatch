import pandas as pd
import numpy as np
import streamlit as st
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv('heart_2020_cleaned.csv')

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split dataset into features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model and scalers
joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Load saved models
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Streamlit UI
st.title('Heart Disease Prediction')
st.write('Enter your health parameters to predict the likelihood of heart disease.')

input_data = {}
for column in X.columns:
    if column in label_encoders:
        input_data[column] = st.selectbox(column, label_encoders[column].classes_)
    else:
        input_data[column] = st.number_input(column, value=0.0)

if st.button('Predict'):
    # Create DataFrame with proper column names
    input_df = pd.DataFrame([input_data], columns=X.columns)
    
    # Encode categorical values
    for column, le in label_encoders.items():
        if column in input_df.columns:
            input_df[column] = le.transform([input_df[column]])[0]

    # Scale the inputs
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    result = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f'Prediction: Heart Disease - {result}')
