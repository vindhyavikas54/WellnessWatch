import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('C:/Users/M.V.Vindhya/OneDrive/Desktop/CHATBOT/processed-data.csv')
df = df.drop(columns=['Severity_Mild', 'Severity_Moderate'])
df.rename(columns={'Severity_None': 'Target'}, inplace=True)

# Split dataset
x = df.drop(columns=['Target'])
y = df['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(x_train, y_train)

# Streamlit UI
st.title('Aasthma Severity Prediction')
st.write('Enter your symptoms and details to predict aasthma severity.')

# Input fields
input_data = {}
for column in x.columns:
    input_data[column] = st.selectbox(f'{column}', [0, 1])

if st.button('Predict'):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    result = 'No Severity' if prediction[0] == 1 else 'Severe'
    st.write(f'Prediction: {result}')
