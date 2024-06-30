from flask import Flask, request, render_template_string
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import joblib
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(open('index.html').read())

@app.route('/submit', methods=['POST'])
def submit():
    # Ambil data dari form
    input_data = {
        'CreditScore': [float(request.form['CreditScore'])],
        'Geography_Germany': [int(request.form['Geography_Germany'])],
        'Geography_Spain': [int(request.form['Geography_Spain'])],
        'Gender_Male': [int(request.form['Gender_Male'])],
        'Age': [int(request.form['Age'])],
        'Tenure': [int(request.form['Tenure'])],
        'Balance': [float(request.form['Balance'])],
        'NumOfProducts': [int(request.form['NumOfProducts'])],
        'HasCrCard': [int(request.form['HasCrCard'])],
        'IsActiveMember': [int(request.form['IsActiveMember'])],
        'EstimatedSalary': [float(request.form['EstimatedSalary'])]
    }

    # Buat dataframe dari data input
    custom_input_df = pd.DataFrame(input_data)
    
    # Load model
    # classifier = joblib.load('DTRModel.pkl')
    # Load the scaler
    scaler = joblib.load("scaler.pkl")
   # Load the model
    model = load_model("model.h5")

    churn = pd.read_csv("Churn_Modelling.csv")
    churn.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    churn = pd.get_dummies(churn, drop_first=True)

    for col in churn.drop('Exited', axis=1).columns:
        if col not in custom_input_df.columns:
            custom_input_df[col] = 0
    # Reorder the custom input data to match the training data columns
    custom_input_df = custom_input_df[churn.drop('Exited', axis=1).columns]

    # Scale the custom input data
    custom_input_scaled = scaler.transform(custom_input_df)

    # Convert data to float32
    custom_input_scaled = np.array(custom_input_scaled, dtype=np.float32)

    # Predict using the model
    predictions = model.predict(custom_input_scaled)
    # print(predictions)

    # Prediksi
    # prediction = classifier.predict(custom_input_df)

    # Tampilkan hasil
    return f'<h1>Output:</h1><p>Prediksi: {predictions[0]}</p>'

if __name__ == '__main__':
    app.run(debug=True)
