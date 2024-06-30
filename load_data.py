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

import sys

def main(input_args):
    input_dict = {
        'CreditScore': [float(input_args[0])],
        'Geography_Germany': [int(input_args[1])],
        'Geography_Spain': [int(input_args[2])],
        'Gender_Male': [int(input_args[3])],
        'Age': [int(input_args[4])],
        'Tenure': [int(input_args[5])],
        'Balance': [float(input_args[6])],
        'NumOfProducts': [int(input_args[7])],
        'HasCrCard': [int(input_args[8])],
        'IsActiveMember': [int(input_args[9])],
        'EstimatedSalary': [float(input_args[10])]
    }

    custom_input_df = pd.DataFrame(input_dict)
    classifier = joblib.load('DTRModel.pkl')

    prediction = classifier.predict(custom_input_df)

    return prediction

if __name__ == "__main__":
    input_args = sys.argv[1:]
    result = main(input_args)
    print(result)

# custom_data = {
#     'CreditScore': [650],
#     'Geography_Germany': [0],
#     'Geography_Spain': [1],
#     'Gender_Male': [1],
#     'Age': [30],
#     'Tenure': [5],
#     'Balance': [50000.0],
#     'NumOfProducts': [2],
#     'HasCrCard': [1],
#     'IsActiveMember': [0],
#     'EstimatedSalary': [75000.0]
# }

# custom_input_df = pd.DataFrame(custom_data)
# custom_input_df

# classifier = joblib.load('DTRModel.pkl')

# classifier.predict(custom_input_df)