import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def mb_to_bytes(value):
    """Convert memory size in MB to bytes."""
    if isinstance(value, int):
        value = str(value)
    value = value.strip()
    if 'M' in value:
        return int(float(value.replace('M', '').strip()) * 1048576)
    try:
        return int(float(value))
    except ValueError:
        return value

def binary_labeling(class_value):
    if class_value == 'normal':
        return 0  # 0 for normal
    else:
        return 1 

def preprocess_data(df, label_encoder=None):
    """Preprocess the dataset."""
    df['Bytes'] = df['Bytes'].apply(mb_to_bytes)    
    df['class_encoded'] = df['class'].apply(binary_labeling)
    
    label_encoder = LabelEncoder()
    df['Proto'] = label_encoder.fit_transform(df['Proto'])
    df['Flags'] = label_encoder.fit_transform(df['Flags'])
    
    imputer = SimpleImputer(strategy='mean')  # You can change the strategy as needed

    # Identify numerical features for imputation
    numerical_features = ["Duration","Src Pt","Dst Pt", "Packets"]
    df[numerical_features] = imputer.fit_transform(df[numerical_features])
    
    # X = df.drop(columns=['Date first seen', 'attackDescription', 'attackID', 'attackType', 'Flows', 'Tos', 'class'])
    X = df.drop(columns=['Date first seen', 'attackDescription', 'attackID', 'attackType', 'Flows', 'Tos', 'class'])
    y = df['class_encoded']
    return X, y

def start_build_TT_separate(df_train, df_test):
    """Preprocess the train and test datasets and return them as DataFrames."""
    
    X_train, y_train = preprocess_data(df_train)
    X_test, y_test = preprocess_data(df_test)
    
    return  X_train, X_test, y_train, y_test 

# def start_build_TT_separate(df_train, df_test, validation_size=0.2):
#     """Preprocess the train, validation, and test datasets and return them as DataFrames."""
    
#     # Preprocess the data
#     X_train, y_train = preprocess_data(df_train)
#     X_test, y_test = preprocess_data(df_test)
    
#     # Split the training data into train and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)
    
#     return X_train, X_val, X_test, y_train, y_val, y_test