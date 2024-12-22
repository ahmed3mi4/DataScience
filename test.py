import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# To display plots inline in Jupyter Notebook
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

path_data = r"C:\Users\RTX\Desktop\4th\Data Science\project\weatherAUS.csv"
#path_data = r"C:\Data Science tasks\DataScience\weatherAUS.csv"
data = pd.read_csv(path_data)
datav1 = data.copy()
# Step 3: Fill missing values for numerical columns
# Fill with mean for temperature-related columns
num_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation','Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
            'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm','Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
for col in num_cols:
    if col in datav1.columns:
        datav1.loc[:, col] = datav1[col].fillna(data[col].mean())

# Step 4: Fill missing values for categorical columns
# Fill with mode for categorical columns
cat_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
for col in cat_cols:
    if col in datav1.columns:
        datav1.loc[:, col] = datav1[col].fillna(data[col].mode()[0])
# Drop extra variables
# Columns to drop: 'Date', 'RISK_MM' (data leakage), and potentially sparse or irrelevant ones
columns_to_drop = ['RISK_MM']
cleaned_data = datav1.drop(columns=columns_to_drop)

# Check the resulting dataset
print("Remaining Columns:\n", datav1.columns)

def drop_outliers(df):
    # Defining the value ranges for each column
    column_ranges = {
        'MinTemp': (-10, 30), 
        'MaxTemp': (10, 45), 
        'Rainfall': (0, 500), 
        'Evaporation': (0, 10),
        'Sunshine': (0, 14),
        'WindGustSpeed': (0, 150),
        'WindSpeed9am': (0, 60),
        'WindSpeed3pm': (0, 70),
        'Humidity9am': (30, 90),
        'Humidity3pm': (20, 80),
        'Pressure9am': (1000, 1025),
        'Pressure3pm': (1000, 1025),
        'Cloud9am': (0, 100),
        'Cloud3pm': (0, 100),
        'Temp9am': (10, 30),
        'Temp3pm': (15, 45)
    }
    
    # Iterate over the columns and apply the range filter
    for col, (min_val, max_val) in column_ranges.items():
        if col in df.columns:
            df = df[(df[col] >= min_val) & (df[col] <= max_val)]
    
    return df
cleaned_data_or = drop_outliers(cleaned_data)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Select numerical columns to scale and normalize
num_cols = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation','Sunshine', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
            'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm','Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']

# Standardize (Z-score) scaling
scaler = StandardScaler()
cleaned_data_or[num_cols] = scaler.fit_transform(cleaned_data_or[num_cols])

# Min-Max normalization
normalizer = MinMaxScaler()
cleaned_data_or[num_cols] = normalizer.fit_transform(cleaned_data_or[num_cols])

# Verify scaling by checking the summary statistics
cleaned_data_or[num_cols].describe()

def arrange_data(df):
    # Ensure that 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d',errors='coerce')
    
    # Sort data by 'Date'
    df = df.sort_values(by='Date', ascending=True)
    
    return df
data_arranged = arrange_data(cleaned_data_or)

def group_data(df):
    
    # Select only numeric columns for aggregation
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Group rows by Location and calculate mean for numerical columns
    grouped_by_location = df.groupby('Location')[numeric_cols].mean().reset_index()

    # Aggregate data by year and month for trends
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    grouped_by_month = df.groupby(['Year', 'Month'])[numeric_cols].mean().reset_index()

    # Group columns logically (keeping all columns for inspection)
    temp_related = df[['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm']]
    rain_related = df[['Rainfall', 'RainToday', 'RainTomorrow']]
    wind_related = df[['WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm']]
    humidity_pressure = df[['Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm']]
    cloud_sunshine = df[['Cloud9am', 'Cloud3pm', 'Sunshine']]

    # Return grouped data
    return {
        "grouped_by_location": grouped_by_location,
        "grouped_by_month": grouped_by_month,
        "temperature_related": temp_related,
        "rain_related": rain_related,
        "wind_related": wind_related,
        "humidity_and_pressure": humidity_pressure,
        "cloud_and_sunshine": cloud_sunshine
    }
grouped_data = group_data(data_arranged)

def standardize_text_columns(df, columns):
    for col in columns:
        # Strip extra spaces
        df[col] = df[col].str.strip()

        # Convert to lowercase
        df[col] = df[col].str.lower()

        # Replace common placeholders with NaN
        df[col] = df[col].replace(['?', 'na', 'n/a', 'none'], pd.NA)

    return df
text_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow']
data_str = standardize_text_columns(data_arranged, text_cols)

wind_dir_mapping = {
    'n': 'north',
    's': 'south',
    'e': 'east',
    'w': 'west',
    'ne': 'northeast',
    'se': 'southeast',
    'sw': 'southwest',
    'nw': 'northwest',
    'nne': 'north-northeast',
    'ene': 'east-northeast',
    'ese': 'east-southeast',
    'sse': 'south-southeast',
    'ssw': 'south-southwest',
    'wsw': 'west-southwest',
    'wnw': 'west-northwest',
    'nnw': 'north-northwest'
}
def standardize_wind_directions(df, columns, mapping):
    for col in columns:
        # Convert to lowercase
        df[col] = df[col].str.lower()
        # Map to standardized names
        df[col] = df[col].map(mapping)
    return df

# Apply standardization to the relevant columns
wind_columns = ['WindGustDir', 'WindDir9am','WindDir3pm']
data_str = standardize_wind_directions(data_str, wind_columns, wind_dir_mapping)

# Convert Yes/No to 1/0 or lowercase for consistency
data_str['RainToday'] = data_str['RainToday'].map({'yes': 1, 'no': 0}).astype('Int64')
data_str['RainTomorrow'] = data_str['RainTomorrow'].map({'yes': 1, 'no': 0}).astype('Int64')
# Features and target
x = data_str.drop(columns=['Date','RainTomorrow','Year','Month'])
y = data_str['RainTomorrow']
# Identify categorical and numerical columns
categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'Location']
numerical_cols = x.select_dtypes(include=['float64', 'int64']).columns

# Apply Label Encoding to categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    label_encoders[col] = le  # Save encoder for potential inverse transformations

import joblib

# Save the label encoders and scaler
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Step 3: Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

import joblib


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train_scaled, y_train)

# Evaluate the model
y_pred = rf_model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Save the trained model, scaler, and label encoders
joblib.dump(rf_model, 'rain_prediction_rf_model.pkl')


print("Model and preprocessing components saved successfully!")
