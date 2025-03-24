import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\Shreyansh Singh\Desktop\ML3\data\data.csv")

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Split the dataset into features and target variable
x = df.drop(columns='fail')
y = df['fail']  # Use the correct column name here

# Preprocessing pipeline
preprocess_pipeline = Pipeline(
    steps=[
        ('scaler', StandardScaler())
    ])

# Model pipeline
model_pipeline = Pipeline(
    steps=[
        ('model', RandomForestClassifier())  # Instantiate the classifier
    ])

# Full pipeline combining preprocessing and model
pipeline = Pipeline(
    steps=[
        ('preprocessing', preprocess_pipeline),
        ('model', model_pipeline)
    ])

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = pipeline.fit(x_train, y_train)

# Make predictions
prediction = pipeline.predict(x_test)

# Evaluate the model
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, prediction))

# Save the model
import joblib
joblib.dump(model, 'Machine_Failure_classification.pkl')
