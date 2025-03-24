import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Split the dataset into features and target variable
    x = df.drop(columns='fail')
    y = df['fail']  # Use the correct column name here

    # Preprocessing pipeline
    preprocess_pipeline = Pipeline(
        steps=[
            ('scaler', StandardScaler())
        ]
    )

    # Apply the preprocessing to the features
    x_processed = preprocess_pipeline.fit_transform(x)
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_processed, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
