import mlflow
import pandas as pd
import numpy as np
import joblib 
import mlflow.sklearn
from sklearn.metrics import accuracy_score



model  = joblib.load(r"C:\Users\Shreyansh Singh\Desktop\ML3\app\Machine_Failure_classification.pkl")

df = pd.read_csv(r"C:\Users\Shreyansh Singh\Desktop\ML3\data\data.csv")

x = df.drop(columns='fail')
y = df['fail']

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)

prediction = model.predict(x_test)
accuracy = accuracy_score(prediction,y_test)

with mlflow.start_run():
    mlflow.sklearn.log_model(model , 'Failure_classification.pkl')
    mlflow.log_metric("Accuracy",accuracy)
    df.to_csv('machine_failure.csv' , index = False)
    mlflow.log_artifact('machine_failure.csv')

print("Logging Sucessfully done  ")

joblib.dump(model , 'Machine_Failure_classification.pkl')
    