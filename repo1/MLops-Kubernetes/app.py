import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('app/Machine_Failure_classification.pkl')

def main():
    st.title("Machine Failure Prediction")

    feature_names = ['footfall','tempMode','AQ','USS','CS','VOC','RP','IP','Temperature']  # Replace with actual feature names
    input_data = {}

    for feature in feature_names:
        input_data[feature] = st.text_input(f"{feature.capitalize()}:")

    input_df = pd.DataFrame([input_data])

    if st.button("Predict"):
        prediction = model.predict(input_df)
        result = 'Failure' if prediction[0] == 1 else 'No Failure'
        st.write(f"Prediction: **{result}**")

if __name__ == '__main__':
    main()
