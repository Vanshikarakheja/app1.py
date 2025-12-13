import numpy as np
import pickle
import streamlit as st




# Load the model
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
pickle_path = os.path.join(script_dir, 'regressor.pkl')  # no need to hardcode full path

with open(pickle_path, 'rb') as f:
    model = pickle.load(f)





st.title("💊 Medical Insurance Cost Prediction App")
st.write("Enter patient details to predict insurance cost")

# Input fields
age = st.number_input("Age", min_value=1, max_value=100, step=1)

sex = st.selectbox("Sex", ["Male", "Female"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)

children = st.number_input("Number of Children", min_value=0, max_value=5, step=1)

smoker = st.selectbox("Smoker", ["Yes", "No"])

region = st.selectbox(
    "Region",
    ["southeast", "southwest", "northeast", "northwest"]
)

# Encoding (MUST match training)
sex = 0 if sex == "Male" else 1
smoker = 0 if smoker == "Yes" else 1
region_dict = {
    "southeast": 0,
    "southwest": 1,
    "northeast": 2,
    "northwest": 3
}
region = region_dict[region]

if st.button("Predict Insurance Cost"):
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(input_data)
    st.success(f"💰 Estimated Insurance Cost: ${prediction[0]:.2f}")
