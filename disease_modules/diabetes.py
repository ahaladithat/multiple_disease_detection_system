import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("models/diabetes_model.pkl")

# Dummy background data with correct feature names matching model training
def get_background_data():
    return pd.DataFrame(np.array([[0, 100, 70, 20, 79, 25.0, 0.5, 33]]),
                        columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])

def app():
    st.subheader("🩸 Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies", 0, 20)
    glucose = st.number_input("Glucose", 0, 200)
    blood_pressure = st.number_input("Blood Pressure", 0, 150)
    skin_thickness = st.number_input("Skin Thickness", 0, 100)
    insulin = st.number_input("Insulin", 0, 900)
    bmi = st.number_input("BMI", 0.0, 70.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.number_input("Age", 1, 120)

    if st.button("Predict"):
        try:
            model = load_model()
            input_data = np.array([[pregnancies, glucose, blood_pressure,
                                    skin_thickness, insulin, bmi, dpf, age]])

            prediction = model.predict(input_data)[0]
            st.success("✅ Positive for Diabetes" if prediction == 1 else "❎ Negative for Diabetes")

            # SHAP Explainability (fixed for SVC)
            def predict_fn(x):
                return model.predict_proba(x)

            background_data = get_background_data()
            feature_names = background_data.columns
            input_df = pd.DataFrame(input_data, columns=feature_names)

            explainer = shap.Explainer(predict_fn, background_data)
            shap_values = explainer(input_df)

            st.subheader("🧠 Model Explanation with SHAP")
            st.markdown("""
                The SHAP (SHapley Additive exPlanations) plot below shows how each input feature contributed to the prediction.

                - Features in **pink** pushed the prediction **toward diabetes (1)**.
                - Features in **blue** pushed the prediction **away from diabetes (0)**.
                - The **length** of each bar shows how much that feature influenced the model’s decision.

                This helps you understand **why** the model gave this result.
            """)

            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0][:, 1], max_display=8, show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {e}")
