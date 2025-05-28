import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    return joblib.load("models/heart_model.pkl")

def get_background_data():
    return pd.DataFrame([np.zeros(13)], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])

def app():
    st.subheader("❤️ Heart Disease Prediction")

    inputs = {}
    feature_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

    label_map = {
        'age': ("Age (years)", "Patient's age"),
        'sex': ("Sex", "1 = male, 0 = female"),
        'cp': ("Chest Pain Type", "0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic"),
        'trestbps': ("Resting Blood Pressure (mmHg)", "Typical resting value around 120"),
        'chol': ("Serum Cholesterol (mg/dL)", "Normal range: 125–200 mg/dL"),
        'fbs': ("Fasting Blood Sugar > 120 mg/dL", "1 = true, 0 = false"),
        'restecg': ("Resting ECG Results", "0 = normal, 1 = ST-T wave abnormality, 2 = probable LVH"),
        'thalach': ("Maximum Heart Rate Achieved", "Typical range 100–200"),
        'exang': ("Exercise-Induced Angina", "1 = yes, 0 = no"),
        'oldpeak': ("Oldpeak", "ST depression induced by exercise"),
        'slope': ("Slope of Peak Exercise ST Segment", "0 = upsloping, 1 = flat, 2 = downsloping"),
        'ca': ("Number of Major Vessels Colored by Fluoroscopy", "Range: 0–3"),
        'thal': ("Thalassemia", "1 = normal, 2 = fixed defect, 3 = reversible defect")
    }

    for name in feature_names:
        label, help_text = label_map.get(name, (name, ""))
        inputs[name] = st.number_input(label, format="%f", help=help_text)

    if st.button("Predict"):
        try:
            model = load_model()
            input_data = np.array([list(inputs.values())])

            prediction = model.predict(input_data)[0]
            class_mapping = {
                0: "Healthy",
                1: "Moderate Risk",
                2: "Moderate Risk",
                3: "High Risk"
            }
            disease_label = class_mapping.get(prediction, "Unknown Type")

            condition_mapping = {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-anginal Pain",
                3: "Asymptomatic"
            }
            cp_value = int(inputs.get("cp", 0))
            condition_label = condition_mapping.get(cp_value, "Unknown Condition")
            st.success(f"""🫀 Risk Assessment: {disease_label} (Type {prediction})

🩺 Likely Condition: {condition_label}""")
            if prediction == 3:
                st.info("🔴 High risk of heart disease detected.")
            elif prediction in [1, 2]:
                st.warning("🟠 Moderate risk of heart disease detected.")
            else:
                st.info("🟢 You appear to be healthy. No major heart disease detected.")

            # SHAP Explainability
            def predict_fn(x):
                return model.predict_proba(x)

            background_data = get_background_data()
            input_df = pd.DataFrame(input_data, columns=feature_names)

            explainer = shap.Explainer(predict_fn, background_data)
            shap_values = explainer(input_df)

            st.subheader("🔎 Model Explanation with SHAP")
            st.markdown("""
                This plot helps explain how each feature influenced the model's prediction:
                - **Pink bars** push the result toward the predicted heart disease type.
                - **Blue bars** push it away.
                - Longer bars mean stronger influence.
            """)

            try:
                fig, ax = plt.subplots(figsize=(6, 3))
                shap.plots.bar(shap_values[0][:, prediction], max_display=10, show=False)
                st.pyplot(fig)
            except Exception as shap_err:
                st.warning("⚠️ SHAP explanation could not be rendered due to size limitations.")
                st.text(f"SHAP error: {shap_err}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
