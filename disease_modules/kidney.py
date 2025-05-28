import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

@st.cache_resource
def load_model():
    model = joblib.load("models/kidney_model.pkl")
    scaler = joblib.load("models/kidney_scaler.pkl")
    return model, scaler

def get_background_data():
    return pd.DataFrame([np.zeros(24)], columns=[
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
        'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
        'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
    ])

def app():
    st.subheader("🩺 Kidney Disease Prediction")

    inputs = {}
    feature_names = [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
        'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
        'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
    ]

    label_map = {
        'age': ("Age (years)", "Patient's age"),
        'bp': ("Blood Pressure (mmHg)", "Measured blood pressure"),
        'sg': ("Specific Gravity", "Urine concentration (e.g., 1.005–1.025)"),
        'al': ("Albumin", "Protein level in urine (0–5)"),
        'su': ("Sugar", "Sugar in urine (0–5)"),
        'rbc': ("Red Blood Cells", "1 = abnormal, 0 = normal"),
        'pc': ("Pus Cell", "1 = abnormal, 0 = normal"),
        'pcc': ("Pus Cell Clumps", "1 = present, 0 = not present"),
        'ba': ("Bacteria", "1 = present, 0 = not present"),
        'bgr': ("Blood Glucose (mg/dL)", "Blood sugar level"),
        'bu': ("Blood Urea (mg/dL)", "Urea in blood"),
        'sc': ("Serum Creatinine (mg/dL)", "Kidney filtration marker"),
        'sod': ("Sodium (mEq/L)", "Sodium level in blood"),
        'pot': ("Potassium (mEq/L)", "Potassium level in blood"),
        'hemo': ("Hemoglobin (g/dL)", "Oxygen-carrying protein in blood"),
        'pcv': ("Packed Cell Volume (%)", "Volume of red blood cells"),
        'wc': ("White Blood Cell Count (cells/cumm)", "Infection indicator"),
        'rc': ("Red Blood Cell Count (millions/cmm)", "Blood cell volume"),
        'htn': ("Hypertension", "1 = yes, 0 = no"),
        'dm': ("Diabetes Mellitus", "1 = yes, 0 = no"),
        'cad': ("Coronary Artery Disease", "1 = yes, 0 = no"),
        'appet': ("Appetite", "1 = good, 0 = poor"),
        'pe': ("Pedal Edema", "1 = yes, 0 = no"),
        'ane': ("Anemia", "1 = yes, 0 = no")
    }

    for name in feature_names:
        label, help_text = label_map.get(name, (name, ""))
        inputs[name] = st.number_input(label, format="%f", help=help_text)

    if st.button("Predict"):
        try:
            model, scaler = load_model()
            input_data = np.array([list(inputs.values())])
            input_data = scaler.transform(input_data)

            prediction = model.predict(input_data)[0]
            st.success("🛑 Kidney Disease Detected" if prediction == 1 else "✅ No Kidney Disease Detected")

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
                - **Pink bars** push the result toward kidney disease.
                - **Blue bars** push the result toward healthy.
                - Longer bars mean stronger influence.
            """)

            try:
                fig, ax = plt.subplots(figsize=(8, 4))
                shap.plots.bar(shap_values[0][:, 1], max_display=10, show=False)
                st.pyplot(fig)
            except Exception as shap_err:
                st.warning("⚠️ SHAP explanation could not be rendered due to size limitations.")
                st.text(f"SHAP error: {shap_err}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
