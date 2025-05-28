import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

@st.cache_resource
def load_model():
    return joblib.load("models/parkinsons_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("models/parkinsons_scaler.pkl")

def get_background_data():
    return pd.DataFrame([np.zeros(22)], columns=[
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
        'spread1', 'spread2', 'D2', 'PPE'
    ])

def app():
    st.subheader("🧠 Parkinson's Disease Prediction")

    inputs = {}
    feature_names = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
        'spread1', 'spread2', 'D2', 'PPE'
    ]

    label_map = {
        'MDVP:Fo(Hz)': ("Average vocal fundamental frequency", "Typical range 120-260 Hz"),
        'MDVP:Fhi(Hz)': ("Maximum vocal frequency", "Maximum pitch of voice"),
        'MDVP:Flo(Hz)': ("Minimum vocal frequency", "Minimum pitch of voice"),
        'MDVP:Jitter(%)': ("Jitter (%)", "Frequency variation of voice"),
        'MDVP:Jitter(Abs)': ("Absolute Jitter", "Micro variation in frequency"),
        'MDVP:RAP': ("Relative Average Perturbation", "Variation measure in voice signal"),
        'MDVP:PPQ': ("Pitch Period Perturbation Quotient", "Cycle-to-cycle variation in voice pitch"),
        'Jitter:DDP': ("Differential Perturbation", "3x RAP - alternative jitter measure"),
        'MDVP:Shimmer': ("Shimmer", "Amplitude variation in voice"),
        'MDVP:Shimmer(dB)': ("Shimmer (dB)", "Decibel-based shimmer"),
        'Shimmer:APQ3': ("Amplitude Perturbation Quotient (3)" , "Short-term shimmer"),
        'Shimmer:APQ5': ("Amplitude Perturbation Quotient (5)", "Smoothed shimmer"),
        'MDVP:APQ': ("Amplitude Perturbation Quotient", "Overall shimmer measure"),
        'Shimmer:DDA': ("Difference of Differences of Amplitude", "3x APQ3"),
        'NHR': ("Noise-to-Harmonics Ratio", "Noise in voice signal"),
        'HNR': ("Harmonics-to-Noise Ratio", "Signal clarity indicator"),
        'RPDE': ("Recurrence Period Density Entropy", "Nonlinear measure of voice disorder"),
        'DFA': ("Detrended Fluctuation Analysis", "Fractal scaling index"),
        'spread1': ("Nonlinear Spread 1", "Voice distribution pattern"),
        'spread2': ("Nonlinear Spread 2", "Another voice spread measure"),
        'D2': ("Correlation Dimension", "Chaotic voice pattern complexity"),
        'PPE': ("Pitch Period Entropy", "Uncertainty in pitch frequency")
    }

    for name in feature_names:
        label, help_text = label_map.get(name, (name, ""))
        inputs[name] = st.number_input(label, format="%f", help=help_text)

    if st.button("Predict"):
        try:
            model = load_model()
            scaler = load_scaler()
            input_data = np.array([list(inputs.values())])
            scaled_input = scaler.transform(input_data)

            prediction = model.predict(scaled_input)[0]
            st.success("🧠 Likely Parkinson's Detected" if prediction == 1 else "✅ No Parkinson's Detected")

            # SHAP Explainability
            def predict_fn(x):
                return model.predict_proba(scaler.transform(x))

            background_data = get_background_data()
            input_df = pd.DataFrame(input_data, columns=feature_names)

            explainer = shap.Explainer(predict_fn, background_data)
            shap_values = explainer(input_df)

            st.subheader("🔎 Model Explanation with SHAP")
            st.markdown("""
                This plot helps explain how each feature influenced the model's prediction:
                - **Pink bars** push the result toward Parkinson's (1).
                - **Blue bars** push the result toward no Parkinson's (0).
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
