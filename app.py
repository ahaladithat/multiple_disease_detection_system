import streamlit as st
from disease_modules import diabetes, parkinsons, heart, kidney  # ✅ include kidney

st.set_page_config(page_title="Multiple Disease Detection", layout="centered")

st.title("🔬 Multiple Disease Detection using ML")

# Sidebar Navigation
disease = st.sidebar.selectbox("Select a Disease to Predict:",
                                ("Diabetes", "Parkinson's", "Heart Disease", "Kidney Disease"))

# Routing to each module
if disease == "Diabetes":
    diabetes.app()
elif disease == "Parkinson's":
    parkinsons.app()
elif disease == "Heart Disease":
    heart.app()
elif disease == "Kidney Disease":   # ✅ Add this line
    kidney.app()
