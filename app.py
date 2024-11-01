import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model

def main():
    st.title("Weather Prediction Model")
    
    # Upload data
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset preview:", data.head())
        
        # Load model
        model = load_model("scripts/weather_model.h5")  # Placeholder path
        # Here, add prediction code using the model
        
        st.write("Predictions coming soon...")

if __name__ == "__main__":
    main()