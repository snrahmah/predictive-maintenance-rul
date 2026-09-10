import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import torch
import torch.nn as nn

st.set_page_config(page_title="Turbofan Engine RUL Prediction", layout="wide")

st.title("Turbofan Engine Remaining Useful Life (RUL) Prediction")
st.write("Predictive maintenance system based on NASA CMAPSS FD001 dataset")

# Load artifacts
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load("models/rf_rul_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    
    with open("models/feature_cols.json") as f:
        feature_cols = json.load(f)
    
    with open("models/active_sensors.json") as f:
        active_sensors = json.load(f)
    
    with open("models/lstm_config.json") as f:
        lstm_config = json.load(f)
    
    return rf_model, scaler, feature_cols, active_sensors, lstm_config

rf_model, scaler, feature_cols, active_sensors, lstm_config = load_artifacts()

st.success("Models and artifacts loaded successfully")
st.write("Number of features:", len(feature_cols))
st.write("Active sensors:", active_sensors)