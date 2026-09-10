# import and load artifacts
#--------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.set_page_config(page_title="Turbofan Engine RUL Prediction", layout="wide")

st.title("Turbofan Engine Remaining Useful Life (RUL) Prediction")
st.write("Predictive maintenance system based on NASA CMAPSS FD001 dataset")

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


# redefining the LSTM architecture
#---------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()

@st.cache_resource
def load_lstm_model():
    model = LSTMModel(
        input_size=lstm_config["input_size"],
        hidden_size=lstm_config["hidden_size"],
        num_layers=lstm_config["num_layers"]
    )
    model.load_state_dict(torch.load("models/lstm_rul_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

lstm_model = load_lstm_model()


# sidebar: selection of models and data input methods
#----------------------------------------------------

st.sidebar.header("Settings")

model_choice = st.sidebar.radio(
    "Choose prediction model:",
    ["Random Forest", "LSTM"]
)

input_method = st.sidebar.radio(
    "Choose input method:",
    ["Upload CSV", "Use sample data"]
)


# preprocessing function
# ----------------------

def add_rolling(df, cols, window=5):
    df = df.sort_values(["unit_number", "time_cycles"]).copy()
    for col in cols:
        df[f"{col}_rollmean"] = df.groupby("unit_number")[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    return df

def preprocess_input(df, scaler, active_sensors, feature_cols):
    op_settings = ["op_setting_1", "op_setting_2", "op_setting_3"]
    cols_to_scale = op_settings + active_sensors
    
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = add_rolling(df, active_sensors, window=5)
    
    return df


# data input (upload the csv)
#----------------------------

st.header("Input Data")

col_names = ["unit_number", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"] + \
            [f"sensor_{i}" for i in range(1, 22)]

df_input = None

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload engine sensor data (space-separated .txt or .csv, same format as CMAPSS test data)",
        type=["txt", "csv"]
    )
    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file, sep=r"\s+", header=None, names=col_names)
        st.success(f"Data loaded: {df_input.shape[0]} rows, {df_input['unit_number'].nunique()} engine units")

else:
    st.info("Using built-in sample data (5 example engine units)")
    np.random.seed(42)
    sample_rows = []
    for unit in range(1, 6):
        n_cycles = np.random.randint(50, 150)
        for cycle in range(1, n_cycles + 1):
            row = [unit, cycle] + list(np.random.uniform(0, 1, 3)) + list(np.random.uniform(0, 1, 21))
            sample_rows.append(row)
    df_input = pd.DataFrame(sample_rows, columns=col_names)


# prediction
#-----------

if df_input is not None:
    st.header("Prediction Results")
    
    df_processed = preprocess_input(df_input.copy(), scaler, active_sensors, feature_cols)
    
    results = []
    
    for unit in df_processed["unit_number"].unique():
        unit_data = df_processed[df_processed["unit_number"] == unit].sort_values("time_cycles")
        
        if model_choice == "Random Forest":
            last_row = unit_data.iloc[[-1]][feature_cols]
            pred_rul = rf_model.predict(last_row)[0]
        
        else:
            seq_len = lstm_config["sequence_length"]
            data_array = unit_data[feature_cols].values
            
            if len(data_array) < seq_len:
                pad_length = seq_len - len(data_array)
                padding = np.tile(data_array[0], (pad_length, 1))
                data_array = np.vstack([padding, data_array])
            
            sequence = data_array[-seq_len:]
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                pred_rul = lstm_model(sequence_tensor).item()
        
        results.append({"unit_number": unit, "predicted_RUL": round(pred_rul, 1)})
    
    results_df = pd.DataFrame(results)


#result (visualize)
#--------------------

    def get_status(rul):
        if rul < 20:
            return "Critical"
        elif rul < 50:
            return "Warning"
        else:
            return "Healthy"
    
    results_df["status"] = results_df["predicted_RUL"].apply(get_status)
    
    st.dataframe(results_df, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Units", len(results_df))
    with col2:
        critical_count = (results_df["status"] == "Critical").sum()
        st.metric("Critical Units", critical_count)
    with col3:
        avg_rul = results_df["predicted_RUL"].mean()
        st.metric("Average Predicted RUL", f"{avg_rul:.1f} cycles")
    
    st.subheader("Predicted RUL by Engine Unit")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = results_df["status"].map({"Critical": "red", "Warning": "orange", "Healthy": "green"})
    ax.bar(results_df["unit_number"].astype(str), results_df["predicted_RUL"], color=colors)
    ax.set_xlabel("Engine Unit")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"Predicted RUL - {model_choice}")
    st.pyplot(fig)
    
    st.subheader("Maintenance Priority")
    priority_df = results_df.sort_values("predicted_RUL").reset_index(drop=True)
    for idx, row in priority_df.iterrows():
        if row["status"] == "Critical":
            st.error(f"Unit {row['unit_number']}: RUL = {row['predicted_RUL']} cycles - Immediate inspection recommended")
        elif row["status"] == "Warning":
            st.warning(f"Unit {row['unit_number']}: RUL = {row['predicted_RUL']} cycles - Schedule maintenance soon")

else:
    st.info("Please upload data or select sample data to see predictions")


# model comparison info 
#------------------------

st.sidebar.header("Model Performance")
comparison_df = pd.read_csv("models/model_comparison.csv")
st.sidebar.dataframe(comparison_df, use_container_width=True)

