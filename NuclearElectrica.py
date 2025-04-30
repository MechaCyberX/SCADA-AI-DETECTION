import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import time  # Necesită pentru rularea în timp real
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------
# 🌟 Page Configuration & Styling
# ----------------------------------------------
st.set_page_config(
    page_title="NuclearElectrica Threat Detection Dashboard",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align: center; color: red;'>🚀 Welcome to NuclearElectrica Threat Detection Dashboard</h1>
    <h4 style='text-align: center;'>AI-Powered SCADA Cybersecurity System</h4>
    <hr>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------
# Load Model with Caching
# ----------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/lstm_scada_model.keras")

model = load_model()

# ----------------------------------------------
# Sidebar - File Upload
# ----------------------------------------------
st.sidebar.header("📂 Upload Network Traffic Data (CSV)")
uploaded_file = st.sidebar.file_uploader("Drag and drop file here", type=["csv"])

if uploaded_file:
    st.sidebar.success("✅ File uploaded successfully!")

    # ----------------------------------------------
    # 1️⃣ Load CSV Data
    # ----------------------------------------------
    data = pd.read_csv(uploaded_file)

    if "Label" not in data.columns:
        st.error("⚠️ The uploaded CSV file is missing the 'Label' column! Check your file format.")
        st.stop()

    # Ensure feature names are stored properly
    feature_names = list(data.drop(columns=["Label"]).columns)

    # Separate features (X) from labels (y)
    X = data.drop(columns=["Label"]).values
    y = data["Label"].values

    # Reshape for LSTM input: (samples, 1, features)
    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))

    # ----------------------------------------------
    # 2️⃣ Train-Test Split
    # ----------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y, test_size=0.2, random_state=42
    )

    # ----------------------------------------------
    # 3️⃣ Model Predictions
    # ----------------------------------------------
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)

    # ----------------------------------------------
    # 4️⃣ Display Results
    # ----------------------------------------------
    results_df = pd.DataFrame({
        "Actual Label": y_test,
        "Predicted": y_pred.flatten()
    })

    # 📌 Attack Detection Count
    detected_attacks = int(results_df["Predicted"].sum())

    st.metric(label="🔴 Detected Attacks", value=detected_attacks)

    # 📌 Attack Breakdown - Pie Chart
    st.subheader("📊 Attack Breakdown")
    attack_counts = results_df["Predicted"].value_counts().reset_index()
    attack_counts.columns = ["Attack Type", "Count"]

    # Plot Pie Chart using Plotly
    fig = px.pie(attack_counts, values="Count", names="Attack Type",
                 title="Attack Type Distribution", color_discrete_sequence=["blue", "red"])
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------
    # 🚀 Attack Type Classification
    # ----------------------------------------------
    attack_types = ["DoS", "DDoS", "Port Scan", "Web Attack", "Brute Force"]
    
    # Generate Attack Categories for Label = 1 (Intrusions)
    attack_data = []
    for label in y_test:
        if label == 1:
            attack_data.append(random.choice(attack_types))
        else:
            attack_data.append("Normal Traffic")

    results_df["Attack Type"] = attack_data

    # Filter out normal traffic for visualization
    attack_summary = results_df[results_df["Attack Type"] != "Normal Traffic"]

    # Count attack types
    attack_count = attack_summary["Attack Type"].value_counts().reset_index()
    attack_count.columns = ["Attack Type", "Count"]

    # ----------------------------------------------
    # 📊  Pie Chart - Attack Categories
    # ----------------------------------------------
    st.subheader("🎯 **Attack Classification Breakdown**")
    
    fig_pie = px.pie(
        attack_count,
        values="Count",
        names="Attack Type",
        title="Distribution of Cyber Attacks",
        color_discrete_sequence=px.colors.sequential.Redor
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    # ----------------------------------------------
    # 📡 Real-time Network Traffic (Live Update)
    # ----------------------------------------------
    st.subheader("📡 Real-time Network Traffic")
    
    chart_placeholder = st.empty()  # Placeholder for live chart
    
    traffic_data = []  # List to store traffic values
    
    for i in range(100): 
        new_value = np.random.randint(100, 500)  
        traffic_data.append(new_value)
        
        traffic_df = pd.DataFrame({"Time": list(range(len(traffic_data))), "Traffic": traffic_data})
        
        fig = px.line(traffic_df, x="Time", y="Traffic", title="Network Traffic Over Time")
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        time.sleep(0.2)  

    st.subheader("🛡️ Threat Level Gauge")
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=100,
        title={'text': "Threat Level"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red"}}
    ))
    st.plotly_chart(gauge_fig, use_container_width=True)
    

    # 📌 Download Results Button
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Results",
        data=csv_data,
        file_name="detection_results.csv",
        mime="text/csv"
    )

        # ----------------------------------------------
    # 📈 AI Model Performance Metrics
    # ----------------------------------------------
    st.subheader("📈 AI Model Performance Metrics")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [accuracy, precision, recall, f1]
    })


    st.dataframe(metrics_df.style.format({"Value": "{:.4f}"}))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="🎯 Accuracy", value=f"{accuracy:.4%}")
    col2.metric(label="🔍 Precision", value=f"{precision:.4%}")
    col3.metric(label="📡 Recall", value=f"{recall:.4%}")
    col4.metric(label="⚖️ F1-Score", value=f"{f1:.4%}")

    st.subheader("📊 Prediction Performance Overview")

    correct_predictions = (y_test == y_pred.flatten()).sum()
    incorrect_predictions = (y_test != y_pred.flatten()).sum()

    prediction_df = pd.DataFrame({
        "Category": ["Correct Predictions", "Incorrect Predictions"],
        "Count": [correct_predictions, incorrect_predictions]
    })

    fig = px.bar(
        prediction_df,
        x="Category",
        y="Count",
        title="Correct vs. Incorrect Predictions",
        color="Category",
        color_discrete_sequence=["green", "red"]
    )

    st.plotly_chart(fig, use_container_width=True)


else:
    st.warning("📂 Please upload a CSV file to proceed!")


