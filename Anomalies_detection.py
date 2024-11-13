import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Function to load data
def load_data(uploaded_file):
    """Loads the loan book data from an uploaded Excel or CSV file."""
    if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        data = pd.read_excel(uploaded_file)
    else:
        data = pd.read_csv(uploaded_file)
    return data

# Function to preprocess the data
def preprocess_data(data):
    """Preprocesses the loan book data."""
    data.fillna(0, inplace=True)
    numeric_cols = [
        'LTV', 'Collateral Value', 'Time Taken to Be Disbursed',
        'SANCTION_AMOUNT', 'Disbursement date', 'ROI',
        'Processing Fees', 'CIBIl Score', 'Term of loan / Tenure',
        'EMI amount'
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data

# Streamlit UI
st.title("Anomaly Detection in Loan Book Data")
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Load and preprocess the data
    loan_data = load_data(uploaded_file)
    loan_data = preprocess_data(loan_data)
    
    st.write("Data loaded successfully.")
    st.dataframe(loan_data.head())

    if st.button("Run Anomaly Detection"):
        # Select all numeric features for the Isolation Forest model
        numeric_cols = [
            'LTV', 'Collateral Value', 'Time Taken to Be Disbursed',
            'SANCTION_AMOUNT', 'Disbursement date', 'ROI',
            'Processing Fees', 'CIBIl Score', 'Term of loan / Tenure',
            'EMI amount'
        ]
        X = loan_data[numeric_cols]

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize and train the Isolation Forest model
        iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        loan_data['anomaly'] = iso_forest.fit_predict(X_scaled)

        # Map the output: 1 = Normal, -1 = Anomaly
        loan_data['anomaly'] = loan_data['anomaly'].map({1: 0, -1: 1})  # 0 = Normal, 1 = Anomaly

        # Display the number of anomalies detected
        st.write(f"Number of anomalies detected: {loan_data['anomaly'].sum()}")

        # Display a few rows of the data with the anomaly label
        st.dataframe(loan_data.head())

        # Plot the results
        plt.figure(figsize=(10, 6))
        colors = {0: 'blue', 1: 'red'}
        sns.scatterplot(
            x=loan_data['SANCTION_AMOUNT'], y=loan_data['Collateral Value'],
            hue=loan_data['anomaly'], palette=colors, alpha=0.6
        )
        plt.title('Anomaly Detection: Sanction Amount vs. Collateral Value')
        plt.xlabel('Sanction Amount')
        plt.ylabel('Collateral Value')
        st.pyplot(plt)

        # Save anomalies and normal data to separate DataFrames
        anomalies = loan_data[loan_data['anomaly'] == 1]
        normal_data = loan_data[loan_data['anomaly'] == 0]

        # Download options
        csv_anomalies = anomalies.to_csv(index=False).encode('utf-8')
        csv_normal = normal_data.to_csv(index=False).encode('utf-8')

        st.download_button("Download Anomalies CSV", data=csv_anomalies, file_name='loan_book_anomalies.csv', mime='text/csv')
        st.download_button("Download Normal Data CSV", data=csv_normal, file_name='loan_book_normal_data.csv', mime='text/csv')
