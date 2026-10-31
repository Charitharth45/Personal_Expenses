import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# --- Page Layout Configuration ---
st.set_page_config(page_title="Credit Card Analytics Dashboard", layout="wide")
st.title("💳 Credit Card Transactions & Spend Prediction Dashboard")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    # Read the data from your CSV file
    df = pd.read_csv('cleaned_transactions.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Day'] = df['Date'].dt.day
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ 'cleaned_transactions.csv' not found. Please place it in this directory.")
    st.stop()

# --- 2. Train ML Model Pipeline (Cached for Performance) ---
@st.cache_resource
def train_model(data):
    features = ['City', 'Card Type', 'Exp Type', 'Gender', 'Year', 'Month', 'DayOfWeek', 'Day']
    X = data[features]
    y = data['Amount']
    
    categorical_features = ['City', 'Card Type', 'Exp Type', 'Gender']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))
    ])
    pipeline.fit(X, y)
    return pipeline

model = train_model(df)

# --- Sidebar Filters ---
st.sidebar.header("🔍 Global Data Filters")
selected_city = st.sidebar.multiselect("Select Cities", options=df['City'].unique(), default=df['City'].unique())
selected_card = st.sidebar.multiselect("Select Card Types", options=df['Card Type'].unique(), default=df['Card Type'].unique())

# Filtered View
filtered_df = df[(df['City'].isin(selected_city)) & (df['Card Type'].isin(selected_card))]

# --- Layout: Key Performance Metrics ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Spending Volume", f"₹{filtered_df['Amount'].sum():,.2f}")
with col2:
    st.metric("Total Transactions", f"{filtered_df.shape[0]:,}")
with col3:
    st.metric("Average Transaction Value", f"₹{filtered_df['Amount'].mean():,.2f}")

st.markdown("---")

# --- Layout: Main Analytics Visualisations ---
tab1, tab2 = st.tabs(["📊 Spending Analytics Trends", "🔮 Predictive AI Spends Engine"])

with tab1:
    st.subheader("Historical Transaction Breakdowns")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Pie Chart: Spending Share by Expense Type
        fig, ax = plt.subplots(figsize=(6, 5))
        exp_shares = filtered_df.groupby('Exp Type')['Amount'].sum()
        ax.pie(exp_shares, labels=exp_shares.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        ax.set_title("Spending Share by Expense Category")
        st.pyplot(fig)
        
    with chart_col2:
        # Bar Chart: Spending Breakdown by City
        fig, ax = plt.subplots(figsize=(8, 5.5))
        city_spend = filtered_df.groupby('City')['Amount'].sum().sort_values(ascending=False)
        sns.barplot(x=city_spend.values, y=city_spend.index, ax=ax, palette='viridis')
        ax.set_title("Transaction Value Totals by City")
        ax.set_xlabel("Total Spend (₹)")
        st.pyplot(fig)

with tab2:
    st.subheader("Predict Customer Transaction Value")
    st.write("Input transactional metadata attributes below to predict expected customer credit spending limits.")
    
    # User Feature Inputs for Live Model Scoring
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    with p_col1:
        input_city = st.selectbox("Customer City Location", options=df['City'].unique())
    with p_col2:
        input_card = st.selectbox("Card Category tier", options=df['Card Type'].unique())
    with p_col3:
        input_exp = st.selectbox("Target Expense Area", options=df['Exp Type'].unique())
    with p_col4:
        input_gender = st.radio("Gender Profile", options=df['Gender'].unique(), horizontal=True)
        
    # Execute Model Pipeline Prediction on User Metrics
    if st.button("🔮 Calculate Estimated Spending Target"):
        # Synthesize temporary feature set vector matrix
        mock_profile = pd.DataFrame([{
            'City': input_city, 'Card Type': input_card, 'Exp Type': input_exp, 'Gender': input_gender,
            'Year': 2026, 'Month': 10, 'DayOfWeek': 5, 'Day': 31
        }])
        
        prediction = model.predict(mock_profile)[0]
        st.success(f"🎯 **Predicted Estimated Transaction Cost Bounds:** ₹{prediction:,.2f}")
