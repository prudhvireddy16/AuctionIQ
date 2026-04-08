import streamlit as st
import pandas as pd
import joblib

# 1. Load the AI Model and the list of columns
model = joblib.load('auction_model.pkl')
model_columns = joblib.load('model_columns.pkl')

# 2. UI Setup
st.set_page_config(page_title="AuctionIQ Price Predictor", layout="wide")
st.title(" AuctionIQ: Predictive Price Engine")
st.write("Enter vehicle details below to estimate the final **Hammer Price**.")

# 3. User Input Layout
col1, col2 = st.columns(2)

with col1:
    year = st.slider("Vehicle Year", 1960, 2024, 2010)
    mileage = st.number_input("Mileage", 0, 300000, 50000)
    condition = st.slider("Condition Score (1=Poor, 10=Pristine)", 1.0, 10.0, 7.5)

with col2:
    make = st.selectbox("Vehicle Make", ['Ford', 'Chevy', 'Porsche', 'Ferrari', 'Toyota'])
    location = st.selectbox("Auction Location", ['Scottsdale', 'Palm Beach', 'Las Vegas'])

# 4. Process Inputs for the Model
if st.button("Predict Sale Price"):
    # Create a small dataframe with user inputs
    input_df = pd.DataFrame([[year, mileage, condition]], 
                             columns=['Vehicle_Year', 'Mileage', 'Condition_Score'])
    
    # We need to add the "One-Hot Encoded" columns (Make and Location)
    # Start with all zeros for these columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Set the specific Make and Location to 1
    make_col = f"Make_{make}"
    loc_col = f"Auction_Location_{location}"
    
    if make_col in input_df.columns:
        input_df[make_col] = 1
    if loc_col in input_df.columns:
        input_df[loc_col] = 1
    
    # Ensure columns are in the EXACT SAME order as training
    input_df = input_df[model_columns]

    # 5. Make Prediction
    prediction = model.predict(input_df)[0]

    # 6. Show Result
    st.success(f"### Estimated Sale Price: ${prediction:,.2f}")
    st.info("Note: This prediction is based on historical patterns in the AuctionIQ dataset.")
