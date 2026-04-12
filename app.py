import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Load model and scaler
DATA_DIR = Path('data')

@st.cache_resource
def load_model():
    with open(DATA_DIR / 'final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(DATA_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(DATA_DIR / 'feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, feature_cols

model, scaler, feature_cols = load_model()

# App layout
st.title('NBA Salary Predictor')
st.markdown(
    'Predict what an NBA player should earn based on their '
    '2024-25 per 100 possession stats.'
)
st.markdown('---')

st.subheader('Player stats — per 100 possessions')

col1, col2, col3 = st.columns(3)

with col1:
    pts  = st.number_input('Points (PTS)',        0.0, 80.0, 20.0, 0.1)
    ast  = st.number_input('Assists (AST)',        0.0, 30.0, 5.0,  0.1)
    orb  = st.number_input('Off rebounds (ORB)',   0.0, 20.0, 2.0,  0.1)
    drb  = st.number_input('Def rebounds (DRB)',   0.0, 30.0, 4.0,  0.1)

with col2:
    stl  = st.number_input('Steals (STL)',          0.0, 10.0, 1.0,  0.1)
    blk  = st.number_input('Blocks (BLK)',          0.0, 10.0, 0.5,  0.1)
    tov  = st.number_input('Turnovers (TOV)',        0.0, 15.0, 2.5,  0.1)
    pf   = st.number_input('Personal fouls (PF)',   0.0, 10.0, 2.5,  0.1)

with col3:
    age  = st.number_input('Age',                   18,  45,   26)
    g    = st.number_input('Games played (G)',       1,   82,   60)
    drtg = st.number_input('Defensive rating',       80,  130,  112)
    efg  = st.number_input('eFG%',                  0.0, 1.0,  0.52, 0.01)

st.markdown('---')
col4, col5 = st.columns(2)

with col4:
    p3   = st.number_input('3P%',  0.0, 1.0, 0.35, 0.01)
    p2   = st.number_input('2P%',  0.0, 1.0, 0.50, 0.01)
    ft   = st.number_input('FT%',  0.0, 1.0, 0.75, 0.01)

with col5:
    pos = st.selectbox('Position', ['C', 'PF', 'PG', 'SF', 'SG'])

st.markdown('---')

if st.button('Predict salary'):
    # Build input dataframe
    input_data = {
        'Age':    age,
        'G':      g,
        '3P%':    p3,
        '2P%':    p2,
        'eFG%':   efg,
        'FT%':    ft,
        'ORB':    orb,
        'DRB':    drb,
        'AST':    ast,
        'STL':    stl,
        'BLK':    blk,
        'TOV':    tov,
        'PF':     pf,
        'PTS':    pts,
        'DRtg':   drtg,
        'Pos_PF': 1 if pos == 'PF' else 0,
        'Pos_PG': 1 if pos == 'PG' else 0,
        'Pos_SF': 1 if pos == 'SF' else 0,
        'Pos_SG': 1 if pos == 'SG' else 0,
    }

    input_df = pd.DataFrame([input_data])[feature_cols]
    input_scaled = scaler.transform(input_df)
    log_pred = model.predict(input_scaled)[0]
    salary_pred = np.exp(log_pred)

    st.success(f'Predicted salary: **${salary_pred:,.0f}**')
    st.caption(
        f'${salary_pred/1e6:.1f}M per year — '
        f'model RMSE is $8.7M so actual salary likely falls '
        f'between ${max(0, (salary_pred-8.7e6))/1e6:.1f}M '
        f'and ${(salary_pred+8.7e6)/1e6:.1f}M'
    )

st.markdown('---')
st.caption(
    'Model: Lasso Regression | '
    'Stats: Basketball Reference 2024-25 per 100 possessions | '
    'Salaries: 2025-26 contracts | '
    'Test R²: 0.544 | Test RMSE: $8.7M'
)