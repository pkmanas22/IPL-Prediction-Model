import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

teams = ['Chennai Super Kings',
 'Royal Challengers Bengaluru',
 'Delhi Capitals',
 'Kolkata Knight Riders',
 'Mumbai Indians',
 'Punjab Kings',
 'Rajasthan Royals',
 'Sunrisers Hyderabad',
 'Lucknow Super Giants',
 'Gujarat Titans']

cities = ['Bangalore', 'Chandigarh', 'Delhi', 'Mumbai', 'Kolkata', 'Jaipur',
       'Hyderabad', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban',
       'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Bengaluru', 'Indore', 'Dubai', 'Sharjah', 'Navi Mumbai',
       'Lucknow', 'Guwahati', 'Mohali']


lr_pipe = joblib.load(open('models/logistic_regression.pkl', 'rb'))
rf_pipe = joblib.load(open('models/random_forest.pkl', 'rb'))

def handle_submission (target, current_score, overs, wickets, batting_team, bowling_team, selected_city, model_pipe, model_name) : 
    runs_left = target - current_score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city], 
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],  # FIX: Use wickets_left instead of wickets
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })
    
    result = model_pipe.predict_proba(input_df)
    print(f"Result: {result}")
    
    lose = result[0][0]  # Probability of losing (batting team loses)
    win = result[0][1]   # Probability of winning (batting team wins)
    
    st.subheader(f"{model_name} Prediction")
    st.write(f"{batting_team} - {str(round(win * 100))} %")
    st.write(f"{bowling_team} - {str(round(lose * 100))} %")
    
    # Optional: show pie chart
    fig, ax = plt.subplots()
    ax.pie([win, lose], labels=[batting_team, bowling_team], autopct='%0.2f%%', colors=['#1f77b4', '#ff7f0e'])
    st.pyplot(fig)


st.set_page_config(page_title="IPL Win Predictor")

st.title('IPL Win Predictor 🏏')

st.markdown(
    """
    <style>
        .st-emotion-cache-1w723zb { 
            padding: 2rem 1rem 0rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        .st-emotion-cache-1w723zb { 
            padding: 2rem 1rem 0rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)
with col1: 
    batting_team = st.selectbox('Select the batting team (chasing)', sorted(teams))
with col2: 
    bowling_team = st.selectbox('Select the bowling team (defending)', sorted(teams))

selected_city = st.selectbox('Select the host city', sorted(cities))
target = st.number_input('Targeted score', step=1)

col3, col4, col5 = st.columns(3)
with col3: 
    current_score = st.number_input('Current score', step=1, min_value=0, max_value=target)
with col4: 
    overs = st.number_input('Overs completed', min_value=1, max_value=19)
with col5: 
    wickets = st.number_input('Players dismissed (wickets out)', min_value=0, max_value=9)

submit_btn = st.button('Predict Probability')

# lr, rm = st.columns(2)
# with lr:
#     if submit_btn: 
#         handle_submission(target, current_score, overs, wickets, batting_team, bowling_team, selected_city, lr_pipe, 'Logistic Regression')
# with rm:
#     if submit_btn: 
#         handle_submission(target, current_score, overs, wickets, batting_team, bowling_team, selected_city, rf_pipe, 'Random Forest')

    
# More accurate - Logistic Regression
if submit_btn: 
    handle_submission(target, current_score, overs, wickets, batting_team, bowling_team, selected_city, lr_pipe, 'Logistic Regression')

st.markdown(
    "<div style='text-align: center; padding: 5px; color: grey;'>"
    "&copy; 2025 <a href='https://devmanas.in' style='text-decoration: none; color: inherit;'>Manas Kumar Pradhan</a>. All rights reserved."
    "</div>",
    unsafe_allow_html=True
)
