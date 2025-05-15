import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


# pipe = pickle.load(open('logistic_regression.pkl', 'rb'))
pipe = pickle.load(open('random_forest.pkl', 'rb'))

st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1: 
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2: 
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select the host city', sorted(cities))

target = st.number_input('Targeted score', step=1)

col3, col4, col5 = st.columns(3)

with col3: 
    current_score = st.number_input('Current score', step=1,min_value=0, max_value=target)
with col4: 
    overs = st.number_input('Overs completed', min_value=1, max_value=19)
with col5: 
    wickets = st.number_input('Players dismissed (wickets out)', min_value=0, max_value=9)

if st.button('Predict Probability'): 
    runs_left = target - current_score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = current_score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city], 
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets_left':[wickets],
        'total_runs_x':[target],
        'crr':[crr],
        'rrr':[rrr]
    })

    result = pipe.predict_proba(input_df)

    win = result[0][0]
    lose = result[0][1]

    # st.write(lose)
    # st.write(win)

    st.header(f"{batting_team} - {str(round(win * 100))} %")
    st.header(f"{bowling_team} - {str(round(lose * 100))} %")

    # Optional: show pie chart
    fig, ax = plt.subplots()
    ax.pie([win, lose], labels=[batting_team, bowling_team], autopct='%0.2f%%', colors=['#1f77b4', '#ff7f0e'])
    st.pyplot(fig)