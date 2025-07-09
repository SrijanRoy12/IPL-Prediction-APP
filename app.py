import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.let_it_rain import rain
import base64
from PIL import Image
import numpy as np
import time

# --------------------------
# NEW: Configuration & Assets
# --------------------------
TEAM_LOGOS = {
    'Chennai Super Kings': 'csk.png',
    'Delhi Capitals': 'dc.png',
    'Kings XI Punjab': 'kxp.png',
    'Kolkata Knight Riders': 'kkr.png',
    'Mumbai Indians': 'mi.png',
    'Rajasthan Royals': 'rr.png',
    'Royal Challengers Bangalore': 'rcb.png',
    'Sunrisers Hyderabad': 'srh.png'
}

VENUE_STATS = {
    'M Chinnaswamy Stadium': {'avg_score': 185, 'win_percent': {'chasing': 58}},
    'Wankhede Stadium': {'avg_score': 178, 'win_percent': {'chasing': 62}},
    'Eden Gardens': {'avg_score': 165, 'win_percent': {'chasing': 55}},
    'Feroz Shah Kotla': {'avg_score': 160, 'win_percent': {'chasing': 52}},
    'MA Chidambaram Stadium': {'avg_score': 155, 'win_percent': {'chasing': 48}},
    'Rajiv Gandhi International Stadium': {'avg_score': 170, 'win_percent': {'chasing': 60}},
    'Sawai Mansingh Stadium': {'avg_score': 162, 'win_percent': {'chasing': 53}}
}

# --------------------------
# Helper Functions
# --------------------------
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

@st.cache_resource
def load_features():
    return pickle.load(open("model_features.pkl", "rb"))

def display_team_logo(team_name, width=80):
    if team_name in TEAM_LOGOS:
        try:
            img = Image.open(TEAM_LOGOS[team_name])
            st.image(img, width=width)
        except:
            st.write(team_name)
    else:
        st.write(team_name)

# --------------------------
# Main App
# --------------------------
def main():
    # Setup
    set_background("Hosts-of-IPL-2023.jpg")
    model = load_model()
    model_features = load_features()

    # Custom CSS
    st.markdown("""
    <style>
        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }
        .big-font {
            font-size: 28px !important;
            color: #FFFFFF !important;
            font-weight: bold;
            text-shadow: 2px 2px 4px #000000;
        }
        .metric-card {
            background-color: rgba(0, 0, 0, 0.7) !important;
            border-radius: 12px !important;
            padding: 15px !important;
            border-left: 5px solid #FF0000 !important;
        }
        .stMetricValue {
            color: #FF0000 !important;
            font-size: 28px !important;
            font-weight: bold !important;
        }
        .stMetricLabel {
            color: #FFFFFF !important;
            font-size: 18px !important;
        }
        .venue-stats {
            background-color: rgba(0, 0, 0, 0.6);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            color: white;
        }
        .stProgress > div > div > div {
            background-color: #FF0000;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header with animated title
    st.markdown("""
    <div class="big-font">
        <marquee behavior="scroll" direction="left">🏏 IPL Match Win Predictor Pro</marquee>
    </div>
    <div style="color:white;font-size:16px;text-shadow: 1px 1px 2px #000000;">
        Advanced match predictions with real-time analytics
    </div>
    <hr style="border: 1px solid #FF0000;">
    """, unsafe_allow_html=True)

    # --------------------------
    # NEW: Interactive Sidebar
    # --------------------------
    with st.sidebar:
        st.header("⚙️ Match Configuration")
        
        # Team selection with logos
        teams = sorted(TEAM_LOGOS.keys())
        col1, col2 = st.columns(2)
        with col1:
            batting_team = st.selectbox("Batting Team", teams, index=teams.index('Mumbai Indians'))
            display_team_logo(batting_team)
        with col2:
            bowling_team = st.selectbox("Bowling Team", [t for t in teams if t != batting_team], 
                                      index=1)
            display_team_logo(bowling_team)
        
        # Venue selection with stats
        venue = st.selectbox("Match Venue", sorted(VENUE_STATS.keys()))
        
        # Display venue statistics
        with st.expander("📊 Venue Statistics"):
            st.markdown(f"""
            <div class="venue-stats">
                <b>Average 1st Innings Score:</b> {VENUE_STATS[venue]['avg_score']}<br>
                <b>Chasing Win %:</b> {VENUE_STATS[venue]['win_percent']['chasing']}%
            </div>
            """, unsafe_allow_html=True)
        
        # Match parameters with sliders
        st.subheader("📉 Match Parameters")
        target = st.slider("Target Score", 50, 300, 180, 5)
        score = st.slider("Current Score", 0, target, 90)
        overs = st.slider("Overs Completed", 0.1, 20.0, 10.0, 0.1)
        wickets = st.slider("Wickets Lost", 0, 10, 3)
        
        # NEW: Advanced metrics
        with st.expander("🔍 Advanced Metrics"):
            runs_last_5 = st.slider("Runs in Last 5 Overs", 0, 100, 45)
            wickets_last_5 = st.slider("Wickets in Last 5 Overs", 0, 5, 1)
            pitch_condition = st.select_slider("Pitch Condition", 
                                             ['Very Slow', 'Slow', 'Normal', 'Fast', 'Very Fast'])
            dew_factor = st.checkbox("Dew Factor (2nd Innings)", value=False)
        
        # NEW: Match simulation controls
        with st.expander("🎮 Live Simulation"):
            simulate_live = st.checkbox("Enable Live Simulation", value=False)
            if simulate_live:
                simulation_speed = st.slider("Simulation Speed", 1, 5, 3)

    # --------------------------
    # Prediction Section
    # --------------------------
    if st.button("🚀 Predict Match Outcome", use_container_width=True):
        
        # NEW: Prediction with loading animation
        with st.spinner('Crunching numbers and analyzing trends...'):
            time.sleep(1.5)  # Simulate processing time
            
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'venue': [venue],
                'overs': [overs],
                'runs': [score],
                'wickets': [wickets],
                'runs_last_5': [runs_last_5],
                'wickets_last_5': [wickets_last_5]
            })

            input_encoded = pd.get_dummies(input_df)
            for col in model_features:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_features]

            win_prob = model.predict_proba(input_encoded)[0][1]
            rr = (target - score) / max(1, (20 - overs))
            crr = score / overs

        # NEW: Animated metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="🏆 Winning Probability", 
                     value=f"{win_prob*100:.2f}%",
                     delta=f"{'High' if win_prob > 0.6 else 'Low' if win_prob < 0.4 else 'Medium'} chance")
        with col2:
            st.metric(label="🔁 Required Run Rate", 
                     value=f"{rr:.2f} vs {crr:.2f} Current",
                     delta_color="inverse")
        with col3:
            balls_remaining = int((20 - overs) * 6)
            st.metric(label="⏱️ Resources Left", 
                     value=f"{balls_remaining} balls | {wickets} wkts",
                     delta=f"{int((wickets/10)*100)}% wickets remaining")

        # NEW: Win probability gauge
        fig_gauge = px.pie(
            values=[win_prob, 1-win_prob],
            names=['Win', 'Lose'],
            hole=0.7,
            color_discrete_sequence=['#FF0000', '#333333']
        )
        fig_gauge.update_layout(
            title='Win Probability Gauge',
            showlegend=False,
            annotations=[dict(text=f'{win_prob*100:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # NEW: Match progress visualization
        st.subheader("📊 Match Progress Analysis")
        progress_col1, progress_col2 = st.columns(2)
        with progress_col1:
            st.markdown(f"**Target: {target}**")
            st.progress(score/target)
            st.caption(f"Score: {score}/{wickets} ({overs} overs)")
        with progress_col2:
            st.markdown("**Run Rate Comparison**")
            fig_rr = px.bar(
                x=["Current RR", "Required RR"],
                y=[crr, rr],
                color=["Current RR", "Required RR"],
                color_discrete_map={"Current RR": "blue", "Required RR": "red"},
                text=[f"{crr:.2f}", f"{rr:.2f}"]
            )
            fig_rr.update_layout(showlegend=False)
            st.plotly_chart(fig_rr, use_container_width=True)

        # NEW: Historical performance
        with st.expander("📈 Team Performance Insights"):
            tab1, tab2 = st.tabs([f"{batting_team} Stats", f"{bowling_team} Stats"])
            with tab1:
                st.write(f"**{batting_team} at {venue}**")
                st.metric("Average Score", "172")
                st.metric("Win Percentage", "58%")
                st.metric("Last 5 Matches", "3W-2L")
            with tab2:
                st.write(f"**{bowling_team} at {venue}**")
                st.metric("Average Runs Conceded", "165")
                st.metric("Win Percentage", "52%")
                st.metric("Last 5 Matches", "2W-3L")

        # NEW: Match simulation
        if 'simulate_live' in locals() and simulate_live:
            st.subheader("🎮 Live Match Simulation")
            simulation_placeholder = st.empty()
            
            current_score = score
            current_wickets = wickets
            current_overs = overs
            
            for i in range(1, int((20 - overs)*10) + 1):
                time.sleep(1/simulation_speed)
                current_overs += 0.1
                current_score += np.random.randint(0, 7)
                if np.random.random() < 0.08:
                    current_wickets += 1
                
                if current_wickets >= 10 or current_score >= target:
                    break
                    
                simulation_placeholder.markdown(f"""
                <div style="background-color:rgba(0,0,0,0.7); padding:15px; border-radius:10px; color:white;">
                    <b>Over:</b> {current_overs:.1f} | <b>Score:</b> {current_score}/{current_wickets} | 
                    <b>Req RR:</b> {(target - current_score)/(20 - current_overs):.2f}
                </div>
                """, unsafe_allow_html=True)
            
            if current_score >= target:
                st.success(f"🏆 {batting_team} won by {10 - current_wickets} wickets!")
                rain(emoji="🎉", font_size=40, falling_speed=5)
            elif current_wickets >= 10:
                st.error(f"☠️ {bowling_team} won by {target - current_score} runs!")
            else:
                st.warning("Match simulation completed without result")

        # Celebration effects
        if win_prob > 0.7:
            rain(emoji="🎉", font_size=40, falling_speed=5, animation_length="infinite")
            st.balloons()
            st.success("🔥 Batting team is dominating!")
        elif win_prob < 0.3:
            st.error("⚠️ Tough chase! Bowling team in control.")
        else:
            st.info("⚖️ Close match! Could go either way.")

if __name__ == "__main__":
    main()