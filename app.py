import streamlit as st
import pandas as pd
import pickle
import sportsdataverse.wnba as wnba_api # <-- THE ALIAS IMPORT FIX
from datetime import datetime

# ==========================================
# WNBA LIVE Roster Prop Predictor
# ==========================================

st.set_page_config(page_title="WNBA Live AI Predictor", page_icon="🏀", layout="centered")

@st.cache_data
def load_historical_data(filepath):
    try:
        data = pd.read_csv(filepath)
        data['athlete_id_1'] = data['athlete_id_1'].astype(int)
        return data
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    try:
        with open('model_points.pkl', 'rb') as f: model_pts = pickle.load(f)
        with open('model_rebounds.pkl', 'rb') as f: model_reb = pickle.load(f)
        with open('model_assists.pkl', 'rb') as f: model_ast = pickle.load(f)
        return model_pts, model_reb, model_ast
    except FileNotFoundError:
        return None, None, None

@st.cache_data(ttl=600)
def get_todays_games(today_str):
    try:
        # --- CALLING THE FUNCTION WITH THE ALIAS ---
        return wnba_api.wnba_schedule(dates=today_str)
    except Exception as e:
        st.error(f"Could not fetch today's schedule. Error: {e}")
        return pd.DataFrame()

st.title('🤖 WNBA AI Prop Predictor (Live Rosters)')

hist_df = load_historical_data('wnba_data_for_app.csv')
model_pts, model_reb, model_ast = load_models()

if hist_df is None or model_pts is None:
    st.error("CRITICAL ERROR: Make sure necessary files (`.csv`, `.pkl`) are in the app folder.")
else:
    today_str = datetime.now().strftime("%Y%m%d")
    todays_schedule = get_todays_games(today_str)

    if todays_schedule.empty:
        st.info(f"🏀 No WNBA games found scheduled for today, {datetime.now().strftime('%B %d, %Y')}.")
        st.info("Check back on a game day to see the live predictions!")
    else:
        st.header("Step 1: Select Today's Game")
        game_options = {game['shortName']: game for game in todays_schedule['games'] if 'shortName' in game}

        if not game_options:
            st.warning("Games scheduled, but data feed may be missing details. Check back later.")
        else:
            selected_game_str = st.selectbox("Choose a game:", list(game_options.keys()))
            selected_game_data = game_options[selected_game_str]
            teams = selected_game_data.get('competitions', [{}])[0].get('competitors', [])
            live_player_map = {}
            for team in teams:
                for player in team.get('roster', []):
                    if 'id' in player and 'displayName' in player:
                        live_player_map[int(player['id'])] = player['displayName']

            if not live_player_map:
                st.warning(f"Rosters for {selected_game_str} are not yet available. Check back closer to game time.")
            else:
                st.header("Step 2: Select a Player")
                sorted_player_names = sorted(live_player_map.values())
                selected_player_name = st.selectbox("Choose a player from the roster:", sorted_player_names)
                if selected_player_name:
                    player_id = [pid for pid, name in live_player_map.items() if name == selected_player_name][0]
                    try:
                        player_latest_stats = hist_df[hist_df['athlete_id_1'] == player_id].sort_values(by='game_date').iloc[-1]
                        feature_order = model_pts.get_booster().feature_names
                        features_for_prediction = pd.DataFrame([player_latest_stats[feature_order]])
                        st.header(f"Next Game Projection for: {selected_player_name}")
                        predicted_pts = model_pts.predict(features_for_prediction)[0]
                        predicted_reb = model_reb.predict(features_for_prediction)[0]
                        predicted_ast = model_ast.predict(features_for_prediction)[0]
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Projected Points", f"{predicted_pts:.1f}")
                        col2.metric("Projected Rebounds", f"{predicted_reb:.1f}")
                        col3.metric("Projected Assists", f"{predicted_ast:.1f}")
                        st.metric("Projected PRA", f"{(predicted_pts + predicted_reb + predicted_ast):.1f}")
                    except IndexError:
                        st.warning(f"No historical data found for {selected_player_name}.")
                    except Exception as e:
                        st.error(f"An error during prediction: {e}")