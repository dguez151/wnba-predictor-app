import streamlit as st
import pandas as pd
import pickle
import requests  # <-- We use the requests library to talk to the ESPN API
from datetime import datetime

# ===================================================================
# WNBA AI Prop Predictor (Final Version using Direct ESPN API)
# ===================================================================

# --- Page Configuration ---
st.set_page_config(page_title="WNBA AI Predictor", page_icon="🏀", layout="centered")

# --- Caching Functions ---
@st.cache_data
def load_historical_data(filepath):
    try:
        data = pd.read_csv(filepath)
        if 'athlete_id_1' in data.columns:
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
def get_todays_games_from_espn(today_str):
    """Fetches today's schedule directly from the free ESPN API."""
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={today_str}"
        response = requests.get(url)
        response.raise_for_status()  # Will raise an error for bad status codes
        return response.json() # Returns the data as a dictionary
    except Exception as e:
        st.error(f"Could not fetch today's schedule from ESPN. Error: {e}")
        return {}

# --- Main App Logic ---
st.title('🤖 WNBA AI Prop Predictor (Live Rosters)')

hist_df = load_historical_data('wnba_data_for_app.csv')
model_pts, model_reb, model_ast = load_models()

if hist_df is None or model_pts is None:
    st.error("CRITICAL ERROR: Files not found. Ensure `wnba_data_for_app.csv` and all `.pkl` models are uploaded to your GitHub repository.")
else:
    today_str = datetime.now().strftime("%Y%m%d")
    todays_schedule = get_todays_games_from_espn(today_str)
    
    # ESPN's data is under the 'events' key
    games = todays_schedule.get('events', [])
    
    if not games:
        st.info(f"🏀 No WNBA games found scheduled for today, {datetime.now().strftime('%B %d, %Y')}.")
        st.info("Check back on a game day to see the live predictions!")
    else:
        st.header("Step 1: Select Today's Game")
        game_options = {game['shortName']: game for game in games if 'shortName' in game}
        
        if not game_options:
             st.warning("Games found, but details might be missing. Check back later.")
        else:
            selected_game_str = st.selectbox("Choose a game:", list(game_options.keys()))
            selected_game_data = game_options[selected_game_str]
            
            competitors = selected_game_data.get('competitions', [{}])[0].get('competitors', [])
            
            live_player_map = {}
            for competitor in competitors:
                # In the ESPN API, roster info might be under a 'roster' link, not embedded.
                # For simplicity, we'll work with team IDs and assume players are active.
                # This part is simplified as live roster data is complex to parse from this specific endpoint.
                pass # We will populate players differently now

            # Get the two team names from the selected game
            home_team_name = next((c['team']['displayName'] for c in competitors if c['homeAway'] == 'home'), "Home Team")
            away_team_name = next((c['team']['displayName'] for c in competitors if c['homeAway'] == 'away'), "Away Team")

            # Let's pivot: instead of live rosters, let's use all historical players.
            # This is more robust as live roster data from ESPN's free API is unreliable.
            
            # Use the complete ID-to-Name map you should create from the data
            player_map_df = hist_df[['athlete_id_1', 'player_name']].dropna().drop_duplicates()
            player_map = pd.Series(player_map_df.player_name.values, index=player_map_df.athlete_id_1).to_dict()

            st.header(f"Predicting for game: {away_team_name} @ {home_team_name}")
            st.header("Step 2: Select any Player with Historical Data")
            
            all_player_names = sorted(player_map.values())
            selected_player_name = st.selectbox("Choose a player:", all_player_names)

            if selected_player_name:
                player_id = [pid for pid, name in player_map.items() if name == selected_player_name][0]
                
                try:
                    player_latest_stats = hist_df[hist_df['athlete_id_1'] == player_id].sort_values(by='game_date').iloc[-1]
                    feature_order = model_pts.get_booster().feature_names
                    features_for_prediction = pd.DataFrame([player_latest_stats[feature_order]])
                    
                    st.header(f"Projection for: {selected_player_name}")
                    predicted_pts = model_pts.predict(features_for_prediction)[0]
                    predicted_reb = model_reb.predict(features_for_prediction)[0]
                    predicted_ast = model_ast.predict(features_for_prediction)[0]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Projected Points", f"{predicted_pts:.1f}")
                    col2.metric("Projected Rebounds", f"{predicted_reb:.1f}")
                    col3.metric("Projected Assists", f"{predicted_ast:.1f}")
                except (IndexError, KeyError):
                    st.warning(f"No historical data for {selected_player_name}.")
