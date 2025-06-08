import streamlit as st
import pandas as pd
import pickle
import requests
from datetime import datetime

# ===================================================================
# WNBA AI Predictor (Final Version - Live ESPN Rosters)
# ===================================================================

# --- Page Configuration ---
st.set_page_config(page_title="WNBA Live AI Predictor", page_icon="🏀")

# --- Data Loading and Caching ---
@st.cache_data
def load_historical_data(filepath):
    """Loads the main historical game stats dataset."""
    try:
        data = pd.read_csv(filepath)
        data['athlete_id_1'] = data['athlete_id_1'].astype(int)
        return data
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    """Loads the pre-trained prediction models."""
    try:
        with open('model_points.pkl', 'rb') as f: model_pts = pickle.load(f)
        with open('model_rebounds.pkl', 'rb') as f: model_reb = pickle.load(f)
        with open('model_assists.pkl', 'rb') as f: model_ast = pickle.load(f)
        return model_pts, model_reb, model_ast
    except FileNotFoundError:
        return None, None, None

@st.cache_data(ttl=86400) # Cache the roster for a full day (24 * 60 * 60 seconds)
def get_all_players_from_espn():
    """
    Fetches every team's roster to build a complete, live map of player IDs to names.
    This is the most reliable way to get a full, current list of all players.
    """
    st.write("Fetching live WNBA roster data for the season...")
    player_map = {}
    try:
        teams_url = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams"
        teams_response = requests.get(teams_url, timeout=10)
        teams_response.raise_for_status()
        teams_data = teams_response.json().get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', [])

        for team_entry in teams_data:
            team_id = team_entry['team']['id']
            roster_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams/{team_id}?enable=roster"
            roster_response = requests.get(roster_url, timeout=10)
            roster_response.raise_for_status()
            roster_data = roster_response.json().get('team', {}).get('athletes', [])
            
            for player in roster_data:
                player_map[int(player['id'])] = player['fullName']
        
        st.success("Live roster data successfully loaded!")
        return player_map
    except Exception as e:
        st.error(f"Could not fetch live WNBA rosters. Error: {e}", icon="📡")
        return None

# --- Main Application Logic ---
st.title('🤖 WNBA AI Prop Predictor')
st.info("Select a player to project their stats, using a live-updating player list.")

# Load all assets
hist_df = load_historical_data('wnba_data_for_app.csv')
model_pts, model_reb, model_ast = load_models()
player_map = get_all_players_from_espn()

if not all([hist_df is not None, player_map is not None, model_pts is not None]):
    st.error(
        "A critical asset could not be loaded. This could be a missing file in the repository "
        "(`wnba_data_for_app.csv` or model `.pkl` files) or an issue fetching live roster data from ESPN."
    )
else:
    # --- UI Elements ---
    st.sidebar.header("Select a Player")
    
    # The dropdown menu is now populated with EVERY player in the league
    all_player_names = sorted(player_map.values())
    
    selected_player_name = st.sidebar.selectbox(
        'Choose a player:',
        all_player_names,
        index=0  # Default to the first player alphabetically
    )

    if st.sidebar.button(f'Predict Stats for {selected_player_name}', type="primary"):
        try:
            # Find the player's ID from our live player map
            player_id = [pid for pid, name in player_map.items() if name == selected_player_name][0]

            # Find the most recent stats for this player from our historical dataset
            player_latest_stats = hist_df[hist_df['athlete_id_1'] == player_id].sort_values(by='game_date').iloc[-1]
            
            # Ensure the feature order is correct before predicting
            feature_order = model_pts.get_booster().feature_names
            features_for_prediction = pd.DataFrame([player_latest_stats[feature_order]])

            st.header(f"Projection for: {selected_player_name}")

            # Make the predictions
            predicted_pts = model_pts.predict(features_for_prediction)[0]
            predicted_reb = model_reb.predict(features_for_prediction)[0]
            predicted_ast = model_ast.predict(features_for_prediction)[0]

            # Display the results
            col1, col2, col3 = st.columns(3)
            col1.metric("Projected Points", f"{predicted_pts:.1f}")
            col2.metric("Projected Rebounds", f"{predicted_reb:.1f}")
            col3.metric("Projected Assists", f"{predicted_ast:.1f}")

        except IndexError:
            st.error(f"'{selected_player_name}' is a current roster player, but no historical game data was found for them in our dataset. This is common for rookies or players who recently joined the league.")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
