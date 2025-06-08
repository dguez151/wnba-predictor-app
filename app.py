import streamlit as st
import pandas as pd
import pickle

# ==============================================================
# WNBA AI Prop Predictor (Stable Version - No Live Data)
# ==============================================================

# --- Page Configuration ---
st.set_page_config(page_title="WNBA AI Predictor", page_icon="🏀")

# --- Caching Functions ---
@st.cache_data
def load_historical_data(filepath):
    """Loads the historical dataset from the CSV."""
    try:
        data = pd.read_csv(filepath)
        data['athlete_id_1'] = data['athlete_id_1'].astype(int)
        return data
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    """Loads the pre-trained models."""
    try:
        with open('model_points.pkl', 'rb') as f: model_pts = pickle.load(f)
        with open('model_rebounds.pkl', 'rb') as f: model_reb = pickle.load(f)
        with open('model_assists.pkl', 'rb') as f: model_ast = pickle.load(f)
        return model_pts, model_reb, model_ast
    except FileNotFoundError:
        return None, None, None

# --- Main App Logic ---
st.title('🤖 WNBA AI Prop Predictor')
st.info("This app uses historical data to project a player's stats for their next game.")

# Load assets
hist_df = load_historical_data('wnba_data_for_app.csv')
model_pts, model_reb, model_ast = load_models()

if hist_df is None or model_pts is None:
    st.error("CRITICAL ERROR: Make sure all files (.csv, .pkl) are uploaded to your GitHub repository.")
else:
    # --- UI Elements ---
    st.sidebar.header("Select a Player")

    # This dictionary now uses the REAL IDs you found.
    # This is the key to making the app work.
    player_id_to_name = {
        618: "Player ID: 618",
        651: "Player ID: 651",
        689: "Player ID: 689",
        732: "Player ID: 732",
        764: "Player ID: 764",
        766: "Player ID: 766"
    }
    
    # Create a 'player_name' column for the dropdown using our map
    hist_df['player_name'] = hist_df['athlete_id_1'].map(player_id_to_name)
    
    # Get a list of players we have names for from the map
    available_players = sorted(hist_df[hist_df['player_name'].notna()]['player_name'].unique())
    
    if not available_players:
        st.sidebar.error("The player IDs in the code do not match the IDs in the data file.")
    else:
        selected_player_name = st.sidebar.selectbox('Choose a player:', available_players)
        
        st.sidebar.markdown("---")
        st.sidebar.success("The AI is ready!")

        if selected_player_name:
            player_id = [pid for pid, name in player_id_to_name.items() if name == selected_player_name][0]
            
            try:
                # Find the player's most recent game stats
                player_latest_stats = hist_df[hist_df['athlete_id_1'] == player_id].sort_values(by='game_date').iloc[-1]
                
                # Ensure the feature order is correct
                feature_order = model_pts.get_booster().feature_names
                features_for_prediction = pd.DataFrame([player_latest_stats[feature_order]])

                st.header(f"Projection for: {selected_player_name}")

                # Predict stats
                predicted_pts = model_pts.predict(features_for_prediction)[0]
                predicted_reb = model_reb.predict(features_for_prediction)[0]
                predicted_ast = model_ast.predict(features_for_prediction)[0]

                # Display stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Projected Points", f"{predicted_pts:.1f}")
                col2.metric("Projected Rebounds", f"{predicted_reb:.1f}")
                col3.metric("Projected Assists", f"{predicted_ast:.1f}")
                
            except IndexError:
                st.error(f"Could not find historical data for {selected_player_name}.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
