import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="WNBA AI Predictor", page_icon="🏀")

@st.cache_data
def load_data(filepath):
    try:
        return pd.read_csv(filepath)
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

st.title('🤖 WNBA Daily-Updated AI Predictor')

df = load_data('wnba_data_for_app.csv')
model_pts, model_reb, model_ast = load_models()

if df is None or model_pts is None:
    st.error("ERROR: Core data or model files not found in repository.")
else:
    df['athlete_id_1'] = df['athlete_id_1'].astype(int)
    
    # We can no longer assume hard-coded names. Let's create the player list from the data.
    # Note: Our historical PBP data does not contain names, only IDs.
    # The most robust solution is to select by ID.
    all_player_ids = sorted(df['athlete_id_1'].unique())

    st.sidebar.header("Select a Player by ID")
    selected_player_id = st.sidebar.selectbox("Choose a player ID:", all_player_ids)

    if st.sidebar.button("Predict Stats", type="primary"):
        try:
            player_latest_stats = df[df['athlete_id_1'] == selected_player_id].sort_values(by='game_date').iloc[-1]
            feature_order = model_pts.get_booster().feature_names
            features_for_prediction = pd.DataFrame([player_latest_stats[feature_order]])
            
            st.header(f"Projection for Player ID: {selected_player_id}")
            predicted_pts = model_pts.predict(features_for_prediction)[0]
            predicted_reb = model_reb.predict(features_for_prediction)[0]
            predicted_ast = model_ast.predict(features_for_prediction)[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Projected Points", f"{predicted_pts:.1f}")
            col2.metric("Projected Rebounds", f"{predicted_reb:.1f}")
            col3.metric("Projected Assists", f"{predicted_ast:.1f}")

        except IndexError:
            st.error(f"No historical data found for Player ID: {selected_player_id}")
        except Exception as e:
            st.error(f"An error occurred: {e}")