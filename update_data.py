# update_data.py
import pandas as pd
import requests
from datetime import datetime, timedelta

def get_game_data_for_date(date_str):
    """Fetches all game data for a specific date from ESPN's API."""
    try:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates={date_str}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching data for {date_str}: {e}")
        return None

def parse_espn_data(raw_data):
    """Parses raw JSON from ESPN into a clean list of player game logs."""
    game_logs = []
    games = raw_data.get('events', [])
    for game in games:
        competition = game.get('competitions', [{}])[0]
        game_date = competition.get('date')
        
        for competitor in competition.get('competitors', []):
            team_id = competitor.get('id')
            # The player stats are nested deep inside
            for athlete in competitor.get('roster', []):
                stats = athlete.get('statistics', [{}])[0].get('stats', [])
                if len(stats) >= 3: # Need at least Pts, Reb, Ast
                    game_logs.append({
                        'game_id': game.get('id'),
                        'athlete_id_1': int(athlete.get('id')),
                        'season': raw_data.get('season', {}).get('year'),
                        'game_date': game_date,
                        'points': float(stats[0]), # Assuming order: Pts, Reb, Ast
                        'rebounds': float(stats[1]),
                        'assists': float(stats[2])
                    })
    return game_logs

def update_features(df):
    """Calculates rolling average features on the entire dataset."""
    df = df.sort_values(by=['athlete_id_1', 'game_date'])
    for window in [3, 5, 10]:
        for stat in ['points', 'rebounds', 'assists']:
            df[f'avg_{stat}_last_{window}'] = df.groupby('athlete_id_1')[stat].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
    return df

if __name__ == "__main__":
    # --- 1. Load existing data ---
    try:
        historical_df = pd.read_csv("wnba_data_for_app.csv")
    except FileNotFoundError:
        print("Data file not found. Please ensure wnba_data_for_app.csv exists.")
        exit()

    # --- 2. Fetch yesterday's data ---
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y%m%d')
    print(f"Fetching game data for {yesterday_str}...")
    
    new_data_raw = get_game_data_for_date(yesterday_str)
    
    if new_data_raw and new_data_raw.get('events'):
        # --- 3. Parse and combine data ---
        new_game_logs = parse_espn_data(new_data_raw)
        if new_game_logs:
            new_logs_df = pd.DataFrame(new_game_logs)
            print(f"Found {len(new_logs_df)} new player performances.")
            
            # Combine old and new, dropping duplicates to be safe
            combined_df = pd.concat([historical_df, new_logs_df]).drop_duplicates(
                subset=['game_id', 'athlete_id_1'], keep='last'
            )
            
            # --- 4. Recalculate features and save ---
            print("Recalculating features on updated dataset...")
            updated_df = update_features(combined_df)
            updated_df.dropna(inplace=True) # Remove rows where features couldn't be calculated
            
            updated_df.to_csv("wnba_data_for_app.csv", index=False)
            print("Successfully updated 'wnba_data_for_app.csv' with new game data.")
        else:
            print("No new player performances to add.")
    else:
        print("No games found for yesterday. Data file remains unchanged.")