#General Data (player level data is filtered from generalized pitcher or hitter data)


from baseball_utils import *
# --- Download and process batter-side data ---
batter_ogdata = statcast(start_dt="2024-03-28", end_dt="2025-06-30")


batter_data = batter_ogdata.copy()
batter_data['batter'] = pd.to_numeric(batter_data['batter'], errors='coerce')
batter_data['pitcher'] = pd.to_numeric(batter_data['pitcher'], errors='coerce')

ids = pd.unique(batter_data[['batter', 'pitcher']].values.ravel())
id_to_name = playerid_reverse_lookup(ids, key_type='mlbam')
id_to_name['full_name'] = id_to_name['name_first'] + ' ' + id_to_name['name_last']
id_to_name = id_to_name[['key_mlbam', 'full_name']]

# Merge names
batter_data = batter_data.merge(id_to_name, how='left', left_on='pitcher', right_on='key_mlbam')
batter_data.rename(columns={'full_name': 'pitcher_name'}, inplace=True)
batter_data.drop(columns='key_mlbam', inplace=True)

batter_data = batter_data.merge(id_to_name, how='left', left_on='batter', right_on='key_mlbam')
batter_data.rename(columns={'full_name': 'batter_name'}, inplace=True)
batter_data.drop(columns='key_mlbam', inplace=True)

# Add pitch_group
pitch_group_map = {
    'FF': '4F', 'FT': '2F', 'SI': '2F', 'FA': 'CF', 'FC': 'CF',
    'SL': 'S', 'ST': 'S', 'CS': 'C', 'CU': 'C', 'KC': 'C',
    'CH': 'CH', 'FS': 'CH'
}
batter_data['pitch_group'] = batter_data['pitch_type'].map(pitch_group_map)

# --- Save batter data ---
batter_data.to_pickle("batter_data.pkl")


# --- Download and process pitcher-side data ---
pitcher_ogdata = statcast(start_dt="2024-03-28", end_dt="2025-06-30")

pitcher_data = pitcher_ogdata.copy()
pitcher_data['batter'] = pd.to_numeric(pitcher_data['batter'], errors='coerce')
pitcher_data['pitcher'] = pd.to_numeric(pitcher_data['pitcher'], errors='coerce')

ids = pd.unique(pitcher_data[['batter', 'pitcher']].values.ravel())
id_to_name = playerid_reverse_lookup(ids, key_type='mlbam')
id_to_name['full_name'] = id_to_name['name_first'] + ' ' + id_to_name['name_last']
id_to_name = id_to_name[['key_mlbam', 'full_name']]

pitcher_data = pitcher_data.merge(id_to_name, how='left', left_on='pitcher', right_on='key_mlbam')
pitcher_data.rename(columns={'full_name': 'pitcher_name'}, inplace=True)
pitcher_data.drop(columns='key_mlbam', inplace=True)

pitcher_data = pitcher_data.merge(id_to_name, how='left', left_on='batter', right_on='key_mlbam')
pitcher_data.rename(columns={'full_name': 'batter_name'}, inplace=True)
pitcher_data.drop(columns='key_mlbam', inplace=True)

# --- Save pitcher data ---
pitcher_data.to_pickle("pitcher_data.pkl")



#Batting Data for Hitter Archetype
# Load FanGraphs batting stats (2025 regular season)
batting = batting_stats(2025, qual=0)  # qual=0 removes the PA filter

#Encoder data
encoder_data = statcast(start_dt= "2025-06-01", end_dt = "2025-06-14")

#Find McNeil in the lineup data
bat_stats_2025 = batting_stats(2025, qual=False)

# Save Batting Data for Hitter Archetype
batting.to_pickle("batting_2025.pkl")

# Save Encoder Data (Statcast)
encoder_data.to_pickle("encoder_data.pkl")

# Save McNeil Lineup Data
bat_stats_2025.to_pickle("bat_stats_2025.pkl")