from baseball_utils import *
from General_Initialization import *
batter_data = pd.read_pickle("batter_data.pkl")
pitcher_data = pd.read_pickle("pitcher_data.pkl")
batting = pd.read_pickle("batting_2025.pkl")
encoder_data = pd.read_pickle("encoder_data.pkl")
bat_stats_2025 = pd.read_pickle("bat_stats_2025.pkl")




from General_Initialization import (
    encode_count, pitch_group_map, features, elite_woba_threshold,
    power_barrel_threshold, patient_chase_threshold, get_archetype, classify_archetype,
    assign_pitch_cluster, map_description_to_simple, prob_extra_innings,
    OUT_EVENTS, MULTI_OUTS, outs_from_event, pitcher_team, outs_df,
    convert_outs_to_ip, sample_extra_innings, home_IP_extras, away_IP_extras, consolidate_clusters
)


#6
#Fitting should be in a generalized space
#Transforming should take place in a player specific space

#Initialize and fit encoders
#Initialize encoders
arch_encoder = LabelEncoder()
outcome_encoder = LabelEncoder()
pitch_encoder = LabelEncoder()
stand_encoder = LabelEncoder()

#Fit archetype encoder (good)
batting['Hitter_Archetype'] = batting.apply(classify_archetype, axis=1)
arch_encoder.fit(batting['Hitter_Archetype'].fillna('Unknown'))



#Fit outcomes encoder ---- Perhaps this needs to be fit on a generalized dataset like pitcher_data
# Replace your map_description_to_simple with this more permissive version

def map_description_to_simple(desc: str) -> str:
    if desc is None:
        return 'unknown'
    s = str(desc).strip().lower().replace('-', '_').replace('  ', ' ')

    # direct simple labels (your pack already uses these)
    if s in {'ball', 'strike', 'foul', 'hbp', 'bip'}:
        return s

    # Statcast-style / legacy aliases
    BALLS = {'blocked_ball', 'pitchout'}
    STRIKES = {'called_strike', 'swinging_strike', 'swinging_strike_blocked', 'missed_bunt',
               'strike_called', 'strike_looking', 'swing_miss', 'whiff'}
    FOULS = {'foul_tip', 'foul_bunt', 'bunt_foul_tip'}
    if s in BALLS:
        return 'ball'
    if s in STRIKES:
        return 'strike'
    if s in FOULS or s == 'foul':
        return 'foul'
    if s in {'hit_by_pitch', 'hbp'}:
        return 'hbp'
    if s in {'hit_into_play', 'in_play', 'in play', 'in_play_outs', 'in_play_no_out',
             'in play, out(s)', 'in play, no out', 'in play, run(s)', 'contact', 'bip'}:
        return 'bip'

    return 'unknown'
# Step 1: Apply the mapping to a new column
encoder_data['simple_description'] = encoder_data['description'].apply(map_description_to_simple)

# Step 2: Remove rows with 'unknown' or null simplified descriptions
encoder_data = encoder_data[encoder_data['simple_description'].notna()]
encoder_data = encoder_data[encoder_data['simple_description'] != 'unknown']

# Step 3: Replace original 'description' column with simplified version
encoder_data['description'] = encoder_data['simple_description']
encoder_data.drop(columns='simple_description', inplace=True)

# Step 4: Fit the encoder on the cleaned descriptions
outcome_encoder.fit(encoder_data['description'])

#Fit stand encoder (good)
stand_encoder.fit(pitcher_data['stand'])

#Fit pitch type (FF, 2F, CH, etc.) encoder (good)
pitch_encoder.fit(pitcher_data['pitch_type'])




#8
#Encoders should be saved in a general space
#Save encoders, scalers, and data

# Save Fried encoders
os.makedirs("encoders", exist_ok=True)

joblib.dump(stand_encoder, "encoders/stand_encoder.joblib")
joblib.dump(arch_encoder, "encoders/arch_encoder.joblib")
joblib.dump(pitch_encoder, "encoders/pitch_encoder.joblib")
joblib.dump(outcome_encoder, "encoders/outcome_encoder.joblib")

print("All encoders saved to /encoders")