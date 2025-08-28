
from baseball_utils import *
batter_data = pd.read_pickle("batter_data.pkl")
pitcher_data = pd.read_pickle("pitcher_data.pkl")
batting = pd.read_pickle("batting_2025.pkl")
encoder_data = pd.read_pickle("encoder_data.pkl")
bat_stats_2025 = pd.read_pickle("bat_stats_2025.pkl")

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
# --- 1. Encode count into single index ---
def encode_count(balls, strikes):
    return {
        (0,0): 0, (0,1): 1, (0,2): 2,
        (1,0): 3, (1,1): 4, (1,2): 5,
        (2,0): 6, (2,1): 7, (2,2): 8,
        (3,0): 9, (3,1): 10, (3,2): 11
    }.get((balls, strikes), 0)

# --- 2. Pitch group mapping ---
pitch_group_map = {
    'FF': '4F', 'FT': '2F', 'SI': '2F',
    'FA': 'CF', 'FC': 'CF',
    'SL': 'S',  'ST': 'S',  'CS': 'S',
    'CU': 'C',  'KC': 'C',  'KN': 'C',
    'CH': 'CH', 'FS': 'CH'
}

# --- 3. GMM features used for clustering ---
features = ['release_speed', 'pfx_x', 'pfx_z']

# Example thresholds (adjust as needed)
elite_woba_threshold = 0.360
power_barrel_threshold = .1
patient_chase_threshold = .27
# Assuming df is your DataFrame with columns: 'wOBA', 'barrel%', 'chase%'

def classify_archetype(row):
    elite = 'Elite' if row['wOBA'] >= elite_woba_threshold else 'Non-Elite'
    power = 'Power' if row['Barrel%'] >= power_barrel_threshold else 'Contact'
    patience = 'Patient' if row['O-Swing% (sc)'] <= patient_chase_threshold else 'Aggressive'
    return f"{elite}-{power}-{patience}"


#Add hitter archetype to fried_data
def get_archetype(name, batting_df):
    name = name.lower()
    for _, row in batting_df.iterrows():
        if name in row['Name'].lower():  # Case-insensitive substring match
            return row['Hitter_Archetype']
    return None

#3
#Function and features to assign pitches to a cluster that will be used later
#Assign pitches to a cluster

#Generalized functions that need to be saved in order to cluster indvidual player specific data later on

#Assign pitches to clusters using feature-specific weights

# Define the feature columns
features = ['release_speed', 'pfx_x', 'pfx_z']

# Define pitch groupings
pitch_group_map = {
    'FF': '4F', 
    'FT': '2F', 'SI': '2F',
    'FA': 'CF', 'FC': 'CF', 
    'SL': 'S', 'ST': 'S',
    'CS': 'C', 'CU': 'C', 'KC': 'C',                                
    'CH': 'CH', 'FS': 'CH'                         
}

# Function to assign a pitch cluster to a row
def assign_pitch_cluster(row):
    pitch_type = row['pitch_type']
    handedness = row['p_throws']

    # Map pitch type to pitch group
    pitch_group = pitch_group_map.get(pitch_type)
    if pitch_group is None:
        return None

    model_key = (handedness, pitch_group)
    model_info = gmm_models.get(model_key)
    if model_info is None:
        return None

    gmm_model, scaler, weights = model_info

    try:
        # Convert row to DataFrame
        feature_values = pd.DataFrame([row[features].values], columns=features)

        # Scale and weight features
        feature_values_scaled = scaler.transform(feature_values)
        feature_values_weighted = feature_values_scaled * weights

        # Predict cluster (1-indexed)
        cluster_num = gmm_model.predict(feature_values_weighted)[0] + 1
        return f"{handedness.upper()}{pitch_group}{cluster_num}"
    except Exception:
        return None

# Function to consolidate clusters
def consolidate_clusters(df, pitch_type_col='pitch_type', cluster_col='pitch_cluster'):
    majority_clusters = (
        df.groupby([pitch_type_col, cluster_col])
        .size()
        .reset_index(name='count')
        .sort_values(['pitch_type', 'count'], ascending=[True, False])
        .groupby(pitch_type_col)
        .first()
        .reset_index()
    )

    pitch_type_to_major_cluster = dict(
        zip(majority_clusters[pitch_type_col], majority_clusters[cluster_col])
    )

    df = df.copy()
    df['standardized_cluster'] = df[pitch_type_col].map(pitch_type_to_major_cluster)
    df['was_reassigned'] = df[cluster_col] != df['standardized_cluster']
    return df






# Extra innings simulation

#Complete General Function

# Found out the probability of going into extra innings, make it like a coin flip once this number is found.
# For extra innings to occur if guest team, must throw 9 innings
# From here find the distribution for innings thrown in extra innings (for both home and away teams)

#FLip coin --- if yes how many innings (pull from distribution conditioned on if the team is home or away) --- falls into relief simulation once innings pitched is given

#This simulation is all about finding the probability of going into extra innings and the distribution of innings thrown for home and away teams

prob_extra_innings = 0.09

extra_innings = np.random.rand() < prob_extra_innings
print(extra_innings)

# Statcast “events” that finish a plate appearance with ≥1 out
OUT_EVENTS = {
    'strikeout', 'strikeout_double_play',
    'groundout', 'flyout', 'lineout', 'pop_out',
    'field_out', 'force_out',
    'double_play', 'triple_play', 'grounded_into_double_play',
    'fielders_choice_out', 'sac_bunt', 'sac_fly',
    'bunt_groundout', 'bunt_lineout', 'bunt_pop_out'
}

# How many outs each multi‑out event produces
MULTI_OUTS = {
    'double_play'                : 2,
    'grounded_into_double_play'  : 2,

    'strikeout_double_play'      : 2,
    'triple_play'                : 3
}

def outs_from_event(evt) -> int:
    if not isinstance(evt, str):
        return 0
    evt = evt.lower().strip()
    if evt not in OUT_EVENTS:
        return 0
    return MULTI_OUTS.get(evt, 1)

# ----------------------------------------------------------------------------------
# 1) Pull **all 2025** pitch‑level Statcast data
#    (adjust dates if the season is still ongoing)
# ----------------------------------------------------------------------------------

print("Downloading / reading cached Statcast…")
sc = pitcher_data.copy()

# ----------------------------------------------------------------------------------
# 2) Tag each pitch row with the *pitcher’s* team
#    (top half → home team is fielding; bottom → away team is fielding)
# ----------------------------------------------------------------------------------
def pitcher_team(r):
    return r['home_team'] if r['inning_topbot'] == 'Top' else r['away_team']

sc['pitcher_team'] = sc.apply(pitcher_team, axis=1)

# Keep only extra‑inning rows
extras = sc[sc['inning'] > 9].copy()
print(f" Extra–inning rows: {len(extras):,}")


# ----------------------------------------------------------------------------------
# 3) Count outs for each (game_pk, pitcher_team)
# ----------------------------------------------------------------------------------
outs_df = (
    extras
    .assign(outs=lambda df: df['events'].fillna('').astype(str).map(outs_from_event))
    .groupby(['game_pk', 'pitcher_team'], as_index=False)['outs']
    .sum()
)

# ----------------------------------------------------------------------------------
# 4) Reshape to one row per game  →  columns: home_outs, away_outs
# Pivot so each row is one game; columns: outs_by_team
# Assume wide includes: ['game_pk', 'home_team', 'away_team', ... team-code columns ...]
# For example, team-code columns might look like: wide['NYY'], wide['BOS'], etc.


# 2) Pivot to wide format — one row per game, team columns like 'NYY', 'BOS', etc.
wide = outs_df.pivot(index='game_pk', columns='pitcher_team', values='outs').reset_index()

# 3) Merge with home/away teams so we know which team was home/away
# Assume you already have a DataFrame with game metadata like home/away teams
# Let's call it game_teams with columns: ['game_pk', 'home_team', 'away_team']

# Derive game_teams from extras (unique home/away team per game)
game_teams = extras[['game_pk', 'home_team', 'away_team']].drop_duplicates()
wide = wide.merge(game_teams[['game_pk', 'home_team', 'away_team']], on='game_pk', how='left')


# Compute extra innings pitched from outs
# Convert outs to IP, correctly rounding HOME team extra innings
def convert_outs_to_ip(outs, is_home):
    if is_home:
        # Round *up* to next full inning if any outs recorded
        return np.ceil(outs / 3)
    else:
        # Away team: standard baseball IP format

        full = outs // 3
        remainder = outs % 3
        return full + remainder / 10





# Fill missing team outs with 0 first
wide = wide.fillna(0)

# Now safely assign home and away outs by checking home/away team codes
wide['home_outs'] = wide.apply(lambda row: row.get(row['home_team'], 0), axis=1)
wide['away_outs'] = wide.apply(lambda row: row.get(row['away_team'], 0), axis=1)

# Compute innings pitched in extras
wide['home_IP_extras'] = wide['home_outs'].apply(lambda x: convert_outs_to_ip(x, is_home=True))
wide['away_IP_extras'] = wide['away_outs'].apply(lambda x: convert_outs_to_ip(x, is_home=False))



# Show results


home_IP_extras = wide['home_IP_extras'].values
away_IP_extras = wide['away_IP_extras'].values




def sample_extra_innings(pitcher_team_home=True):
    """
    Sample number of extra innings pitched conditioned on if pitcher is home or away.
    If pitcher_team_home=True, sample from home team's extra innings distribution.
    Else, sample from away team's distribution.
    """
    if pitcher_team_home:
        return np.random.choice(home_IP_extras)
    else:
        return np.random.choice(away_IP_extras)

# Example usage:

# Pitcher on home team:
sampled_home_ip = sample_extra_innings(pitcher_team_home=True)


# Pitcher on away team:
sampled_away_ip = sample_extra_innings(pitcher_team_home=False)




def simulate_reliever_innings(simulated_ip, is_mcn_home, mets_win_pct, yankees_win_pct):
    """
    Simulate innings pitched by Yankees relief pitchers given SP IP and home/away.

    Parameters:
    - simulated_ip (float): Innings pitched by Max Fried (starting pitcher).
    - is_mcn_home (bool): True if McNeil's team (NYM) is home, False if away.
    - mets_win_pct (float): Mets team's winning percentage (0 to 1).
    - yankees_win_pct (float): Yankees team's winning percentage (0 to 1).

    Returns:
    - float: Innings pitched by Yankees relief pitchers.
    """
    if not is_mcn_home:
        # McNeil away → relief pitchers pitch entire remaining inning (9 - SP IP)
        relief_ip = 9 - simulated_ip
    else:
        # McNeil home → depends if NYM hits in bottom of 9th
        prob_not_hitting_9th = mets_win_pct / (mets_win_pct + yankees_win_pct)
        
        # Simulate if NYM hits in 9th
        hits_in_9th = random.random() > prob_not_hitting_9th
        if hits_in_9th:
            relief_ip = 9 - simulated_ip
        else:
            relief_ip = 8 - simulated_ip
            
    relief_ip = max(0, relief_ip)
    
    # Get the integer part and decimal part of relief_ip
    integer_part = int(relief_ip)  # Get the whole number part
    decimal_part = relief_ip - integer_part  # Get the decimal part

    # Map the decimal part to .1 or .2 based on the thresholds for .33 and .67
    if 0.3 <= decimal_part < 0.5:
        relief_ip = integer_part + 0.1  # Map to .1 for close to .33
    elif 0.5 <= decimal_part < 0.7:
        relief_ip = integer_part + 0.2  # Map to .2 for close to .67
    else:
        relief_ip = integer_part  # Keep it as a whole number if it's closer to an integer
    
    return relief_ip


def outs_from_ip(ip: float) -> int:
    whole, frac = divmod(round(ip*10), 10)
    return whole*3 + (2 if frac==2 else 1 if frac==1 else 0)

def simulate_pen_bf(ip_needed: float, bf_per_out, rng=None) -> int:
    """
    Sample BF per out until required outs reached, using a provided bf_per_out distribution.
    Avoids ending on a lone 0.5 (DP tail) when only one out left.
    """
    if rng is None:
        rng = np.random.default_rng()

    # use existing helper in this module
    outs_req = outs_from_ip(ip_needed)
    if outs_req == 0:
        return 0

    bf_per_out = np.asarray(bf_per_out, dtype=float)
    samples = rng.choice(bf_per_out, size=outs_req, replace=True)

    # Ensure we don't end with a lone 0.5 when only one out remains
    if samples[-1] == 0.5:
        while True:
            new = rng.choice(bf_per_out, size=1)[0]
            if new != 0.5:
                samples[-1] = new
                break

    return int(samples.sum())

def hitter_facing_relief(simulated_bf, most_recent_spot, bp_bf_sim):
    """
    Calculate how many times McNeil will face relief pitchers.

    Parameters:
        simulated_bf (int): Number of batters faced by starting pitcher.
        mcneil_pa_vs_sp (int): McNeil's lineup spot (1 to 9).
        relief_bfd (int): Number of batters faced by relief pitchers.

    Returns:
        int: Number of times McNeil faces relief pitchers.
    """

    # Calculate next batter spot after starting pitcher's BF
    next_batter_spot = ((simulated_bf) % 9) + 1  # lineup spots 1-9

    count = 0
    for i in range(bp_bf_sim):
        current_spot = ((next_batter_spot + i - 1) % 9) + 1
        if current_spot == most_recent_spot:
            count += 1

    return count




def simulate_hits_in_extras(prob_extra_innings, 
                                    is_mcn_home,
                                    mcneil_spot,
                                    total_bf_pre_extras,
                                    mcneil_xba,
                                    bf_per_out_dist,
                                    home_IP_extras,
                                    away_IP_extras):
    """
    Simulate whether extra innings occurs and how many hits McNeil gets in extras.

    Parameters:
        prob_extra_innings (float): Probability of game going to extras (e.g., 0.09).
        is_mcn_home (bool): Is McNeil's team home?
        mcneil_spot (int): McNeil's lineup spot (1-9).
        total_bf_pre_extras (int): Total batters Mets sent to plate before extras.
        mcneil_xba (float): McNeil's expected batting average.
        bf_per_out_dist (array): Empirical distribution of BF per out.
        home_IP_extras (list): IP values for home team in extras.
        away_IP_extras (list): IP values for away team in extras.

    Returns:
        dict with:
            'extra_happens': bool
            'extra_ip': float
            'extra_bf': int
            'mcneil_ab': int
            'mcneil_hits': int
    """

    # 1️⃣ Check if extras occur
    if np.random.rand() >= prob_extra_innings:
        #print("❌ Extra innings do not occur. McNeil hits in extras = 0.")
        return {
            'extra_happens': False,
            'extra_ip': 0.0,
            'extra_bf': 0,
            'mcneil_ab': 0,
            'mcneil_hits': 0
        }

    #print("✅ Extra innings occurred!")

    # 2️⃣ Determine how many IP opponent relief pitches in extras
    opponent_home = not is_mcn_home
    extra_ip = np.random.choice(home_IP_extras if opponent_home else away_IP_extras)

    # Adjusting innings to the required format (x.1 or x.2)
    def convert_to_thirds(ip):
        ip_int = int(ip)  # Get the whole part (e.g., 8 from 8.33)
        ip_frac = ip - ip_int  # Get the decimal part (e.g., 0.33 from 8.33)
        
        if np.isclose(ip_frac, 0.33):  # Close to .33 → map to .1
            return ip_int + 0.1
        elif np.isclose(ip_frac, 0.67):  # Close to .67 → map to .2
            return ip_int + 0.2
        else:
            return ip  # Keep it as it is for whole numbers

    extra_ip = convert_to_thirds(extra_ip)
    #print(f"Opponent extra innings pitched: {extra_ip:.3f}")

    # 3️⃣ Convert innings to number of outs
    def outs_from_ip(ip):
        whole, frac = divmod(round(ip * 10), 10)
        return whole * 3 + (2 if frac == 2 else 1 if frac == 1 else 0)

    outs_needed = outs_from_ip(extra_ip)

    # If no outs needed, return zero BF and hits immediately
    if outs_needed == 0:
        #print("Extra innings occur, but no outs are needed.")
        return {
            'extra_happens': True,
            'extra_ip': extra_ip,
            'extra_bf': 0,
            'mcneil_ab': 0,
            'mcneil_hits': 0
        }

    # 4️⃣ Simulate BF from BF/out distribution
    bf_samples = np.random.choice(bf_per_out_dist, size=outs_needed, replace=True)
    
    # Ensure the last out is not 0.5 (from double play logic)
    while bf_samples[-1] == 0.5:
        bf_samples[-1] = np.random.choice(bf_per_out_dist)

    total_bf = int(round(bf_samples.sum()))
    #print(f"Opponent faces {total_bf} batters in extras")

    # 5️⃣ Determine how many of those batters are McNeil
    next_spot = (total_bf_pre_extras % 9) + 1
    mcneil_ab = sum(1 for i in range(total_bf)
                      if ((next_spot + i - 1) % 9 + 1) == mcneil_spot)

    #print(f"McNeil has {mcneil_ab} ABs in extras")

    # 6️⃣ Simulate hits
    hits = np.random.binomial(mcneil_ab, mcneil_xba)
    #print(f"McNeil gets {hits} hits in extras")

    return {
        'extra_happens': True,
        'extra_ip': extra_ip,
        'extra_bf': total_bf,
        'mcneil_ab': mcneil_ab,
        'mcneil_hits': hits
    }
