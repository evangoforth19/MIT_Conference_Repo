# --- Imports ---
import os
import unicodedata
import numpy as np
import pandas as pd
import joblib
import requests
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pybaseball import playerid_lookup, playerid_reverse_lookup

from baseball_utils import *
from General_Initialization import (
    encode_count, pitch_group_map, features, elite_woba_threshold,
    power_barrel_threshold, patient_chase_threshold, get_archetype, classify_archetype,
    assign_pitch_cluster, map_description_to_simple, prob_extra_innings,
    OUT_EVENTS, MULTI_OUTS, outs_from_event, pitcher_team, outs_df,
    convert_outs_to_ip, sample_extra_innings, home_IP_extras, away_IP_extras, consolidate_clusters
)

# --- Data / encoders loaded once (as in your original file) ---
batter_data    = pd.read_pickle("batter_data.pkl")
pitcher_data   = pd.read_pickle("pitcher_data.pkl")
batting        = pd.read_pickle("batting_2025.pkl")
encoder_data   = pd.read_pickle("encoder_data.pkl")
bat_stats_2025 = pd.read_pickle("bat_stats_2025.pkl")

stand_encoder   = joblib.load("encoders/stand_encoder.joblib")
arch_encoder    = joblib.load("encoders/arch_encoder.joblib")
pitch_encoder   = joblib.load("encoders/pitch_encoder.joblib")
outcome_encoder = joblib.load("encoders/outcome_encoder.joblib")

# Ensure archetypes exist in batting
batting["Hitter_Archetype"] = batting.apply(classify_archetype, axis=1)

team_woba_data = {
    "Team": [
        "CHC", "NYY", "TOR", "LAD", "ARI", "BOS", "DET", "NYM", "MIL", "SEA",
        "PHI", "HOU", "STL", "ATH", "ATL", "SDP", "TBR", "BAL", "MIN", "MIA",
        "TEX", "CIN", "SFG", "CLE", "LAA", "WSN", "KCR", "PIT", "CHW", "COL"
    ],
    "wOBA": [
        0.333, 0.337, 0.328, 0.334, 0.329, 0.328, 0.322, 0.317, 0.313, 0.319,
        0.323, 0.318, 0.312, 0.323, 0.311, 0.307, 0.316, 0.314, 0.312, 0.309,
        0.298, 0.313, 0.302, 0.296, 0.311, 0.305, 0.298, 0.285, 0.293, 0.296
    ]
}
team_woba = pd.DataFrame(team_woba_data)


class Pitcher:
    def __init__(
        self,
        first_lower,
        first_upper,
        last_lower,
        last_upper,
        team,
        pitcher_data,
        batting,
        team_woba,
        hitter,
        player_id=None
    ):
        self.first_lower = first_lower
        self.first_upper = first_upper
        self.last_lower  = last_lower
        self.last_upper  = last_upper
        self.full_lower  = f"{first_lower} {last_lower}"
        self.full_upper  = f"{first_upper} {last_upper}"
        self.team        = team

        # Dataframes passed in
        self.pitcher_data = pitcher_data
        self.batting      = batting
        self.team_woba    = team_woba.copy()

        # Link the hitter (for GMMs + cluster encoder)
        self.hitter = hitter

        # Allow direct player_id override
        if player_id is not None:
            self.player_id = int(player_id)
        else:
            self.player_id = self._lookup_player_id()

        # Hard validations up-front (no fallbacks)
        self._validate_hitter_cluster_encoder()
        self._validate_hitter_gmm_models()

        # Pipeline
        self._load_and_clean_pitcher_data()
        self._assign_archetypes_and_clusters()   # uses hitter.gmm_models
        self._encode_pitcher_data()              # uses hitter.cluster_encoder
        self._calculate_ip_model()
        self._calculate_bf_model()
        self._calculate_ip_std()
        self.winning_pct()
        self.bf_list()

        os.makedirs("data", exist_ok=True)
        self.pitcher_data_arch.to_pickle(f"data/{self.last_lower}_data.pkl")

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _transform_with_feature_names(scaler, release_speed, pfx_x, pfx_z):
        cols = list(getattr(scaler, "feature_names_in_", ["release_speed", "pfx_x", "pfx_z"]))
        feats_df = pd.DataFrame([[release_speed, pfx_x, pfx_z]], columns=cols)
        return scaler.transform(feats_df)

    @staticmethod
    def strip_accents(text: str) -> str:
        if not isinstance(text, str):
            return text
        return "".join(
            c for c in unicodedata.normalize("NFD", text)
            if unicodedata.category(c) != "Mn"
        )

    # -------------------------
    # Initialization helpers
    # -------------------------
    def _lookup_player_id(self):
        first = self.strip_accents(self.first_upper).strip()
        last  = self.strip_accents(self.last_upper).strip()

        df = playerid_lookup(last, first)
        if df.empty:
            df = playerid_lookup(last.upper(), first.upper())
        if df.empty:
            df = playerid_lookup(first, last)
        if df.empty:
            try:
                rev = playerid_reverse_lookup([f"{first} {last}"])
                if not rev.empty and "key_mlbam" in rev:
                    rev = rev[rev["key_mlbam"].notna()]
                    if not rev.empty:
                        return int(rev.iloc[0]["key_mlbam"])
            except Exception:
                pass

        if not df.empty and "key_mlbam" in df.columns and df["key_mlbam"].notna().any():
            return int(df["key_mlbam"].dropna().astype(int).iloc[0])

        raise ValueError(
            f"MLBAM id not found for pitcher '{first} {last}'. "
            "Tried multiple name variants. Double-check spelling (accents, Jr./Sr.) "
            "or pass an explicit player_id override."
        )

    def _validate_hitter_cluster_encoder(self):
        enc = getattr(self.hitter, "cluster_encoder", None)
        if enc is None or not hasattr(enc, "classes_") or len(enc.classes_) == 0:
            raise RuntimeError("Hitter cluster_encoder is missing or unfitted.")

    def _validate_hitter_gmm_models(self):
        models = getattr(self.hitter, "gmm_models", None)
        if not isinstance(models, dict) or len(models) == 0:
            raise RuntimeError("Hitter gmm_models are missing; rebuild the hitter object first.")

    def _load_and_clean_pitcher_data(self):
        df = self.pitcher_data[self.pitcher_data["pitcher"] == self.player_id].copy()
        if df.empty:
            raise ValueError(
                f"No pitch-by-pitch data found for player_id={self.player_id}. "
                f"Verify the ID or your pitcher_data source."
            )

        df = df.dropna(subset=["pitch_type"])
        df = df.sort_values(["game_date", "inning", "at_bat_number", "pitch_number"])
        df["balls"] = df["balls"].fillna(0).astype(int)
        df["strikes"] = df["strikes"].fillna(0).astype(int)

        # Map team nickname -> 3-letter code (ideally define once at module level and reuse)
        _pitcher_team_map = {
            "Angels": "LAA", "Astros": "HOU", "Athletics": "OAK", "Blue Jays": "TOR",
            "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
            "Diamondbacks": "ARI", "Dodgers": "LAD", "Giants": "SFG", "Guardians": "CLE",
            "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM", "Nationals": "WSH",
            "Orioles": "BAL", "Padres": "SDP", "Phillies": "PHI", "Pirates": "PIT",
            "Rangers": "TEX", "Rays": "TBR", "Red Sox": "BOS", "Reds": "CIN",
            "Rockies": "COL", "Royals": "KCR", "Tigers": "DET", "Twins": "MIN",
            "White Sox": "CWS", "Yankees": "NYY"
        }
        
        pitcher_abbr = _pitcher_team_map.get(self.team)
        if pitcher_abbr is None:
            raise ValueError(f"Unknown team nickname '{self.team}' — cannot derive 3-letter code.")
        
        df["opponent"] = df.apply(
            lambda row: row["away_team"] if row.get("home_team") == pitcher_abbr else row.get("home_team"),
            axis=1
        )


        out_event_map = {
            "strikeout": 1, "strikeout_double_play": 2, "grounded_into_double_play": 2,
            "sac_fly_double_play": 2, "sac_bunt_double_play": 2, "double_play": 2,
            "triple_play": 3, "caught_stealing_2b": 1, "caught_stealing_3b": 1,
            "caught_stealing_home": 1, "pickoff_1b": 1, "pickoff_2b": 1, "pickoff_3b": 1,
            "batter_out": 1, "other_out": 1, "field_out": 1, "force_out": 1, "lineout": 1,
            "flyout": 1, "pop_out": 1, "sac_fly": 1, "sac_bunt": 1
        }
        df["num_outs"] = df["events"].map(out_event_map).fillna(0).astype(int)

        self.raw_pitcher_data = df

    def _assign_archetypes_and_clusters(self):
        df = self.raw_pitcher_data.copy()

        # Archetypes
        df["Hitter_Archetype"] = df["batter_name"].apply(lambda x: get_archetype(x, self.batting))
        df = df[df["Hitter_Archetype"].isin([
            "Elite-Power-Patient", "Elite-Power-Aggressive", "Elite-Contact-Patient", "Elite-Contact-Aggressive",
            "Non-Elite-Power-Patient", "Non-Elite-Power-Aggressive", "Non-Elite-Contact-Patient", "Non-Elite-Contact-Aggressive"
        ])]

        # Assign local clusters using the HITTER'S GMMs
        def _group_from_pitch_type(pt):
            return pitch_group_map.get(pt)

        def _local_cluster_label(hand, group, comp_idx):
            hand_prefix = "L" if hand == "L" else "R"
            return f"{hand_prefix}{group}{comp_idx + 1}"

        def _assign_with_hitter_gmm(row):
            hand = row.get("p_throws")
            pt   = row.get("pitch_type")
            grp  = _group_from_pitch_type(pt)
            if pd.isna(hand) or grp is None:
                return np.nan

            key = (hand, grp)
            if key not in self.hitter.gmm_models:
                return np.nan

            gmm, scaler, weights = self.hitter.gmm_models[key]

            rs, px, pz = row.get("release_speed"), row.get("pfx_x"), row.get("pfx_z")
            if pd.isna(rs) or pd.isna(px) or pd.isna(pz):
                return np.nan

            Xs = self._transform_with_feature_names(scaler, rs, px, pz) * weights  # (1,3)
            comp_idx = int(gmm.predict(Xs)[0])
            return _local_cluster_label(hand, grp, comp_idx)

        df["pitch_cluster"] = df.apply(_assign_with_hitter_gmm, axis=1)

        # Cleaning + standardization
        df = df[df["zone"].notnull() & df["zone"].isin(range(1, 15))]
        df["description"] = df["description"].apply(map_description_to_simple)
        df = df[df["description"] != "unknown"]

        self.pitcher_data_arch = df

    def _encode_pitcher_data(self):
        df = self.pitcher_data_arch.copy()

        # zone_enc
        df = df[df["zone"].notna()]
        df["zone_enc"] = df["zone"].astype(int) - 1

        # count_enc → guaranteed scalar
        def _safe_count(b, s):
            try:
                out = encode_count(int(b), int(s))
            except Exception:
                return np.nan
            if isinstance(out, dict):
                if not out:
                    return np.nan
                return int(max(out.items(), key=lambda kv: float(kv[1]))[0])
            if isinstance(out, (list, tuple, np.ndarray, pd.Series)):
                arr = np.asarray(out, dtype=float).ravel()
                if arr.size == 0 or np.isnan(arr).all():
                    return np.nan
                return int(np.nanargmax(arr))
            try:
                return int(out)
            except Exception:
                return np.nan

        df["count_enc"] = pd.Series(
            (_safe_count(b, s) for b, s in zip(df["balls"], df["strikes"])),
            index=df.index
        ).astype("Int64")

        # stand_enc → only allow labels the encoder knows; no None
        if df["stand"].isna().any():
            df = df[df["stand"].notna()]
        allowed_stands = set(stand_encoder.classes_)
        df = df[df["stand"].isin(allowed_stands)]
        df["stand_enc"] = stand_encoder.transform(df["stand"])

        # pitch_cluster_enc using the HITTER'S encoder label space (must exist)
        enc = self.hitter.cluster_encoder
        allowed_clusters = set(enc.classes_)
        df = df[df["pitch_cluster"].isin(allowed_clusters)]
        if df.empty:
            raise RuntimeError(
                f"No encodable rows for {self.full_upper}: "
                f"none of the assigned pitch_cluster labels intersect hitter’s label space."
            )
        df["pitch_cluster_enc"] = enc.transform(df["pitch_cluster"])

        # outcome + archetype encodings (strict)
        df["outcome_enc"] = outcome_encoder.transform(df["description"])
        df["arch_enc"]    = arch_encoder.transform(df["Hitter_Archetype"])

        # Final sanity: no NaNs in the features we need later
        need = ["stand_enc", "count_enc", "arch_enc", "pitch_cluster_enc", "zone_enc", "outcome_enc"]
        bad = df[need].isna().sum().to_dict()
        if any(v > 0 for v in bad.values()):
            raise RuntimeError(f"NaNs in encoded pitcher features: {bad}")

        self.pitcher_data_arch = df

    # -------------------------
    # Models for IP / BF / std
    # -------------------------
    def _calculate_ip_model(self):
        df = self.raw_pitcher_data
        ip_df = df.groupby(["game_date", "opponent"]).agg(outs=("num_outs", "sum")).reset_index()
        ip_df["ip"] = ip_df["outs"] / 3

        ip_df["opponent"] = ip_df["opponent"].str.lower().str.strip()
        ip_df["woba_team"] = ip_df["opponent"].map({
            "ari": "ARI", "az": "ARI", "atl": "ATL", "bal": "BAL", "bos": "BOS", "chc": "CHC", "cws": "CHW",
            "cin": "CIN", "cle": "CLE", "col": "COL", "det": "DET", "hou": "HOU", "kc": "KCR", "kcr": "KCR",
            "laa": "LAA", "lad": "LAD", "mia": "MIA", "mil": "MIL", "min": "MIN", "nym": "NYM", "nyy": "NYY",
            "oak": "OAK", "ath": "OAK", "phi": "PHI", "pit": "PIT", "sd": "SDP", "sdp": "SDP", "sea": "SEA",
            "sf": "SFG", "sfg": "SFG", "stl": "STL", "tb": "TBR", "tbr": "TBR", "tex": "TEX", "tor": "TOR",
            "wsh": "WSN", "wsn": "WSN"
        })

        merged = ip_df.merge(self.team_woba, left_on="woba_team", right_on="Team", how="left")
        reg_df = merged.dropna(subset=["wOBA", "ip"])
        X = reg_df[["wOBA"]].values
        y = reg_df["ip"].values

        self.IPLinReg = LinearRegression().fit(X, y)

    def _calculate_bf_model(self):
        df = self.raw_pitcher_data
        outs_by_game = df.groupby(["game_date", "opponent"]).agg(outs=("num_outs", "sum")).reset_index()
        bf_df = df.drop_duplicates(subset=["game_date", "opponent", "batter", "inning", "inning_topbot"])
        bf_df = bf_df.groupby(["game_date", "opponent"]).size().reset_index(name="bf")
        bf_df = bf_df.merge(outs_by_game, on=["game_date", "opponent"], how="left")
        bf_df["ip"] = bf_df["outs"] / 3
        bf_df = bf_df.dropna(subset=["ip", "bf"])

        self.poisson_model = smf.glm(formula="bf ~ ip", data=bf_df, family=sm.families.Poisson()).fit()

    def _calculate_ip_std(self):
        df = self.raw_pitcher_data
        ip_df = df.groupby(["game_date", "opponent"]).agg(outs=("num_outs", "sum")).reset_index()
        ip_df["ip"] = ip_df["outs"] / 3
        self.ip_std = ip_df["ip"].std()

    # -------------------------
    # Context helpers
    # -------------------------
    def winning_pct(self):
        nickname_to_full_name = {
            "Yankees": "New York Yankees", "Mets": "New York Mets",
            "Red Sox": "Boston Red Sox", "Blue Jays": "Toronto Blue Jays",
            "Orioles": "Baltimore Orioles", "Rays": "Tampa Bay Rays",
            "White Sox": "Chicago White Sox", "Guardians": "Cleveland Guardians",
            "Tigers": "Detroit Tigers", "Royals": "Kansas City Royals",
            "Twins": "Minnesota Twins", "Astros": "Houston Astros",
            "Mariners": "Seattle Mariners", "Athletics": "Oakland Athletics",
            "Rangers": "Texas Rangers", "Braves": "Atlanta Braves",
            "Phillies": "Philadelphia Phillies", "Marlins": "Miami Marlins",
            "Nationals": "Washington Nationals", "Cubs": "Chicago Cubs",
            "Cardinals": "St. Louis Cardinals", "Pirates": "Pittsburgh Pirates",
            "Reds": "Cincinnati Reds", "Brewers": "Milwaukee Brewers",
            "Dodgers": "Los Angeles Dodgers", "Giants": "San Francisco Giants",
            "Padres": "San Diego Padres", "Diamondbacks": "Arizona Diamondbacks",
            "Rockies": "Colorado Rockies", "Angels": "Los Angeles Angels"
        }

        if self.team not in nickname_to_full_name:
            raise ValueError(f"Team nickname '{self.team}' not recognized.")

        full_team_name = nickname_to_full_name[self.team]
        url = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            for record in data.get("records", []):
                for team_record in record.get("teamRecords", []):
                    if team_record["team"]["name"] == full_team_name:
                        return float(team_record["winningPercentage"])
        except Exception:
            return None

    def bf_list(self):
        pitcher_team_map = {
            "Angels": "LAA", "Astros": "HOU", "Athletics": "OAK", "Blue Jays": "TOR",
            "Braves": "ATL", "Brewers": "MIL", "Cardinals": "STL", "Cubs": "CHC",
            "Diamondbacks": "ARI", "Dodgers": "LAD", "Giants": "SF", "Guardians": "CLE",
            "Mariners": "SEA", "Marlins": "MIA", "Mets": "NYM", "Nationals": "WSH",
            "Orioles": "BAL", "Padres": "SD", "Phillies": "PHI", "Pirates": "PIT",
            "Rangers": "TEX", "Rays": "TB", "Red Sox": "BOS", "Reds": "CIN",
            "Rockies": "COL", "Royals": "KC", "Tigers": "DET", "Twins": "MIN",
            "White Sox": "CWS", "Yankees": "NYY"
        }

        team_abbr = pitcher_team_map.get(self.team)
        if not team_abbr:
            raise ValueError(f"Team '{self.team}' not found in team map.")

        data = self.pitcher_data.copy()

        def _pitcher_team(row):
            return row["home_team"] if row["inning_topbot"] == "Top" else row["away_team"]

        data["pitcher_team"] = data.apply(_pitcher_team, axis=1)

        team_all = data[data["pitcher_team"] == team_abbr].copy()
        team_all.sort_values(["game_pk", "inning", "at_bat_number", "pitch_number"], inplace=True)

        starters = (
            team_all.groupby("game_pk")["pitcher"].first()
            .rename("starter_id")
            .reset_index()
        )
        team_bp = team_all.merge(starters, on="game_pk")
        team_bp = team_bp[team_bp["pitcher"] != team_bp["starter_id"]].drop("starter_id", axis=1)

        pa_key = ["game_pk", "pitcher", "inning", "at_bat_number"]
        pa_df = (
            team_bp
            .groupby(pa_key, as_index=False)
            .last()[pa_key + ["events"]]
        )

        out_events = {
            "strikeout": 1, "strikeout_double_play": 1,
            "groundout": 1, "flyout": 1, "lineout": 1, "pop_out": 1,
            "force_out": 1, "field_out": 1, "double_play": 1,
            "grounded_into_double_play": 1, "triple_play": 1,
            "fielders_choice_out": 1, "sac_bunt": 1, "sac_fly": 1,
            "bunt_groundout": 1, "bunt_lineout": 1, "bunt_pop_out": 1
        }
        multi_outs = {
            "double_play": 2, "grounded_into_double_play": 2,
            "strikeout_double_play": 2, "triple_play": 3
        }

        bf_list = []
        batters_since_last_out = 0

        for _, row in pa_df.iterrows():
            batters_since_last_out += 1
            evt = (row["events"] or "").lower().strip()
            if evt not in out_events:
                continue
            outs_here = multi_outs.get(evt, 1)

            if outs_here == 1:
                bf_list.append(batters_since_last_out)
            else:
                first_val = batters_since_last_out - 0.5
                bf_list.append(first_val)
                for _ in range(outs_here - 1):
                    bf_list.append(0.5)
            batters_since_last_out = 0

        return bf_list
