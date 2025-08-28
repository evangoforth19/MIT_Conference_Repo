# Hitter_Class.py
from __future__ import annotations

import os
import unicodedata
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import requests
import re



from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import CategoricalNB

from pybaseball import playerid_lookup, playerid_reverse_lookup

# Project helpers
from General_Initialization import encode_count, get_archetype, classify_archetype


class Hitter:
    def __init__(
        self,
        first_lower: str,
        first_upper: str,
        last_lower: str,
        last_upper: str,
        team_name: str,
        team_id: int,
        batter_data: pd.DataFrame,
        encoder_data: pd.DataFrame,
        bat_stats_2025: pd.DataFrame,
        batting: pd.DataFrame,
        player_id: Optional[int] = None,
    ):
        # ----- identity/data -----
        self.first_lower = first_lower
        self.first_upper = first_upper
        self.last_lower = last_lower
        self.last_upper = last_upper
        self.team_name = team_name
        self.team_id = team_id
        self.full_lower = f"{first_lower} {last_lower}"
        self.full_upper = f"{first_upper} {last_upper}"

        # Dataframes passed in
        self.batter_data = batter_data
        self.encoder_data = encoder_data
        self.bat_stats_2025 = bat_stats_2025
        self.batting = batting

        # ----- player_id priority: use provided override, else robust lookup -----
        if player_id is not None:
            self.player_id = int(player_id)
        else:
            self.player_id = self._lookup_player_id()

        # ---- load encoders (trained elsewhere) ----
        encoder_files = {
            "stand_encoder": "encoders/stand_encoder.joblib",
            "arch_encoder": "encoders/arch_encoder.joblib",
            "outcome_encoder": "encoders/outcome_encoder.joblib",
        }
        for name, path in encoder_files.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required encoder file not found: {path}. "
                    f"Cannot initialize Hitter object for {self.full_upper} without it."
                )
            try:
                setattr(self, name, joblib.load(path))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load {name} from {path} for {self.full_upper}: {e}"
                )

        # ---- archetype + encoded archetype ----
        batting['Hitter_Archetype'] = batting.apply(classify_archetype, axis=1)
        self.hitter_archetype = get_archetype(self.last_upper, self.batting)
        if self.hitter_archetype is None:
            raise ValueError(f"Could not resolve Hitter_Archetype for {self.full_upper}.")
        self.arch_enc = self.arch_encoder.transform([self.hitter_archetype])[0]

        # ---- to be set by pipeline ----
        self.importances: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        self.gmm_models: Dict[Tuple[str, str], Tuple[GaussianMixture, StandardScaler, np.ndarray]] = {}
        self.cluster_encoder: Optional[LabelEncoder] = None
        self.encoder_data_with_clusters: Optional[pd.DataFrame] = None
        self.hitter_df: pd.DataFrame = pd.DataFrame()
        self.most_recent_spot: Optional[int] = None
        self.most_recent_date: Optional[str] = None
        self.winning_pct_value: Optional[float] = None
        self.xba: Optional[float] = None

        # outcome/xBA artifacts
        self.nb_outcome_model: Optional[CategoricalNB] = None
        self.outcome_lookup_table = None
        self.outcome_class_labels = None
        self.xba_lookup_table = None
        self.global_bip_xba: Optional[float] = None

        # ===== Pipeline =====
        self._fit_feature_importances()            # 1) RF importances per (R/L, group)
        self._fit_gmm_models()                     # 2) 12 hitter-weighted GMMs
        self._init_cluster_encoder_and_clusters()  # 3) LabelEncoder from GMM label space
        self._prepare_hitter_data()                # 4) assign clusters + encodings on hitter data
        self._build_xba_lookup()                   # 5) robust, no-crash
        self._train_outcome_model()                # 6) hybrid outcome artifacts

        # Metadata (optional)
        self._get_player_metadata()
        self.winning_pct_value = self._winning_pct()

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _strip_accents(text: str) -> str:
        if not isinstance(text, str):
            return text
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    def _lookup_player_id(self) -> int:
        first = self._strip_accents(self.first_upper).strip()
        last = self._strip_accents(self.last_upper).strip()

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
            f"MLBAM id not found for hitter '{first} {last}'. "
            "Tried multiple name variants. Double-check spelling (accents, Jr./Sr.) "
            "or pass an explicit player_id override."
        )

    @staticmethod
    def _ensure_pitch_group(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure a 'pitch_group' column exists based on 'pitch_type'.
        Drops rows with missing pitch_type, but raises an error if >10% are dropped.
        """
        if 'pitch_group' in df.columns:
            return df

        if 'pitch_type' not in df.columns:
            raise ValueError("Missing 'pitch_type' column; cannot derive 'pitch_group'.")

        out = df.copy()

        # Drop rows with missing pitch_type
        total_rows = len(out)
        out = out.dropna(subset=['pitch_type'])
        dropped = total_rows - len(out)

        if total_rows > 0 and (dropped / total_rows) > 0.10:
            raise ValueError(
                f"Over 10% of rows ({dropped}/{total_rows}) missing 'pitch_type'. "
                "Likely systemic data issue."
            )

        pitch_group_map = {
            'FF': '4F', 'FA': '4F', 'FC': 'CF', 'SI': '2F', 'FT': '2F',
            'SL': 'S',  'ST': 'S',  'CU': 'C',  'KC': 'C',  'CS': 'C',
            'CH': 'CH', 'FS': 'CH'
        }
        out['pitch_group'] = out['pitch_type'].map(pitch_group_map)

        return out

    # ---------- Feature importance ----------
    def _filter_and_fit_importance(self, hand: str, group: str):
        bdf = self._ensure_pitch_group(self.batter_data)
        if 'batter_name' not in bdf.columns:
            return 0.3333, 0.3333, 0.3333

        df = bdf[
            (bdf['batter_name'].str.lower() == self.full_lower) &
            (bdf['p_throws'] == hand) &
            (bdf['pitch_group'] == group)
        ].dropna(subset=['release_speed', 'pfx_x', 'pfx_z', 'estimated_ba_using_speedangle'])

        df = df[df['estimated_ba_using_speedangle'] > 0]
        if len(df) < 30:
            return 0.3333, 0.3333, 0.3333

        X = df[['release_speed', 'pfx_x', 'pfx_z']]
        y = df['estimated_ba_using_speedangle']
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        imp = pd.Series(model.feature_importances_, index=X.columns)
        return float(imp['release_speed']), float(imp['pfx_x']), float(imp['pfx_z'])

    def _fit_feature_importances(self):
        for hand in ['R', 'L']:
            for group in ['4F', '2F', 'CF', 'S', 'C', 'CH']:
                self.importances[(hand, group)] = self._filter_and_fit_importance(hand, group)

    # ---------- 12 GMMs ----------
    def _fit_gmm_models(self):
        pitch_group_map = {
            'FF': '4F', 'FA': '4F', 'FC': 'CF', 'SI': '2F', 'FT': '2F',
            'SL': 'S',  'ST': 'S',  'CU': 'C',  'KC': 'C',  'CS': 'C',
            'CH': 'CH', 'FS': 'CH'
        }
        group_map = {
            '4F': ['FF', 'FA'], '2F': ['SI', 'FT'], 'CF': ['FC'],
            'S':  ['SL', 'ST'], 'C':  ['CU', 'KC', 'CS'], 'CH': ['CH', 'FS']
        }
        p = pd.read_pickle("pitcher_data.pkl")
        p = p[p['pitch_type'].isin(pitch_group_map.keys())].copy()
        p['pitch_group'] = p['pitch_type'].map(pitch_group_map)
        base = p[['pitch_type', 'pitch_group', 'p_throws', 'release_speed', 'pfx_x', 'pfx_z']].dropna()

        for hand in ['R', 'L']:
            for group in ['4F', '2F', 'CF', 'S', 'C', 'CH']:
                df = base[
                    (base['p_throws'] == hand) &
                    (base['pitch_type'].isin(group_map[group]))
                ][['release_speed', 'pfx_x', 'pfx_z']].dropna()

                if df.empty:
                    continue

                scaler = StandardScaler()
                X = scaler.fit_transform(df)
                weights = np.array(self.importances.get((hand, group), (1.0, 1.0, 1.0)), dtype=float)
                Xw = X * weights
                n_components = 5 if group == 'S' else 4 if group == '4F' else 2 if group in ['CF', 'CH', 'C'] else 3
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(Xw)
                self.gmm_models[(hand, group)] = (gmm, scaler, weights)

        for (h, g), triple in self.gmm_models.items():
            setattr(self, f"gmm_{h}_{g}", triple)

    # ---------- NEW: Build deterministic label vocabulary from GMMs ----------
    def _build_cluster_label_vocab(self) -> list[str]:
        """
        Build the complete label vocabulary directly from the fitted GMMs.
        Guarantees the encoder knows every label _assign_row_cluster can emit.
        """
        if not self.gmm_models:
            raise RuntimeError(
                "No GMMs trained; cannot build cluster label vocabulary. "
                "Check pitcher_data.pkl and _fit_gmm_models() inputs."
            )
        labels = []
        for (hand, group), (gmm, _, _) in self.gmm_models.items():
            n = int(getattr(gmm, "n_components", 0) or 0)
            if n <= 0:
                raise RuntimeError(f"GMM for {(hand, group)} has zero components.")
            labels.extend([f"{hand}{group}{i+1}" for i in range(n)])
        return sorted(set(labels))

    # ---------- Canonical cluster encoder (now from GMM label space) ----------
    def _init_cluster_encoder_and_clusters(self):
        """
        Fit the hitter's cluster_encoder directly from the hitter's GMM label space.
        This guarantees the encoder is always fitted with the full set of local cluster labels,
        independent of how many rows exist in encoder_data.
        """
        # Build expected labels straight from the fitted GMMs
        if not getattr(self, "gmm_models", None):
            raise RuntimeError(f"No GMMs present for {self.full_upper}; cannot build label space.")

        labels = []
        for (hand, group), (gmm, _, _) in self.gmm_models.items():
            n = int(getattr(gmm, "n_components", 0) or 0)
            if n <= 0:
                raise RuntimeError(f"GMM for {(hand, group)} has no components.")
            labels.extend([f"{hand}{group}{i+1}" for i in range(n)])
        classes = sorted(set(labels))
        if not classes:
            raise RuntimeError(f"Empty label space synthesized for {self.full_upper}.")

        # Fit encoder on the full local label space
        enc = LabelEncoder()
        enc.fit(classes)
        self.cluster_encoder = enc
        self.allowed_cluster_labels = np.array(enc.classes_)  # optional: for debugging

        # Keep a convenience view of encoder_data with assigned clusters (not used to fit)
        edf = self._ensure_pitch_group(self.encoder_data.copy())
        edf['pitch_cluster'] = edf.apply(self._assign_row_cluster, axis=1)
        self.encoder_data_with_clusters = edf

        os.makedirs("encoders", exist_ok=True)
        joblib.dump(self.cluster_encoder, f"encoders/cluster_encoder__{self.last_lower}.joblib")

    # ---------- assign cluster using hitter GMMs (class method) ----------
    def _assign_row_cluster(self, row: pd.Series) -> str | float:
        """
        Assign a local cluster label using hitter-specific GMMs.

        If invalid/missing inputs, returns np.nan and (optionally) records a reason
        in self._cluster_na_reasons for upstream diagnostics.
        """
        # lazy-init NA reason counter
        if not hasattr(self, "_cluster_na_reasons"):
            from collections import Counter as _Counter
            self._cluster_na_reasons = _Counter()

        def _note(reason: str):
            try:
                self._cluster_na_reasons[reason] += 1
            except Exception:
                pass

        hand = row.get('p_throws', None)
        group = row.get('pitch_group', None)

        # Validate hand/group
        if hand not in ['R', 'L'] or group not in ['4F', '2F', 'CF', 'S', 'C', 'CH']:
            _note("invalid_hand_or_group")
            return np.nan

        key = (hand, group)
        if key not in self.gmm_models:
            _note(f"missing_gmm_{hand}_{group}")
            return np.nan

        # Required kinematic features
        rs = row.get('release_speed')
        px = row.get('pfx_x')
        pz = row.get('pfx_z')
        if pd.isna(rs) or pd.isna(px) or pd.isna(pz):
            _note("missing_features_release_speed_pfx_x_pfx_z")
            return np.nan

        gmm, scaler, weights = self.gmm_models[key]

        # Guard against malformed GMM/Scaler
        try:
            x_df = pd.DataFrame([[rs, px, pz]], columns=['release_speed', 'pfx_x', 'pfx_z'])
            xw = scaler.transform(x_df) * weights
            comp = int(gmm.predict(xw)[0])
        except Exception:
            _note("gmm_predict_error")
            return np.nan

        return f"{hand}{group}{comp+1}"

    @staticmethod
    def _normalize_str(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return ""
        s = str(x)
        # strip accents, lower, single spaces
        s = Hitter._strip_accents(s).lower().strip()
        return re.sub(r"\s+", " ", s)
    
    @staticmethod
    def _get_row_for_hitter(
        df: pd.DataFrame,
        *,
        player_id: int | None,
        first_upper: str | None = None,
        last_upper: str | None = None,
        full_upper: str | None = None,
        team_name: str | None = None,
    ):
        if df is None or df.empty:
            return None
    
        # 0) Prefer season 2025 if present
        df2 = df.copy()
        if "Season" in df2.columns:
            # accept int 2025 or string "2025"
            df2 = df2[df2["Season"].astype(str) == "2025"] if not df2.empty else df2
    
        # 1) Try ID across common column variants
        if player_id is not None:
            id_cols = ["player_id", "mlbamid", "mlbam", "key_mlbam", "mlb_id", "batter_id"]
            for col in id_cols:
                if col in df2.columns:
                    m = df2[col].astype(str) == str(player_id)
                    if m.any():
                        return df2.loc[m].iloc[0]
    
        # 2) Normalize names and team for flexible matching
        name_cols = [c for c in ["Name", "Player", "player", "player_name"] if c in df2.columns]
        if not name_cols:
            return None
    
        # Build normalized name column once
        name_col = name_cols[0]
        norm_name = df2[name_col].apply(Hitter._normalize_str)
    
        full_norm = Hitter._normalize_str(full_upper or "")
        first_norm = Hitter._normalize_str(first_upper or "")
        last_norm  = Hitter._normalize_str(last_upper or "")
        team_norm  = Hitter._normalize_str(team_name or "")
    
        # 2a) Exact full-name match
        if full_norm:
            eq_full = norm_name == full_norm
            if eq_full.any():
                return df2.loc[eq_full].sort_values(by=[c for c in ["AB","PA"] if c in df2.columns], ascending=False).iloc[0]
    
        # 2b) Both tokens present (first & last)
        if first_norm and last_norm:
            has_both = norm_name.str.contains(rf"\b{re.escape(first_norm)}\b") & \
                       norm_name.str.contains(rf"\b{re.escape(last_norm)}\b")
            if has_both.any():
                cand = df2.loc[has_both]
                # If multiple and team present, try to disambiguate by team
                team_cols = [c for c in ["Team", "Tm", "team", "team_name"] if c in cand.columns]
                if team_cols and team_norm:
                    tn = cand[team_cols[0]].apply(Hitter._normalize_str)
                    by_team = cand[tn.str.contains(re.escape(team_norm))]  # substring ok
                    if not by_team.empty:
                        cand = by_team
                return cand.sort_values(by=[c for c in ["AB","PA"] if c in df2.columns], ascending=False).iloc[0]
    
        # 2c) Last-name fallback
        if last_norm:
            has_last = norm_name.str.contains(rf"\b{re.escape(last_norm)}\b")
            if has_last.any():
                cand = df2.loc[has_last]
                # Disambiguate by team if available
                team_cols = [c for c in ["Team", "Tm", "team", "team_name"] if c in cand.columns]
                if team_cols and team_norm:
                    tn = cand[team_cols[0]].apply(Hitter._normalize_str)
                    by_team = cand[tn.str.contains(re.escape(team_norm))]
                    if not by_team.empty:
                        cand = by_team
                return cand.sort_values(by=[c for c in ["AB","PA"] if c in df2.columns], ascending=False).iloc[0]
    
        return None
    
    @staticmethod
    def _safe_get(row, key, default=0):
        try:
            val = row[key]
        except Exception:
            return default
        return default if pd.isna(val) else val
    
    @staticmethod
    def compute_seasonal_xpa(
        bat_stats_2025: pd.DataFrame, *,
        player_id: int | None,
        last_upper: str,
        first_upper: str | None = None,
        full_upper: str | None = None,
        team_name: str | None = None,
    ):
        """
        Returns (xba_ab, xpa) where:
          xba_ab = Statcast xBA (per AB)
          xpa    = per-PA hit probability = xBA * (AB/PA)
        """
        r = Hitter._get_row_for_hitter(
            bat_stats_2025,
            player_id=player_id,
            first_upper=first_upper,
            last_upper=last_upper,
            full_upper=full_upper,
            team_name=team_name,
        )
        if r is None:
            raise ValueError(f"Missing season row in bat_stats_2025 for {last_upper} (by id or name).")
    
        if "xBA" not in r.index or pd.isna(r["xBA"]):
            raise ValueError(f"Missing seasonal xBA in bat_stats_2025 for {full_upper or last_upper}.")
    
        AB = float(Hitter._safe_get(r, "AB", default=np.nan))
        PA = Hitter._safe_get(r, "PA", default=np.nan)
    
        if pd.isna(PA):
            BB  = float(Hitter._safe_get(r, "BB", 0.0))
            HBP = float(Hitter._safe_get(r, "HBP", 0.0))
            SF  = float(Hitter._safe_get(r, "SF", 0.0))
            SH  = float(Hitter._safe_get(r, "SH", Hitter._safe_get(r, "SAC", 0.0)) or 0.0)
            CI  = float(Hitter._safe_get(r, "CI", 0.0))
            if pd.isna(AB):
                raise ValueError(f"Missing AB (and PA) in bat_stats_2025 for {full_upper or last_upper}.")
            PA = AB + BB + HBP + SF + SH + CI
    
        if AB <= 0 or PA <= 0:
            raise ValueError(f"Non-positive AB ({AB}) or PA ({PA}) for {full_upper or last_upper}.")
    
        xba_ab = float(r["xBA"])
        xpa = xba_ab * (AB / PA)
        return xba_ab, xpa



    # ---------- Prepare hitter rows (assign + encode) ----------
    def _prepare_hitter_data(self):
        # ---- 0) Preconditions ----
        if 'batter_name' not in self.batter_data.columns:
            raise ValueError("batter_data missing required column 'batter_name'.")

        # ---- 1) Start from a copy; enforce batter_name presence with 10% cap ----
        df = self.batter_data.copy()
        total0 = len(df)
        if total0 == 0:
            raise ValueError(f"No rows in batter_data for any player (cannot prepare data for {self.full_upper}).")

        # Drop rows with missing batter_name
        df = df.dropna(subset=['batter_name'])
        dropped_bn = total0 - len(df)
        if (dropped_bn / total0) > 0.10:
            raise ValueError(
                f">10% rows missing batter_name ({dropped_bn}/{total0}) — systemic data issue."
            )

        # Filter to this hitter only
        df = df[df['batter_name'].str.lower() == self.full_lower].copy()
        if df.empty:
            raise ValueError(f"No batter_data rows found for hitter '{self.full_upper}' after filtering.")

        # ---- 2) Ensure pitch_group (this will also drop/validate pitch_type per your 10% rule) ----
        df = self._ensure_pitch_group(df)

        # ---- 3) Assign clusters from fitted GMMs (may yield NaN if inputs missing) ----
        df['pitch_cluster'] = df.apply(self._assign_row_cluster, axis=1)

        # ---- Assign Hitter Archetype
        if 'Hitter_Archetype' not in df.columns:
            df['Hitter_Archetype'] = get_archetype(self.last_upper, self.batting)

        # ---- 4) Validate required columns exist (no silent creation) ----
        must_have = [
            'pitch_cluster', 'zone', 'balls', 'strikes', 'stand', 'description',
            'Hitter_Archetype'
        ]
        missing_cols = [c for c in must_have if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in batter_data for {self.full_upper}: {missing_cols}")

        # ---- 5) Drop NA rows in required columns with 10% cap ----
        total1 = len(df)
        df_clean = df.dropna(subset=must_have)
        dropped_req = total1 - len(df_clean)
        if total1 > 0 and (dropped_req / total1) > 0.10:
            # Provide a small breakdown of which columns are most responsible
            na_counts = {c: int(df[c].isna().sum()) for c in must_have}
            raise ValueError(
                f">10% rows dropped due to NA in required columns ({dropped_req}/{total1}). "
                f"NA breakdown: {na_counts}"
            )
        df = df_clean

        # ---- 6) Map description to canonical tokens (strict) ----
        def _map_desc(desc):
            if desc in ['ball', 'blocked_ball', 'pitchout']:
                return 'ball'
            if desc in ['called_strike', 'swinging_strike', 'swinging_strike_blocked', 'missed_bunt']:
                return 'strike'
            if desc in ['foul', 'foul_tip', 'foul_bunt', 'bunt_foul_tip']:
                return 'foul'
            if desc == 'hit_by_pitch':
                return 'hbp'
            if desc == 'hit_into_play':
                return 'bip'
            raise ValueError(f"Unrecognized pitch description '{desc}' for {self.full_upper}")

        df['description'] = df['description'].apply(_map_desc)

        # ---- 7) Archetype tag (constant for this hitter) ----
        df['Hitter_Archetype'] = self.hitter_archetype

        # ---- 8) Encode clusters (encoder must be present/fitted) ----
        if getattr(self, "cluster_encoder", None) is None or not hasattr(self.cluster_encoder, "classes_"):
            raise RuntimeError(f"cluster_encoder missing/unfitted for {self.full_upper}.")

        if not set(df['pitch_cluster'].unique()).issubset(set(self.cluster_encoder.classes_)):
            unknown = sorted(set(df['pitch_cluster'].unique()) - set(self.cluster_encoder.classes_))
            raise ValueError(f"pitch_cluster contains labels not in encoder for {self.full_upper}: {unknown}")

        df['pitch_cluster_enc'] = self.cluster_encoder.transform(df['pitch_cluster'])

        # ---- 9) Basic numerics (strict) ----
        # Do not silently coerce to 0 — validate numerics and error on bad rows.
        for col in ['balls', 'strikes', 'zone']:
            coerced = pd.to_numeric(df[col], errors='coerce')
            bad = coerced.isna().sum()
            if bad:
                raise ValueError(f"Non-numeric or NA values in '{col}' for {self.full_upper} (count={bad}).")
            df[col] = coerced

        df['zone_enc'] = (df['zone'] - 1).round().astype(int)

        # ---- 10) Encode count/stand/outcome/arch (no fallbacks) ----
        # count_enc must be a scalar int 0..11; raise on any failure
        def _encode_count_scalar(b, s):
            out = encode_count(int(b), int(s))
            if isinstance(out, (list, tuple, np.ndarray, pd.Series)):
                # choose argmax if distribution returned
                arr = np.asarray(out, dtype=float).ravel()
                if arr.size == 0 or np.isnan(arr).all():
                    raise ValueError(f"encode_count produced empty/NaN distribution for {self.full_upper}.")
                return int(np.nanargmax(arr))
            try:
                return int(out)
            except Exception as e:
                raise ValueError(f"encode_count failed for balls={b}, strikes={s}: {e}")

        df['count_enc'] = [_encode_count_scalar(b, s) for b, s in zip(df['balls'], df['strikes'])]

        # stand_enc
        if df['stand'].isna().any():
            raise ValueError(f"NA in 'stand' after cleaning for {self.full_upper}.")
        df['stand_enc'] = self.stand_encoder.transform(df['stand'])

        # outcome_enc
        if df['description'].isna().any():
            raise ValueError(f"NA in 'description' after mapping for {self.full_upper}.")
        df['outcome_enc'] = self.outcome_encoder.transform(df['description'])

        # arch_enc (constant)
        df['arch_enc'] = self.arch_encoder.transform(df['Hitter_Archetype'])

        # ---- 11) Final keep/persist ----
        keep_cols = [
            'pitch_cluster_enc', 'zone_enc', 'count_enc', 'outcome_enc', 'arch_enc',
            'estimated_ba_using_speedangle', 'description'
        ]
        extra_cols = ['pitch_cluster', 'zone', 'balls', 'strikes', 'stand', 'Hitter_Archetype']
        self.hitter_df = df[keep_cols + extra_cols].copy()

        self._persist_hitter_df()

        # ---- 12) Seasonal xBA (per AB) and per-PA hit probability ----
        xba_ab, xpa = self.compute_seasonal_xpa(
            self.bat_stats_2025,
            player_id=self.player_id,
            last_upper=self.last_upper,
            first_upper=self.first_upper,
            full_upper=self.full_upper,
            team_name=self.team_name,
        )
        # Keep both for transparency; use xpa in the sim to avoid inflation
        self.xba_ab = float(xba_ab)   # expected BA per AB (Statcast xBA)
        self.xpa    = float(xpa)      # expected hit probability per PA (use this in MC)
        # Backward-compat if other code references self.xba:
        self.xba = self.xpa


    # ---------- build per-hitter xBA lookup (robust) ----------
    def _build_xba_lookup(self):
        """
        Build hierarchical xBA lookup from hitter_df.
        Requirements:
          - self.hitter_df exists, non-empty
          - required columns present
          - at least one BIP row with valid xBA
        No seasonal or constant fallbacks. Fail fast on violations.
        """
        os.makedirs("models", exist_ok=True)

        # 0) Preflight: hitter_df must exist and have rows
        if not hasattr(self, "hitter_df") or self.hitter_df is None or self.hitter_df.empty:
            raise ValueError(
                f"hitter_df is missing/empty for {self.full_upper}; cannot build xBA lookup."
            )

        # 1) Required columns must exist (no silent creation)
        required_cols = ['pitch_cluster_enc', 'zone_enc', 'count_enc', 'description']
        missing = [c for c in required_cols if c not in self.hitter_df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in hitter_df for {self.full_upper}: {missing}"
            )

        # 2) Filter to rows that have all required encodings present
        df = self.hitter_df.copy()
        # Strict: do not fabricate values; all must be present
        bad_mask = df[required_cols].isna().any(axis=1)
        if bad_mask.any():
            bad_n = int(bad_mask.sum())
            total = len(df)
            raise ValueError(
                f"{bad_n}/{total} rows have NA in required fields for {self.full_upper}. "
                f"First few bad indices: {df.index[bad_mask][:5].tolist()}"
            )

        # 3) Restrict to BIP rows with valid xBA
        bip = df[(df['description'] == 'bip') & df['estimated_ba_using_speedangle'].notna()].copy()
        if bip.empty:
            raise ValueError("No BIP rows to build xBA lookup; cannot proceed.")

        # 4) Global BIP xBA (for diagnostics only)
        self.global_bip_xba = float(
            bip['estimated_ba_using_speedangle'].astype(float).clip(0.0, 1.0).mean()
        )

        # 5) Build hierarchical sum/n tables
        L3, L2, L1 = {}, {}, {}
        G = {('__GLOBAL__',): {"sum": 0.0, "n": 0}}

        def _add(tbl, key, x):
            if key not in tbl:
                tbl[key] = {"sum": float(x), "n": 1}
            else:
                tbl[key]["sum"] += float(x)
                tbl[key]["n"] += 1

        for row in bip.itertuples(index=False):
            p = int(getattr(row, 'pitch_cluster_enc'))
            z = int(getattr(row, 'zone_enc'))
            k = int(getattr(row, 'count_enc'))
            x = float(getattr(row, 'estimated_ba_using_speedangle'))
            if 0.0 <= x <= 1.0:
                _add(L3, (p, z, k), x)
                _add(L2, (p, z), x)
                _add(L1, (p,), x)
                _add(G, ('__GLOBAL__',), x)

        self.xba_lookup_table = {
            "L3": L3, "L2": L2, "L1": L1, "G": G,
            "min_support": (20, 30, 40),
            "prior_equiv": (20, 40, 80),
            "mode": "mean",
            "schema": "hierarchical_sum_n_v1"
        }

        joblib.dump(self.xba_lookup_table, "models/XBA_Lookup_Hierarchical.joblib")

    # ---------- build per-hitter outcome hybrid (lookup + NB) ----------
    def _train_outcome_model(self, min_lookup_count: int = 5):
        """
        Train the hybrid outcome model (lookup + CategoricalNB) with no fallbacks.
        Requirements:
          - self.hitter_df exists and is non-empty
          - required columns present
          - no NA in required columns
          - at least 1 row after filtering
          - at least 2 distinct outcome classes to fit NB
        Raises on any violation. Persists trained artifacts only on success.
        """
        os.makedirs("models", exist_ok=True)

        # 0) Preflight: hitter_df must exist and have rows
        if not hasattr(self, "hitter_df") or self.hitter_df is None or self.hitter_df.empty:
            raise ValueError(
                f"hitter_df is missing/empty for {self.full_upper}; cannot train outcome model."
            )

        # 1) Required columns present?
        required = ['pitch_cluster_enc', 'zone_enc', 'count_enc', 'outcome_enc']
        missing = [c for c in required if c not in self.hitter_df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns for outcome model ({self.full_upper}): {missing}"
            )

        # 2) Strict NA policy (we already enforced earlier; double-check here)
        df = self.hitter_df.copy()
        bad_mask = df[required].isna().any(axis=1)
        if bad_mask.any():
            bad_n = int(bad_mask.sum())
            total = len(df)
            raise ValueError(
                f"{bad_n}/{total} rows have NA in required outcome fields for {self.full_upper}. "
                f"First few bad indices: {df.index[bad_mask][:5].tolist()}"
            )

        # 3) Filtered set must not be empty
        if len(df) == 0:
            raise ValueError(
                f"Insufficient data to train outcome model for {self.full_upper} (empty after filtering)."
            )

        # 4) Prepare X, y; require ≥2 classes in y for NB to be meaningful
        X = df[['pitch_cluster_enc', 'zone_enc', 'count_enc']].astype(int).values
        y = df['outcome_enc'].astype(int).values
        unique_y = np.unique(y)
        if unique_y.size < 2:
            raise ValueError(
                f"Insufficient class variety to train outcome model for {self.full_upper}: "
                f"found {unique_y.size} class(es) in 'outcome_enc'."
            )

        # 5) Build lookup table (counts only; thresholds applied downstream at inference)
        lookup_table = defaultdict(Counter)
        for feats, target in zip(X, y):
            lookup_table[tuple(feats)][int(target)] += 1

        # Optional: sanity check for extremely sparse table (informative, not a fallback)
        total_keys = len(lookup_table)
        if total_keys == 0:
            raise ValueError(
                f"No keys populated in outcome lookup table for {self.full_upper}; cannot train."
            )

        # 6) Fit NB (categorical)
        nb_model = CategoricalNB()
        nb_model.fit(X, y)
        class_labels = nb_model.classes_

        # 7) Persist artifacts
        self.outcome_lookup_table = lookup_table
        self.nb_outcome_model = nb_model
        self.outcome_class_labels = class_labels

        joblib.dump(self.nb_outcome_model,     "models/hybrid_outcome_nb_model.joblib")
        joblib.dump(self.outcome_lookup_table, "models/hybrid_outcome_lookup_table.joblib")
        joblib.dump(self.outcome_class_labels, "models/hybrid_outcome_class_labels.joblib")

    # ---------- persistence ----------
    def _persist_hitter_df(self):
        os.makedirs("data", exist_ok=True)
        self.hitter_df.to_pickle(f"data/{self.last_lower}_data.pkl")

    # ---------- metadata ----------
    def _get_player_metadata(self):
        def get_recent_game_ids(team_id, lookback_days=14):
            today = datetime.utcnow().date()
            start, end = (today - timedelta(days=lookback_days)).isoformat(), today.isoformat()
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}&startDate={start}&endDate={end}"
            try:
                resp = requests.get(url, timeout=10)
                dates = resp.json().get('dates', [])
                return [
                    {'gamePk': g['gamePk'], 'date': g['gameDate'][:10]}
                    for d in dates for g in d.get('games', [])
                    if g.get('status', {}).get('detailedState') in [
                        "Final", "Completed Early", "In Progress", "In Progress - Delay"
                    ]
                ]
            except Exception:
                return []

        def get_batting_order_spot(game_id, player_id):
            try:
                url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
                team = requests.get(url, timeout=10).json()['liveData']['boxscore']['teams']
                for side in ['home', 'away']:
                    for p in team[side]['players'].values():
                        if str(p['person']['id']) == str(player_id):
                            return int(str(p.get('battingOrder', '0'))[0]) if 'battingOrder' in p else None
            except Exception:
                pass
            return None

        if self.player_id is not None and self.team_id is not None:
            games = sorted(get_recent_game_ids(self.team_id), key=lambda x: x['date'], reverse=True)
            for g in games:
                spot = get_batting_order_spot(g['gamePk'], self.player_id)
                if spot:
                    self.most_recent_spot = spot
                    self.most_recent_date = g['date']
                    break

    def _winning_pct(self) -> Optional[float]:
        nickname_to_full_name = {
            "Yankees": "New York Yankees", "Red Sox": "Boston Red Sox",
            "Blue Jays": "Toronto Blue Jays", "Orioles": "Baltimore Orioles",
            "Rays": "Tampa Bay Rays", "White Sox": "Chicago White Sox",
            "Guardians": "Cleveland Guardians", "Tigers": "Detroit Tigers",
            "Royals": "Kansas City Royals", "Twins": "Minnesota Twins",
            "Astros": "Houston Astros", "Mariners": "Seattle Mariners",
            "Athletics": "Oakland Athletics", "Rangers": "Texas Rangers",
            "Angels": "Los Angeles Angels", "Braves": "Atlanta Braves",
            "Phillies": "Philadelphia Phillies", "Marlins": "Miami Marlins",
            "Nationals": "Washington Nationals", "Mets": "New York Mets",
            "Cubs": "Chicago Cubs", "Cardinals": "St. Louis Cardinals",
            "Pirates": "Pittsburgh Pirates", "Reds": "Cincinnati Reds",
            "Brewers": "Milwaukee Brewers", "Dodgers": "Los Angeles Dodgers",
            "Giants": "San Francisco Giants", "Padres": "San Diego Padres",
            "Diamondbacks": "Arizona Diamondbacks", "Rockies": "Colorado Rockies"
        }
        if self.team_name not in nickname_to_full_name:
            return None
        full_team_name = nickname_to_full_name[self.team_name]
        url = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            for record in data.get('records', []):
                for team_record in record.get('teamRecords', []):
                    if team_record['team']['name'] == full_team_name:
                        return float(team_record['winningPercentage'])
        except Exception:
            pass
        return None
