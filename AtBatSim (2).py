# AtBatSim.py — lean, stateless, reload-safe (no project-wide imports)

import numpy as np
from baseball_utils import *

import pandas as pd


# ---------- tiny helpers (no GI import) ----------
def _resolve_hitter_stand(hitter, pitcher):
    """
    Return 'L' or 'R' for the hitter's stance in this matchup.

    Priority:
      1) hitter.batting or hitter.bat_stats_2025 -> 'bats'
      2) mode of hitter.batter_data['stand']
      3) legacy scan in pitcher_df for a name match (if available)
      4) default 'L'

    If the hitter is switch ('S'), use opposite of pitcher throw ('L'/'R').
    """
    def _upper(x):
        return None if x is None else str(x).strip().upper()

    # ---- tiny, pandas-agnostic helpers ----
    def _has_cols_df(obj):
        # duck-typing: DataFrame-like if has 'columns' and 'to_numpy' or 'empty'
        return hasattr(obj, "columns") and (hasattr(obj, "to_numpy") or hasattr(obj, "empty"))

    def _get_df_column(df, name):
        # safe getter without 'in df.columns' truthiness
        if hasattr(df, "get"):
            s = df.get(name, None)
            if s is not None:
                return s
        try:
            return df[name]
        except Exception:
            return None

    def _flatten_to_list(obj):
        # Normalize DF/Series/ndarray/list/tuple to a flat Python list of strings (no NAs/empties)
        try:
            import numpy as _np  # local import, but np is already available
        except Exception:
            _np = None

        arr = None
        if obj is None:
            return []
        # DataFrame
        if hasattr(obj, "to_numpy") and hasattr(obj, "columns"):
            try:
                arr = obj.to_numpy().ravel()
            except Exception:
                arr = None
        # Series-like
        if arr is None and hasattr(obj, "to_numpy") and not hasattr(obj, "columns"):
            try:
                arr = obj.to_numpy().ravel()
            except Exception:
                arr = None
        # numpy-like
        if arr is None and hasattr(obj, "ravel"):
            try:
                arr = obj.ravel()
            except Exception:
                arr = None
        # list/tuple
        if arr is None and isinstance(obj, (list, tuple)):
            arr = obj
        if arr is None:
            return []

        out = []
        for v in arr:
            s = "" if v is None else str(v).strip()
            if s:  # drop empties
                out.append(s)
        return out

    def _majority_letter(values):
        # Majority vote on first letter (L/R/S) among strings
        from collections import Counter
        letters = [v[0].upper() for v in values if v]
        if not letters:
            return None
        c = Counter(letters)
        top, _ = max(c.items(), key=lambda kv: (kv[1], kv[0]))
        return top

    # --- 1) Try hitter profile tables for a 'bats' field ---
    bats = None
    profile_sources = [getattr(hitter, "batting", None), getattr(hitter, "bat_stats_2025", None)]
    for src in profile_sources:
        if src is None or not _has_cols_df(src):
            continue

        # Prefer player_id if present
        row = None
        pid = getattr(hitter, "player_id", None)
        if pid is not None and ("player_id" in getattr(src, "columns", [])):
            try:
                m = (src["player_id"] == pid)
                if hasattr(m, "any") and bool(m.any()):
                    row = src.loc[m].iloc[0]
            except Exception:
                row = None

        if row is None:
            # Fallback by name
            name_col = None
            for c in getattr(src, "columns", []):
                lc = str(c).lower()
                if lc in ("name", "player", "batter_name"):
                    name_col = c
                    break
            if name_col is not None:
                try:
                    ser = src[name_col].astype(str)
                    m = ser.str.contains(getattr(hitter, "full_upper", ""), case=False, na=False)
                    if hasattr(m, "any") and bool(m.any()):
                        row = src.loc[m].iloc[0]
                except Exception:
                    row = None

        if row is not None:
            for cand in ("bats", "BatSide", "bat_side", "stand_primary"):
                try:
                    if hasattr(row, "index") and (cand in row.index):
                        val = _upper(row[cand])
                        if val in ("L", "R", "S"):
                            bats = val
                            break
                except Exception:
                    continue
        if bats is not None:
            break

    # --- 2) Fallback to mode of hitter.batter_data['stand'] ---
    if bats is None:
        bd = getattr(hitter, "batter_data", None)
        if _has_cols_df(bd) and ("stand" in getattr(bd, "columns", [])):
            try:
                stand_vals = _flatten_to_list(bd["stand"])
                # Normalize to L/R/S letters if present
                stand_letters = [s[0].upper() for s in stand_vals if s]
                maj = _majority_letter(stand_letters)
                if maj in ("L", "R", "S"):
                    bats = maj
            except Exception:
                pass

    # --- Pitcher throwing hand (no pandas boolean checks) ---
    p_throws = None

    # Attribute-level quick wins
    for attr in ("throws", "p_throws", "pitcher_throws"):
        val = getattr(pitcher, attr, None)
        if isinstance(val, str) and val.strip():
            p_throws = _upper(val)[0]
            break

    # DataFrame-level/pack-level
    if p_throws not in ("L", "R", "S"):
        pdp = getattr(pitcher, "pitcher_data", None)
        if _has_cols_df(pdp):
            for col in ("p_throws", "throws", "p_throws_hand", "pitcher_throws"):
                try:
                    raw = _get_df_column(pdp, col)
                    vals = _flatten_to_list(raw)
                    if vals:
                        top = _majority_letter(vals)
                        if top in ("L", "R", "S"):
                            p_throws = top
                            break
                except Exception:
                    continue

    # --- Switch hitters face opposite of pitcher throw ---
    if bats == "S" and p_throws in ("L", "R"):
        return "R" if p_throws == "L" else "L"

    if bats in ("L", "R"):
        return bats

    # --- 3) Legacy final fallback: substring scan on pitcher_df (kept for robustness) ---
    pitcher_df = getattr(pitcher, "pitcher_data_arch", None)
    if not (_has_cols_df(pitcher_df) and getattr(pitcher_df, "empty", True) is False):
        pitcher_df = getattr(pitcher, "pitcher_data", None)

    if _has_cols_df(pitcher_df):
        bn = _get_df_column(pitcher_df, "batter_name")
        st = _get_df_column(pitcher_df, "stand")
        if bn is not None and st is not None:
            try:
                names = _flatten_to_list(bn)
                stands = _flatten_to_list(st)
                # simple scan for a name hit
                target = str(getattr(hitter, "full_upper", "")).lower()
                for name, stand in zip(names, stands):
                    if target and isinstance(name, str) and (target in name.lower()):
                        first = stand[0].upper() if isinstance(stand, str) and stand else None
                        if first in ("L", "R", "S"):
                            return first if first in ("L", "R") else ("R" if p_throws == "L" else "L")
                        break
            except Exception:
                pass

    # --- 4) Rock-solid default ---
    return "L"



def _encode_count(balls, strikes):
    table = {
        (0,0):0,(0,1):1,(0,2):2,(1,0):3,(1,1):4,(1,2):5,
        (2,0):6,(2,1):7,(2,2):8,(3,0):9,(3,1):10,(3,2):11
    }
    return table.get((int(balls), int(strikes)), 0)

def _argmax_counter(counter_dict):
    # counter_dict: {class_label: count}
    return max(counter_dict.items(), key=lambda kv: (kv[1], kv[0]))[0]

def hybrid_predict_row(key, lookup_table, nb_model, proba_classes):
    """
    If key is in lookup_table use argmax counts; else NB sample by probs.
    `proba_classes` must align with nb_model.classes_.
    """
    if key in lookup_table and len(lookup_table[key]) > 0:
        return _argmax_counter(lookup_table[key])
    probs = nb_model.predict_proba([list(key)])[0]  # (n_classes,)
    return np.random.choice(proba_classes, p=probs)

def hybrid_pitch_predict(X, nb_model, lookup_table, class_labels):
    """
    X: list of [stand_enc, count_enc, arch_enc]
    Returns: list[int] of predicted pitch_cluster_enc (global ids)
    """
    preds = []
    for row in X:
        key = tuple(int(v) for v in row)
        pred = hybrid_predict_row(key, lookup_table, nb_model, class_labels)
        preds.append(int(pred))
    return preds  # just the list

def predict_zone_hybrid(X, zone_nb_model, zone_lookup_table, zone_class_labels):
    """
    X: list of [pitch_cluster_enc, count_enc, stand_enc]
    Returns: list[int] of predicted zone_enc
    """
    preds = []
    for row in X:
        key = tuple(int(v) for v in row)
        pred = hybrid_predict_row(key, zone_lookup_table, zone_nb_model, zone_class_labels)
        preds.append(int(pred))
    return preds

# ---------- outcome (guarded hybrid) ----------

# Tunables
MIN_LOOKUP_SUPPORT = 3        # need at least this many obs in a bucket
MIN_LOOKUP_DOMINANCE = 0.70   # top class must be ≥ this fraction

def _lookup_is_trustworthy(counter_dict):
    total = sum(counter_dict.values())
    if total < MIN_LOOKUP_SUPPORT:
        return False
    top = max(counter_dict.values())
    return (top / total) >= MIN_LOOKUP_DOMINANCE

def predict_outcome(
    *, pitch_cluster_enc, zone_enc, count_enc,
    nb_model, lookup_table, class_labels,
    force_model: bool = False
):
    """
    Hierarchical lookup backoff with quality gates; optional force_model to skip lookup.
    Keys: (c,z,k) → (c,z) → (c,k) → (c,) → (z,k) → (z,)
    """
    c, z, k = int(pitch_cluster_enc), int(zone_enc), int(count_enc)

    if not force_model:
        for key in ((c, z, k), (c, z), (c, k), (c,), (z, k), (z,)):
            bucket = lookup_table.get(key)
            if bucket:
                if _lookup_is_trustworthy(bucket):
                    # deterministic argmax to keep this path fast
                    return _argmax_counter(bucket)
                else:
                    # reject sparse/peaky bucket → fall through to model
                    break

    probs = nb_model.predict_proba([[c, z, k]])[0]
    return int(np.random.choice(class_labels, p=probs))


# ---------- xBA prediction (supports hierarchical + legacy) ----------

def _xba_is_hierarchical(table: dict) -> bool:
    # New schema has these keys
    return isinstance(table, dict) and all(k in table for k in ("L3", "L2", "L1", "G"))

def _mean_sum_n(stat):
    # stat: {"sum": float, "n": int}
    n = stat.get("n", 0) if isinstance(stat, dict) else 0
    return (stat["sum"] / n) if (n and n > 0) else None

def _shrunken_posterior(sum_x, n, prior_mean, prior_n):
    # posterior mean with equivalent sample size prior
    return (sum_x + prior_mean * prior_n) / (n + prior_n)

def _predict_xba_hierarchical(c, z, k, table, global_fallback=0.300):
    # Unpack
    L3, L2, L1, G = table["L3"], table["L2"], table["L1"], table["G"]
    min3, min2, min1 = table.get("min_support", (20, 30, 40))
    e3, e2, e1       = table.get("prior_equiv", (20, 40, 80))

    key3 = (c, z, k); key2 = (c, z); key1 = (c,); keyG = ('__GLOBAL__',)

    # Global mean
    g_stat = G.get(keyG)
    g_mu = _mean_sum_n(g_stat) if g_stat else float(global_fallback)

    # L1 shrinks to global
    s1 = L1.get(key1)
    if s1 and s1.get("n", 0) >= min1:
        mu1 = _shrunken_posterior(s1["sum"], s1["n"], g_mu, e1)
    else:
        mu1 = g_mu

    # L2 shrinks to L1
    s2 = L2.get(key2)
    if s2 and s2.get("n", 0) >= min2:
        mu2 = _shrunken_posterior(s2["sum"], s2["n"], mu1, e2)
    else:
        mu2 = mu1

    # L3 shrinks to L2
    s3 = L3.get(key3)
    if s3 and s3.get("n", 0) >= min3:
        xba = _shrunken_posterior(s3["sum"], s3["n"], mu2, e3)
    elif s2 and s2.get("n", 0) > 0:
        xba = mu2
    elif s1 and s1.get("n", 0) > 0:
        xba = mu1
    else:
        xba = g_mu if g_mu is not None else float(global_fallback)

    # clip for safety
    return float(max(0.0, min(1.0, xba)))

def predict_xba(pitch_cluster_enc, zone_enc, count_enc, xba_lookup_table, global_fallback=0.300):
    """
    xBA lookup using:
      - NEW hierarchical table: EB shrinkage L3->L2->L1->GLOBAL
      - LEGACY table: hierarchical backoff over tuple keys that map to lists of xBA values
      - Final fallback: global_fallback
    """
    if not xba_lookup_table:
        return float(global_fallback)

    c, z, k = int(pitch_cluster_enc), int(zone_enc), int(count_enc)

    # New schema path
    if _xba_is_hierarchical(xba_lookup_table):
        return _predict_xba_hierarchical(c, z, k, xba_lookup_table, global_fallback=global_fallback)

    # Legacy schema path (dict[tuple] -> list of xBA)
    for key in ((c, z, k), (c, z), (c, k), (c,), (z, k), (z,)):
        vals = xba_lookup_table.get(key)
        if vals:
            return float(np.mean(vals))
    return float(global_fallback)


# Optional: lightweight logger (no-op by default)
def print_at_bat_log(log):
    # Replace with any pretty-printer you want; left blank to keep module light.
    pass


# ---------- Main simulation ----------

def simulate_at_bat_between(
    hitter,
    pitcher,
    nb_pitch_model,
    pitch_lookup_table,
    pitch_class_labels,
    nb_zone_model,
    zone_lookup_table,
    zone_class_labels,
    verbose=True,
    verbose_audit=False
):
    """
    Works with either full class instances or 'pack' namespaces that expose the same attributes.
    Hot-path optimizations:
      - O(1) handedness lookup via pitcher.stand_by_batter_lower
      - No label inverse_transform unless verbose_audit=True
      - No GI import (local _encode_count)
    """
    # Two-strike foul breaker state
    FOUL_STREAK_CAP = 10
    foul_streak = 0
    force_model_outcome = False

    # Hard failsafe: max pitches per AB
    MAX_PITCHES_PER_AB = 20
    pitch_count = 0

    # --- Resolve pitcher dataframe from object ---
    pitcher_df = getattr(pitcher, "pitcher_data_arch", None)
    if pitcher_df is None or (hasattr(pitcher_df, "empty") and pitcher_df.empty):
        pitcher_df = getattr(pitcher, "pitcher_data", None)
    if pitcher_df is None or len(pitcher_df) == 0:
        raise ValueError("Pitcher object has no non-empty `pitcher_data_arch` or `pitcher_data`.")

    # --- Pull encoders/models from hitter object ---
    cluster_encoder = hitter.cluster_encoder
    stand_encoder   = hitter.stand_encoder
    arch_encoder    = getattr(hitter, "arch_encoder", None)
    outcome_encoder = hitter.outcome_encoder

    nb_outcome_model      = hitter.nb_outcome_model
    outcome_lookup_table  = hitter.outcome_lookup_table
    outcome_class_labels  = hitter.outcome_class_labels
    xba_lookup_table      = hitter.xba_lookup_table
    global_bip_xba        = float(getattr(hitter, "global_bip_xba", 0.300) or 0.300)

    # --- Resolve names (for logging only) ---
    hitter_name_str  = getattr(hitter, "full_lower", None) or \
                       f"{getattr(hitter, 'first_lower', 'unknown')} {getattr(hitter, 'last_lower', 'unknown')}"
    pitcher_name_str = getattr(pitcher, "full_lower", None) or \
                       f"{getattr(pitcher, 'first_lower', 'unknown')} {getattr(pitcher, 'last_lower', 'unknown')}"

    # --- Handedness lookup (profile-first, robust, switch-aware) ---
    hitter_hand = _resolve_hitter_stand(hitter, pitcher)

    # Encode with hitter's stand_encoder if present; else global fallback
    stand_encoder = getattr(hitter, "stand_encoder", None)
    if stand_encoder is None:
        import joblib as _joblib
        stand_encoder = _joblib.load("encoders/stand_encoder.joblib")
    stand_enc = stand_encoder.transform([hitter_hand])[0]

    # --- Archetype encoding (prefer precomputed arch_enc on packs) ---
    if hasattr(hitter, "arch_enc") and hitter.arch_enc is not None:
        arch_enc = int(hitter.arch_enc)
    else:
        # fallback: derive from hitter_df/batting using arch_encoder
        hitter_rows = getattr(hitter, "batter_data", None)
        if hitter_rows is None:
            raise ValueError("arch_enc missing and no batter_data available to derive it.")
        rows = hitter_rows[hitter_rows['batter_name'].str.contains(hitter_name_str, case=False, na=False)]
        if rows.empty:
            raise ValueError(f"No archetype data found for {hitter_name_str}")
        if 'arch_enc' in rows.columns:
            arch_enc = int(rows['arch_enc'].values[0])
        else:
            if arch_encoder is None:
                raise ValueError("arch_enc not found and hitter.arch_encoder missing.")
            label = rows['Hitter_Archetype'].values[0] if 'Hitter_Archetype' in rows.columns \
                    else getattr(hitter, "hitter_archetype", None)
            if label is None:
                raise ValueError("Cannot infer hitter archetype label to encode.")
            arch_enc = int(arch_encoder.transform([label])[0])

    balls, strikes = 0, 0
    pitch_num = 1
    make_logs = bool(verbose or verbose_audit)
    log = [] if make_logs else None

    while True:
        # ---- Pitch cap failsafe ----
        pitch_count += 1
        if pitch_count > MAX_PITCHES_PER_AB:
            # Force end as OUT; log for visibility
            if make_logs:
                log.append({"Pitch #": pitch_num, "Terminal": "ABORT_PITCH_CAP", "cap": MAX_PITCHES_PER_AB})
                if verbose: print_at_bat_log(log)
            return 'OUT', log if make_logs else None

        count_enc = _encode_count(balls, strikes)

        # --- PITCH PREDICTION ---
        preds = hybrid_pitch_predict(
            [[stand_enc, count_enc, arch_enc]],
            nb_pitch_model, pitch_lookup_table, pitch_class_labels
        )
        pitch_global_id = preds[0]

        # Only compute human-readable label when auditing
        if verbose_audit:
            pitch_cluster_label = cluster_encoder.inverse_transform([pitch_global_id])[0]

        # --- ZONE PREDICTION ---
        zone_enc = predict_zone_hybrid(
            [[pitch_global_id, count_enc, stand_enc]],
            zone_nb_model=nb_zone_model,
            zone_lookup_table=zone_lookup_table,
            zone_class_labels=zone_class_labels
        )[0]

        # --- OUTCOME PREDICTION ---
        outcome_enc = predict_outcome(
            pitch_cluster_enc=pitch_global_id,
            zone_enc=zone_enc,
            count_enc=count_enc,
            nb_model=nb_outcome_model,
            lookup_table=outcome_lookup_table,
            class_labels=outcome_class_labels,
            force_model=force_model_outcome  # two-strike foul breaker can force model
        )
        outcome_label = outcome_encoder.inverse_transform([outcome_enc])[0]

        # --- Terminal outcomes first (fast path w/o logging) ---
        if not make_logs:
            if outcome_label == "bip":
                xba = predict_xba(pitch_global_id, zone_enc, count_enc, xba_lookup_table, global_fallback=global_bip_xba)
                return ('HIT', None) if (np.random.rand() < xba) else ('OUT', None)
            if outcome_label == "hbp":
                return 'HBP', None

        # --- Logging path for terminal outcomes ---
        if make_logs and outcome_label in ("bip", "hbp"):
            entry = {"Pitch #": pitch_num, "Zone": zone_enc + 1, "Outcome": outcome_label}
            if verbose_audit:
                entry["Pitch Cluster"] = pitch_cluster_label
            if outcome_label == "bip":
                xba = predict_xba(pitch_global_id, zone_enc, count_enc, xba_lookup_table, global_fallback=global_bip_xba)
                is_hit = np.random.rand() < xba
                entry["xBA"] = round(xba, 3)
                entry["BIP Result"] = "Hit" if is_hit else "Out"
                log.append(entry)
                if verbose: print_at_bat_log(log)
                return ('HIT', log) if is_hit else ('OUT', log)
            else:  # hbp
                entry["Terminal"] = "HBP"
                log.append(entry)
                if verbose: print_at_bat_log(log)
                return 'HBP', log

        # --- Count update ---
        # map: outcome -> (add_balls, add_strikes, max_2_flag)
        add_balls, add_strikes, max2 = {
            "ball":   (1, 0, False),
            "strike": (0, 1, False),
            "foul":   (0, 1, True),
            "hbp":    (0, 0, False),  # handled earlier
            "bip":    (0, 0, False),  # handled earlier
        }[outcome_label]

        if add_balls:
            balls += add_balls
        if add_strikes:
            if max2 and strikes >= 2:
                pass
            else:
                strikes += add_strikes

        # ---- Two-strike foul breaker bookkeeping ----
        if outcome_label == "foul" and strikes == 2:
            foul_streak += 1
            if foul_streak >= FOUL_STREAK_CAP:
                force_model_outcome = True
        else:
            if force_model_outcome and (outcome_label != "foul" or strikes < 2):
                force_model_outcome = False
            foul_streak = 0

        # --- Terminal counts ---
        if balls >= 4:
            if make_logs:
                log.append({"Pitch #": pitch_num, "Terminal": "WALK"})
                if verbose: print_at_bat_log(log)
                return 'WALK', log
        if strikes >= 3:
            if make_logs:
                log.append({"Pitch #": pitch_num, "Terminal": "K"})
                if verbose: print_at_bat_log(log)
                return 'K', log

        if make_logs:
            entry = {"Pitch #": pitch_num, "Zone": zone_enc + 1, "Outcome": outcome_label}
            if verbose_audit:
                entry["Pitch Cluster"] = pitch_cluster_label
            log.append(entry)

        pitch_num += 1
