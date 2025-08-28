# mc_matchup.py
# Full script: Monte Carlo matchup, ALL-BOOKS best-price odds pull + Kelly sizing, settlement,
# and QUIET output limited to: (1) hit-count probability table, (2) settlement card.

import os, sys, glob, time, types, joblib, numpy as np, pandas as pd
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.naive_bayes import CategoricalNB
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple

import AtBatSim
from baseball_utils import *
from General_Initialization import map_description_to_simple  # your mapper

# ===== Config =====
MIN_LOOKUP_SUPPORT = 3
MIN_LOOKUP_DOMINANCE = 0.70
FOUL_STREAK_CAP = 10

def load_pack(path: str):
    return types.SimpleNamespace(**joblib.load(path))



# ---------- Silent prints ----------
@contextmanager
def suppress_stdout_stderr():
    """Temporarily silence stdout/stderr (for noisy builders, reports, etc.)."""
    with open(os.devnull, 'w') as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = devnull, devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

# ---------- Helpers ----------
def _encode_count(balls, strikes):
    table = {(0,0):0,(0,1):1,(0,2):2,(1,0):3,(1,1):4,(1,2):5,(2,0):6,(2,1):7,(2,2):8,(3,0):9,(3,1):10,(3,2):11}
    return table.get((int(balls), int(strikes)))

def _lookup_is_trustworthy(counter: Counter, *, min_support=MIN_LOOKUP_SUPPORT, min_dom=MIN_LOOKUP_DOMINANCE):
    total = sum(counter.values())
    if total < min_support:
        return False
    top = max(counter.values())
    return (top / total) >= min_dom

def _get_nonempty_df(obj, primary_attr: str, fallback_attr: str):
    df = getattr(obj, primary_attr, None)
    if df is None or (hasattr(df, "empty") and df.empty):
        df = getattr(obj, fallback_attr, None)
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    return df

def _argmax_counter(counter_dict):
    return max(counter_dict.items(), key=lambda kv: (kv[1], kv[0]))[0]

def _hybrid_predict_row_audit(key, lookup_table, nb_model, class_labels, stage_label, audit, force_model=False):
    if (not force_model) and (key in lookup_table) and (len(lookup_table[key]) > 0):
        bucket = lookup_table[key]
        if _lookup_is_trustworthy(bucket):
            pred = _argmax_counter(bucket)
            audit["lookup_hits"][stage_label] += 1
            return int(pred), "lookup"
        audit["lookup_rejects"][stage_label] += 1
    probs = nb_model.predict_proba([list(key)])[0]
    pred = int(np.random.choice(class_labels, p=probs))
    audit["nb_uses"][stage_label] += 1
    return pred, "model"

def _hybrid_pitch_predict_audit(rows, nb_model, lookup_table, class_labels, audit, force_model=False):
    out = []
    for row in rows:
        key = tuple(int(v) for v in row)
        pred, _ = _hybrid_predict_row_audit(key, lookup_table, nb_model, class_labels, "pitch", audit, force_model)
        out.append(pred)
    return out

def _predict_zone_hybrid_audit(rows, nb_model, lookup_table, class_labels, audit, force_model=False):
    out = []
    for row in rows:
        key = tuple(int(v) for v in row)
        pred, _ = _hybrid_predict_row_audit(key, lookup_table, nb_model, class_labels, "zone", audit, force_model)
        out.append(pred)
    return out

def _predict_outcome_audit(pitch_cluster_enc, zone_enc, count_enc, nb_model, lookup_table, class_labels, audit, force_model=False):
    c, z, k = int(pitch_cluster_enc), int(zone_enc), int(count_enc)
    if not force_model:
        for key in ((c, z, k), (c, z), (c, k), (c,), (z, k), (z,)):
            if key in lookup_table and len(lookup_table[key]) > 0:
                bucket = lookup_table[key]
                if _lookup_is_trustworthy(bucket):
                    audit["lookup_hits"]["outcome"] += 1
                    return int(_argmax_counter(bucket)), "lookup"
                else:
                    audit["lookup_rejects"]["outcome"] += 1
                break
    probs = nb_model.predict_proba([[c, z, k]])[0]
    pred = int(np.random.choice(class_labels, p=probs))
    audit["nb_uses"]["outcome"] += 1
    return pred, "model"

# ----- shared resolver: use hitter.xba as per-PA hit probability for RP/EX -----
def resolve_pa_hit_prob_from_xba(h):
    val = getattr(h, "xba", None)
    if val is None:
        raise ValueError(f"{getattr(h, 'full_upper','Hitter')}: xba is missing; expected xba == per-PA hit prob.")
    return float(val)

# ===== PA simulator (audited) =====
def simulate_at_bat_between_AUDIT(
    hitter, pitcher,
    nb_pitch_model, pitch_lookup_table, pitch_class_labels,
    nb_zone_model,  zone_lookup_table,  zone_class_labels,
    *,
    cluster_encoder=None,
    MAX_PITCHES_PER_PA=30,
    MAX_SECONDS_PER_PA=3.0,
    print_every=False,
    global_audit=None
):
    t0 = time.perf_counter()

    outcome_encoder      = hitter.outcome_encoder
    nb_outcome_model     = hitter.nb_outcome_model
    outcome_lookup_table = hitter.outcome_lookup_table
    outcome_class_labels = hitter.outcome_class_labels
    xba_lookup_table     = getattr(hitter, "xba_lookup_table")
    global_bip_xba       = float(getattr(hitter, "global_bip_xba"))

    h_name = getattr(hitter, "full_lower", f"{getattr(hitter,'first_lower','?')} {getattr(hitter,'last_lower','?')}")
    p_name = getattr(pitcher, "full_lower", f"{getattr(pitcher,'first_lower','?')} {getattr(pitcher,'last_lower','?')}")

    df = _get_nonempty_df(pitcher, "pitcher_data_arch", "pitcher_data")
    if df is None:
        raise ValueError("Pitcher has no pitcher_data_arch/pitcher_data.")

    hitter_hand = AtBatSim._resolve_hitter_stand(hitter, pitcher)
    if global_audit is not None:
        global_audit["hand_resolution"][f"{h_name} vs {p_name}"].append(hitter_hand)
    assert hitter_hand in ("L", "R"), f"Resolved invalid hitter_hand={hitter_hand!r}"

    stand_enc = hitter.stand_encoder.transform([hitter_hand])[0]
    arch_enc  = int(getattr(hitter, "arch_enc"))

    balls = strikes = 0
    pitch_num = 1
    foul_streak = 0
    force_model_outcome = False

    while True:
        if pitch_num > MAX_PITCHES_PER_PA:
            global_audit["aborts"]["max_pitches"] += 1
            return "ABORT_MAX_PITCHES"
        if (time.perf_counter() - t0) > MAX_SECONDS_PER_PA:
            global_audit["aborts"]["timeout"] += 1
            return "ABORT_TIMEOUT"

        count_enc = _encode_count(balls, strikes)

        pitch_global_id = _hybrid_pitch_predict_audit(
            [[int(stand_enc), int(count_enc), int(arch_enc)]],
            nb_pitch_model, pitch_lookup_table, pitch_class_labels,
            audit=global_audit, force_model=False
        )[0]

        zone_enc = _predict_zone_hybrid_audit(
            [[int(pitch_global_id), int(count_enc), int(stand_enc)]],
            nb_model=nb_zone_model, lookup_table=zone_lookup_table, class_labels=zone_class_labels,
            audit=global_audit, force_model=False
        )[0]

        outcome_enc, src = _predict_outcome_audit(
            pitch_cluster_enc=pitch_global_id, zone_enc=zone_enc, count_enc=count_enc,
            nb_model=nb_outcome_model, lookup_table=outcome_lookup_table,
            class_labels=outcome_class_labels, audit=global_audit, force_model=force_model_outcome
        )
        try:
            raw_label = str(outcome_encoder.inverse_transform([outcome_enc])[0])
        except Exception:
            raw_label = str(outcome_enc)
        simple = map_description_to_simple(raw_label)

        if simple == "unknown":
            probs = nb_outcome_model.predict_proba([[int(pitch_global_id), int(zone_enc), int(count_enc)]])[0]
            sampled = int(np.random.choice(outcome_class_labels, p=probs))
            try:
                raw_label = str(outcome_encoder.inverse_transform([sampled])[0])
            except Exception:
                raw_label = str(sampled)
            simple = map_description_to_simple(raw_label)
            if simple == "unknown":
                simple = "strike"

        if simple == "bip":
            xba = AtBatSim.predict_xba(pitch_global_id, zone_enc, count_enc, xba_lookup_table, global_fallback=global_bip_xba)
            xba = float(np.clip(xba, 0.0, 1.0))
            global_audit["xba_samples"].append(xba)
            global_audit["bip_events"] += 1
            if np.random.rand() < xba:
                global_audit["hits_from_bip"] += 1
                return "HIT"
            else:
                return "OUT"

        if simple == "hbp":
            return "HBP"

        prev = (balls, strikes)
        if simple == "ball":
            balls += 1
        elif simple == "strike":
            strikes += 1
        elif simple == "foul":
            if strikes < 2:
                strikes += 1
            if strikes == 2:
                foul_streak += 1
                if foul_streak >= FOUL_STREAK_CAP and not force_model_outcome:
                    force_model_outcome = True
                    global_audit["foul_breaker_trips"] += 1
        else:
            strikes += 1

        if simple != "foul" or strikes < 2:
            foul_streak = 0
            force_model_outcome = False

        if balls >= 4: return "WALK"
        if strikes >= 3: return "K"

        progressed = (prev != (balls, strikes)) or (simple in ("bip","hbp")) or (simple == "foul" and prev[1] == 2)
        if not progressed:
            global_audit["aborts"]["no_progress"] += 1
            return "ABORT_NO_PROGRESS"

        pitch_num += 1

# ===== Build models from pitcher DF =====
def build_models_from_pitcher_df_AUDIT(pitcher_pack):
    df = getattr(pitcher_pack, "pitcher_data_arch", None)
    if df is None or len(df) == 0:
        raise ValueError("pitcher_data_arch missing from pack. Export packs with include_full_df=True.")

    # ---------- PITCH MODEL ----------
    need_cols_pitch = ["stand_enc", "count_enc", "arch_enc", "pitch_cluster_enc"]
    missing_pitch = [c for c in need_cols_pitch if c not in df.columns]
    if missing_pitch:
        raise ValueError(f"pitcher_data_arch missing columns for pitch model: {missing_pitch}")

    pitch_df = df[need_cols_pitch].copy()
    for c in need_cols_pitch:
        pitch_df[c] = pd.to_numeric(pitch_df[c], errors="coerce")
    dropped_pitch_before = len(pitch_df)
    pitch_df = pitch_df.dropna(subset=need_cols_pitch)
    kept_pitch = len(pitch_df)

    if kept_pitch == 0:
        nb_pitch = CategoricalNB()
        X_dummy = np.array([[0, 0, 0], [1, 1, 1]], dtype=int)
        y_dummy = np.array([0, 1], dtype=int)
        nb_pitch.fit(X_dummy, y_dummy)
        pitch_lookup = defaultdict(Counter)
        pitch_classes = nb_pitch.classes_
    else:
        Xp = pitch_df[["stand_enc", "count_enc", "arch_enc"]].astype(int).values
        yp = pitch_df["pitch_cluster_enc"].astype(int).values
        pitch_lookup = defaultdict(Counter)
        for x, y in zip(Xp, yp):
            pitch_lookup[tuple(x)][int(y)] += 1
        nb_pitch = CategoricalNB().fit(Xp, yp)
        pitch_classes = nb_pitch.classes_

    # ---------- ZONE MODEL ----------
    if "zone" not in df.columns:
        raise ValueError("pitcher_data_arch missing 'zone' for zone model.")

    dfz = df.copy()
    dfz = dfz[dfz["zone"].notna() & dfz["zone"].isin(range(1, 15))].copy()
    dfz["zone_enc"] = (pd.to_numeric(dfz["zone"], errors="coerce") - 1).astype("Int64")

    need_cols_zone = ["pitch_cluster_enc", "count_enc", "stand_enc", "zone_enc"]
    missing_zone = [c for c in need_cols_zone if c not in dfz.columns]
    if missing_zone:
        raise ValueError(f"pitcher_data_arch missing columns for zone model: {missing_zone}")

    for c in ["pitch_cluster_enc", "count_enc", "stand_enc"]:
        dfz[c] = pd.to_numeric(dfz[c], errors="coerce")

    dropped_zone_before = len(dfz)
    dfz = dfz.dropna(subset=need_cols_zone)
    kept_zone = len(dfz)

    if kept_zone == 0:
        nb_zone = CategoricalNB()
        X_dummy = np.array([[0, 0, 0], [1, 1, 1]], dtype=int)
        y_dummy = np.array([0, 1], dtype=int)
        nb_zone.fit(X_dummy, y_dummy)
        zone_lookup = defaultdict(Counter)
        zone_classes = nb_zone.classes_
    else:
        Xz = dfz[["pitch_cluster_enc", "count_enc", "stand_enc"]].astype(int).values
        yz = dfz["zone_enc"].astype(int).values
        zone_lookup = defaultdict(Counter)
        for x, y in zip(Xz, yz):
            zone_lookup[tuple(x)][int(y)] += 1
        nb_zone = CategoricalNB().fit(Xz, yz)
        zone_classes = nb_zone.classes_

    return nb_pitch, pitch_lookup, nb_pitch.classes_, nb_zone, zone_lookup, nb_zone.classes_

# ===== Monte Carlo with PA + source auditing =====
def simulate_total_hits_AUDIT_V2(
    hitter, pitcher, num_trials,
    nb_pitch_model, pitch_lookup_table, pitch_class_labels,
    nb_zone_model,  zone_lookup_table,  zone_class_labels,
    *,
    print_every=False,
    seed=123
):
    np.random.seed(seed)

    results = []
    global_audit = {
        "lookup_hits": {"pitch":0,"zone":0,"outcome":0},
        "lookup_rejects": {"pitch":0,"zone":0,"outcome":0},
        "nb_uses": {"pitch":0,"zone":0,"outcome":0},
        "bip_events": 0,
        "hits_from_bip": 0,
        "xba_samples": [],
        "foul_breaker_trips": 0,
        "aborts": {"max_pitches":0,"timeout":0,"no_progress":0},
        "hand_resolution": defaultdict(list),
        "pa_tally": {"sp":0,"rp":0,"extras":0,"total":0},
        "bf_tally": {"sp":0,"rp":0,"extras":0,"total":0},
        "ip_sp_samples": [],
        "bf_sp_samples": [],
        "bf_rp_samples": [],
        "bf_extras_samples": [],
    }

    p_rp_ex = resolve_pa_hit_prob_from_xba(hitter)
    spot = int(getattr(hitter, "most_recent_spot"))
    is_home = True

    team_woba = pd.DataFrame({
        "Team": ["CHC","NYY","TOR","LAD","ARI","BOS","DET","NYM","MIL","SEA","PHI","HOU","STL","OAK","ATL","SDP","TBR","BAL","MIN","MIA","TEX","CIN","SFG","CLE","LAA","WSN","KCR","PIT","CHW","COL"],
        "wOBA": [0.333,0.337,0.328,0.334,0.329,0.328,0.322,0.317,0.313,0.319,0.323,0.318,0.312,0.323,0.311,0.307,0.316,0.314,0.312,0.309,0.298,0.313,0.302,0.296,0.311,0.305,0.298,0.285,0.293,0.296]
    })
    team_to_abbr = {
        "Angels":"LAA","Astros":"HOU","Athletics":"OAK","Blue Jays":"TOR","Braves":"ATL","Brewers":"MIL",
        "Cardinals":"STL","Cubs":"CHC","Diamondbacks":"ARI","Dodgers":"LAD","Giants":"SFG","Guardians":"CLE",
        "Mariners":"SEA","Marlins":"MIA","Mets":"NYM","Nationals":"WSN","Orioles":"BAL","Padres":"SDP",
        "Phillies":"PHI","Pirates":"PIT","Rangers":"TEX","Rays":"TBR","Reds":"CIN","Red Sox":"BOS",
        "Rockies":"COL","Royals":"KCR","Tigers":"DET","Twins":"MIN","White Sox":"CWS","Yankees":"NYY"
    }
    hitter_abbr = team_to_abbr[hitter.team_name]
    team_woba_val = float(team_woba.loc[team_woba["Team"] == hitter_abbr, "wOBA"].values[0])

    IP_model = pitcher.IPLinReg
    BF_model = pitcher.poisson_model
    ip_sigma = float(pitcher.ip_std)

    BF_PER_OUT = np.array(getattr(pitcher, "bf_per_out"), dtype=float)
    HOME_EXTRAS = getattr(pitcher, "home_IP_extras", [])
    AWAY_EXTRAS = getattr(pitcher, "away_IP_extras", [])
    P_EXTRAS    = float(getattr(pitcher, "prob_extra_innings", 0.09) or 0.09)

    def round_to_thirds(ip): return round(ip * 3) / 3

    for t in range(num_trials):
        exp_ip = float(IP_model.predict([[team_woba_val]])[0])
        sim_ip = round_to_thirds(np.random.normal(exp_ip, ip_sigma))
        sim_ip = float(np.clip(sim_ip, 0.0, 9.0))
        global_audit["ip_sp_samples"].append(sim_ip)

        exp_bf = float(BF_model.predict(pd.DataFrame({"ip":[sim_ip]}))[0])
        sim_bf = int(np.random.poisson(exp_bf))
        sim_bf = max(0, sim_bf)
        global_audit["bf_sp_samples"].append(sim_bf)

        full_cycles = sim_bf // 9
        remainder   = sim_bf % 9
        pa_vs_sp    = full_cycles + (1 if spot <= remainder else 0)
        pa_vs_sp    = int(max(0, pa_vs_sp))

        hits_vs_sp = 0
        for i in range(pa_vs_sp):
            res = simulate_at_bat_between_AUDIT(
                hitter, pitcher,
                nb_pitch_model, pitch_lookup_table, pitch_class_labels,
                nb_zone_model,  zone_lookup_table,  zone_class_labels,
                cluster_encoder=getattr(hitter, "cluster_encoder", None),
                MAX_PITCHES_PER_PA=30, MAX_SECONDS_PER_PA=3.0,
                print_every=False, global_audit=global_audit
            )
            if res == "HIT": hits_vs_sp += 1

        hitter_win = float(getattr(hitter, "winning_pct_value"))
        pitcher_win= float(getattr(pitcher, "winning_pct_value"))

        if is_home:
            prob_not_hitting_9th = hitter_win / (hitter_win + pitcher_win + 1e-9)
            bat_9th = np.random.random() > prob_not_hitting_9th
            relief_ip = (9 - sim_ip) if bat_9th else (8 - sim_ip)
        else:
            relief_ip = 9 - sim_ip

        relief_ip = max(0.0, float(relief_ip))
        n = int(relief_ip); frac = relief_ip - n
        if 0.3 <= frac < 0.5: relief_ip = n + 0.1
        elif 0.5 <= frac < 0.7: relief_ip = n + 0.2
        else: relief_ip = float(n)

        def outs_from_ip(ip: float) -> int:
            whole, frac = divmod(round(ip*10), 10)
            return whole*3 + (2 if frac==2 else 1 if frac==1 else 0)

        outs_req = outs_from_ip(relief_ip)
        if outs_req <= 0:
            bp_bf = 0
        else:
            samples = np.random.choice(BF_PER_OUT, size=outs_req, replace=True)
            if samples.size > 0 and samples[-1] == 0.5:
                new = np.random.choice(BF_PER_OUT)
                while new == 0.5:
                    new = np.random.choice(BF_PER_OUT)
                samples[-1] = new
            bp_bf = int(max(0, round(samples.sum())))
        global_audit["bf_rp_samples"].append(bp_bf)

        next_spot = (sim_bf % 9) + 1
        pa_vs_rp = sum(1 for i in range(bp_bf) if ((next_spot + i - 1) % 9 + 1) == spot)
        pa_vs_rp = int(max(0, pa_vs_rp))

        hits_vs_rp = int(np.random.binomial(n=pa_vs_rp, p=p_rp_ex))

        hits_ex = 0; pa_vs_ex = 0; ex_bf = 0
        if np.random.random() < float(getattr(pitcher, "prob_extra_innings", P_EXTRAS)):
            pool = HOME_EXTRAS if (not is_home) else AWAY_EXTRAS
            if pool:
                extra_ip = float(np.random.choice(pool))
                whole, frac = divmod(round(extra_ip*10), 10)
                ex_outs = whole*3 + (2 if frac==2 else 1 if frac==1 else 0)
                if ex_outs > 0:
                    ex_samples = np.random.choice(BF_PER_OUT, size=ex_outs, replace=True)
                    if ex_samples.size > 0 and ex_samples[-1] == 0.5:
                        new = np.random.choice(BF_PER_OUT)
                        while new == 0.5:
                            new = np.random.choice(BF_PER_OUT)
                        ex_samples[-1] = new
                    ex_bf = int(max(0, round(ex_samples.sum())))
                    nxt = ((sim_bf + bp_bf) % 9) + 1
                    pa_vs_ex = sum(1 for i in range(ex_bf) if ((nxt + i - 1) % 9 + 1) == spot)
                    hits_ex = int(np.random.binomial(pa_vs_ex, p_rp_ex))
        global_audit["bf_extras_samples"].append(ex_bf)

        total_pa = pa_vs_sp + pa_vs_rp + pa_vs_ex
        total_hits = hits_vs_sp + hits_vs_rp + hits_ex
        results.append(int(total_hits))

        global_audit["pa_tally"]["sp"] += pa_vs_sp
        global_audit["pa_tally"]["rp"] += pa_vs_rp
        global_audit["pa_tally"]["extras"] += pa_vs_ex
        global_audit["pa_tally"]["total"] += total_pa

        global_audit["bf_tally"]["sp"] += sim_bf
        global_audit["bf_tally"]["rp"] += bp_bf
        global_audit["bf_tally"]["extras"] += ex_bf
        global_audit["bf_tally"]["total"] += (sim_bf + bp_bf + ex_bf)

    return results, global_audit

# ===== Convenience builders =====
def build_models_from_pitcher_name(pitcher_last_lower: str, hitter_last_lower: str | None = None):
    """
    Tries, in order:
      1) packs/pitcher_<pitcher>__<hitter>.joblib       (if hitter provided)
      2) packs/pitcher_<pitcher>.joblib                 (single-name)
      3) any packs/pitcher_<pitcher>__*.joblib          (first match)
    """
    p = pitcher_last_lower.strip().lower()
    h = (hitter_last_lower or "").strip().lower()

    if h:
        exact = os.path.join("packs", f"pitcher_{p}__{h}.joblib")
        if os.path.exists(exact):
            pitcher = load_pack(exact)
            (nb_pitch_model, pitch_lookup_table, pitch_class_labels,
             nb_zone_model,  zone_lookup_table,  zone_class_labels) = build_models_from_pitcher_df_AUDIT(pitcher)
            return pitcher, nb_pitch_model, pitch_lookup_table, pitch_class_labels, nb_zone_model, zone_lookup_table, zone_class_labels

    single = os.path.join("packs", f"pitcher_{p}.joblib")
    if os.path.exists(single):
        pitcher = load_pack(single)
        (nb_pitch_model, pitch_lookup_table, pitch_class_labels,
         nb_zone_model,  zone_lookup_table,  zone_class_labels) = build_models_from_pitcher_df_AUDIT(pitcher)
        return pitcher, nb_pitch_model, pitch_lookup_table, pitch_class_labels, nb_zone_model, zone_lookup_table, zone_class_labels

    pattern = os.path.join("packs", f"pitcher_{p}__*.joblib")
    matches = sorted(glob.glob(pattern))
    if matches:
        pitcher = load_pack(matches[0])
        (nb_pitch_model, pitch_lookup_table, pitch_class_labels,
         nb_zone_model,  zone_lookup_table,  zone_class_labels) = build_models_from_pitcher_df_AUDIT(pitcher)
        return pitcher, nb_pitch_model, pitch_lookup_table, pitch_class_labels, nb_zone_model, zone_lookup_table, zone_class_labels

    raise FileNotFoundError(
        f"No pitcher pack found for pitcher='{pitcher_last_lower}'. "
        f"Tried composite with hitter='{hitter_last_lower}', single-name, and any composite."
    )

def load_hitter_by_name(hitter_last_lower: str):
    return load_pack(f"packs/hitter_{hitter_last_lower.lower()}.joblib")

# ---------- main Monte Carlo (non-audit path) ----------
def simulate_total_hits(hitter, pitcher, num_trials,
                        nb_pitch_model, pitch_lookup_table, pitch_class_labels,
                        nb_zone_model,  zone_lookup_table,  zone_class_labels,
                        verbose=False):

    def round_to_thirds(ip): return round(ip * 3) / 3

    hit_results = []
    is_home = True

    team_woba = pd.DataFrame({
        "Team": ["CHC","NYY","TOR","LAD","ARI","BOS","DET","NYM","MIL","SEA",
                 "PHI","HOU","STL","OAK","ATL","SDP","TBR","BAL","MIN","MIA",
                 "TEX","CIN","SFG","CLE","LAA","WSN","KCR","PIT","CHW","COL"],
        "wOBA": [0.333,0.337,0.328,0.334,0.329,0.328,0.322,0.317,0.313,0.319,
                 0.323,0.318,0.312,0.323,0.311,0.307,0.316,0.314,0.312,0.309,
                 0.298,0.313,0.302,0.296,0.311,0.305,0.298,0.285,0.293,0.296]
    })
    team_to_abbr = {
        "Angels":"LAA","Astros":"HOU","Athletics":"OAK","Blue Jays":"TOR","Braves":"ATL","Brewers":"MIL",
        "Cardinals":"STL","Cubs":"CHC","Diamondbacks":"ARI","Dodgers":"LAD","Giants":"SFG","Guardians":"CLE",
        "Mariners":"SEA","Marlins":"MIA","Mets":"NYM","Nationals":"WSN","Orioles":"BAL","Padres":"SDP",
        "Phillies":"PHI","Pirates":"PIT","Rangers":"TEX","Rays":"TBR","Reds":"CIN","Red Sox":"BOS",
        "Rockies":"COL","Royals":"KCR","Tigers":"DET","Twins":"MIN","White Sox":"CWS","Yankees":"NYY"
    }

    hitter_abbr = team_to_abbr[hitter.team_name]
    team_woba_val = team_woba.loc[team_woba["Team"] == hitter_abbr, "wOBA"].values[0]

    IP_model = pitcher.IPLinReg
    BF_model = pitcher.poisson_model
    ip_sigma = float(pitcher.ip_std)

    p_rp_ex = resolve_pa_hit_prob_from_xba(hitter)
    spot = int(getattr(hitter, "most_recent_spot"))

    BF_PER_OUT = np.array(getattr(pitcher, "bf_per_out"), dtype=float)
    HOME_EXTRAS = getattr(pitcher, "home_IP_extras", [])
    AWAY_EXTRAS = getattr(pitcher, "away_IP_extras", [])
    P_EXTRAS    = float(getattr(pitcher, "prob_extra_innings", 0.09) or 0.09)

    win_pct_dict = {
        hitter.team_name: float(getattr(hitter, "winning_pct_value")),
        pitcher.team:     float(getattr(pitcher,  "winning_pct_value")),
    }

    for _ in range(num_trials):
        expected_ip = float(IP_model.predict([[team_woba_val]])[0])
        simulated_ip = round_to_thirds(np.random.normal(expected_ip, ip_sigma))
        simulated_ip = float(np.clip(simulated_ip, 0.0, 9.0))

        expected_bf = float(BF_model.predict(pd.DataFrame({"ip":[simulated_ip]}))[0])
        simulated_bf = int(np.random.poisson(expected_bf))

        full_cycles = simulated_bf // 9
        remainder   = simulated_bf % 9
        pa_vs_sp    = full_cycles + (1 if spot <= remainder else 0)

        hits_vs_sp = 0
        for _ in range(pa_vs_sp):
            result, _log = AtBatSim.simulate_at_bat_between(
                hitter=hitter, pitcher=pitcher,
                nb_pitch_model=nb_pitch_model, pitch_lookup_table=pitch_lookup_table, pitch_class_labels=pitch_class_labels,
                nb_zone_model=nb_zone_model, zone_lookup_table=zone_lookup_table, zone_class_labels=zone_class_labels,
                verbose=False, verbose_audit=False
            )
            if result == "HIT":
                hits_vs_sp += 1

        def simulate_reliever_innings(simulated_ip, is_hitter_home, hitter_win_pct, pitcher_win_pct):
            if not is_hitter_home:
                relief_ip = 9 - simulated_ip
            else:
                prob_not_hitting_9th = hitter_win_pct / (hitter_win_pct + pitcher_win_pct + 1e-9)
                hits_in_9th = np.random.random() > prob_not_hitting_9th
                relief_ip = 9 - simulated_ip if hits_in_9th else 8 - simulated_ip
            relief_ip = max(0.0, relief_ip)
            n = int(relief_ip); frac = relief_ip - n
            if 0.3 <= frac < 0.5: return n + 0.1
            if 0.5 <= frac < 0.7: return n + 0.2
            return float(n)

        def outs_from_ip(ip: float) -> int:
            whole, frac = divmod(round(ip*10), 10)
            return whole*3 + (2 if frac==2 else 1 if frac==1 else 0)

        def simulate_pen_bf(ip_needed: float, bf_per_out) -> int:
            outs_req = outs_from_ip(ip_needed)
            if outs_req <= 0: return 0
            arr = np.asarray(bf_per_out, dtype=float)
            if arr.size == 0: return 0
            samples = np.random.choice(arr, size=outs_req, replace=True)
            if samples[-1] == 0.5:
                while True:
                    new = np.random.choice(arr, size=1)[0]
                    if new != 0.5:
                        samples[-1] = new
                        break
            return int(max(0, round(samples.sum())))

        def hitter_facing_relief(simulated_bf: int, lineup_spot: int, bp_bf_sim: int) -> int:
            next_spot = (simulated_bf % 9) + 1
            return sum(1 for i in range(bp_bf_sim) if ((next_spot + i - 1) % 9 + 1) == lineup_spot)

        def simulate_hits_in_extras(prob_extra_innings, is_hitter_home, hitter_spot,
                                    total_bf_pre_extras, hitter_xba,
                                    bf_per_out_dist, home_IP_extras, away_IP_extras):
            if np.random.rand() >= prob_extra_innings:
                return {'extra_happens': False, 'mcneil_hits': 0}
            extras_pool = home_IP_extras if (not is_hitter_home) else away_IP_extras
            if not extras_pool:
                return {'extra_happens': True, 'mcneil_hits': 0}

            extra_ip = float(np.random.choice(extras_pool))
            ip_int = int(extra_ip); ip_frac = extra_ip - ip_int
            if np.isclose(ip_frac, 0.33): extra_ip = ip_int + 0.1
            elif np.isclose(ip_frac, 0.67): extra_ip = ip_int + 0.2

            outs_needed = outs_from_ip(extra_ip)
            if outs_needed <= 0:
                return {'extra_happens': True, 'mcneil_hits': 0}

            arr = np.asarray(bf_per_out_dist, dtype=float)
            if arr.size == 0:
                total_bf = 0
            else:
                bf_samples = np.random.choice(arr, size=outs_needed, replace=True)
                while bf_samples[-1] == 0.5:
                    bf_samples[-1] = np.random.choice(arr)
                total_bf = int(round(bf_samples.sum()))

            next_spot = (total_bf_pre_extras % 9) + 1
            mc_ab = sum(1 for i in range(total_bf) if ((next_spot + i - 1) % 9 + 1) == hitter_spot)
            hits = np.random.binomial(mc_ab, float(hitter_xba))
            return {'extra_happens': True, 'mcneil_hits': int(hits)}

        rel_ip = simulate_reliever_innings(
            simulated_ip, True,
            win_pct_dict[hitter.team_name],
            win_pct_dict[pitcher.team]
        )
        bp_bf   = simulate_pen_bf(rel_ip, BF_PER_OUT)
        pa_vs_rp = hitter_facing_relief(simulated_bf, spot, bp_bf)
        hits_vs_rp = np.random.binomial(n=pa_vs_rp, p=p_rp_ex)

        extras = simulate_hits_in_extras(
            prob_extra_innings=P_EXTRAS,
            is_hitter_home=True,
            hitter_spot=spot,
            total_bf_pre_extras=simulated_bf + bp_bf,
            hitter_xba=p_rp_ex,
            bf_per_out_dist=BF_PER_OUT,
            home_IP_extras=HOME_EXTRAS,
            away_IP_extras=AWAY_EXTRAS
        )

        total_hits = hits_vs_sp + hits_vs_rp + extras["mcneil_hits"]
        hit_results.append(int(total_hits))

    return hit_results

# ===== Betting comparison + Kelly sizing (All books → keep lowest implied prob per side/line) =====
import datetime as dt
import requests

API_KEY  = "ea063b1f7af936815eeb5380ee5d9be0"
BASE     = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb"
REGION   = "us"
ODDSFMT  = "american"
MARKETS  = "batter_hits,batter_hits_alternate"
GAME_TIME_ET = 15
SEARCH_EVERY_MIN = (0, 30)

def mc_probabilities(results: List[int]) -> Dict[str, float]:
    n = max(1, len(results))
    counts = pd.Series(results).value_counts().to_dict()
    p0      = counts.get(0, 0) / n
    p1plus  = 1.0 - p0
    p2plus  = sum(v for k, v in counts.items() if k >= 2) / n
    p3plus  = sum(v for k, v in counts.items() if k >= 3) / n
    p_le_1  = 1.0 - p2plus
    p_le_2  = 1.0 - p3plus
    return {"P0": p0, "P1plus": p1plus, "P2plus": p2plus, "P3plus": p3plus, "P<=1": p_le_1, "P<=2": p_le_2}

# For printing the per-hit distribution table
def hits_distribution(results: List[int]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame({"hits":[0], "prob":[1.0]})
    s = pd.Series(results)
    counts = s.value_counts().sort_index()
    probs = (counts / len(s)).reset_index()
    probs.columns = ["hits", "prob"]
    return probs

def norm_name(s: str) -> str:
    if s is None: return ""
    return " ".join(str(s).strip().split()).upper()

def is_player(desc: str, full_name: str = "Aaron Judge") -> bool:
    d = norm_name(desc)
    f = norm_name(full_name)
    last = f.split()[-1]
    first = f.split()[0]
    return (f == d) or (last in d and (first in d or d.startswith(first[0] + ".") or ("," in d and d.split(",")[1].strip().startswith(first[0]))))

def american_to_decimal(american: float) -> float:
    american = float(american)
    return 1 + (american/100.0 if american > 0 else 100.0/(-american))

def implied_from_price(price, fmt: str = ODDSFMT) -> Tuple[float, float]:
    if fmt == "decimal":
        dec = float(price)
        return (1.0/dec, dec)
    dec = american_to_decimal(float(price))
    return (1.0/dec, dec)

def iso_for_date_hour_utc(d: dt.date, hour_utc: int, minute: int = 0) -> str:
    return dt.datetime(d.year, d.month, d.day, hour_utc, minute, 0).isoformat() + "Z"

def noon_to_gametime_iso_candidates(d: dt.date, game_time_et: int, is_dst=True) -> List[str]:
    utc_offset = 4 if is_dst else 5
    start_utc = 12 + utc_offset
    end_utc   = game_time_et + utc_offset
    candidates = []
    for hour in range(start_utc, end_utc + 1):
        for minute in SEARCH_EVERY_MIN:
            candidates.append(iso_for_date_hour_utc(d, hour, minute))
    return sorted(set(candidates))

def is_draftkings_book(bk: Dict[str, Any]) -> bool:
    key = (bk.get("key") or "").lower()
    title = norm_name(bk.get("title"))
    return key == "draftkings" or "DRAFTKINGS" in title

def unwrap_data(resp_json):
    if isinstance(resp_json, dict) and "data" in resp_json:
        return resp_json["data"], resp_json.get("timestamp")
    return resp_json, None

def get_events_on(d: dt.date, event_probe_iso: str):
    url = f"{BASE}/events/?apiKey={API_KEY}&date={event_probe_iso}&regions={REGION}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data, _ = unwrap_data(r.json())
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected events payload shape at {event_probe_iso}: {type(data)}")
    return data

# ---------- Keep only lowest implied prob per (event_id, side, line) ----------
def keep_lowest_odds_per_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (event_id, side, line in {0.5, 1.5, 2.5}), keep the single row with the
    LOWEST implied probability p_breakeven = 1/odds_decimal (i.e., HIGHEST decimal odds).
    """
    if df is None or df.empty:
        return df

    target_lines = {0.5, 1.5, 2.5}
    df = df[df["line"].round(3).isin(target_lines)].copy()
    if df.empty:
        return df

    if "odds_decimal" not in df.columns:
        def _american_to_decimal(american: float) -> float:
            american = float(american)
            return 1 + (american/100.0 if american > 0 else 100.0/(-american))
        df["odds_decimal"] = df["price_american"].map(_american_to_decimal).astype(float)

    df["p_breakeven"] = 1.0 / df["odds_decimal"].astype(float)

    idx = df.groupby(["event_id", "side", "line"])["p_breakeven"].idxmin()
    best = df.loc[idx.values].copy()

    # Presentation order
    order_line = {0.5: 0, 1.5: 1, 2.5: 2}
    order_side = {"Under": 0, "Over": 1}
    best["line_order"] = best["line"].map(order_line).fillna(99)
    best["side_order"] = best["side"].map(order_side).fillna(99)
    best = best.sort_values(["event_id", "line_order", "side_order", "p_breakeven"]).drop(
        columns=["line_order", "side_order"]
    )
    return best

# ---------- Any-book snapshot fetcher (noon ET → game time) ----------
def try_get_event_player_hits_anybook(
    event_id: str,
    iso_candidates: List[str],
    include_books: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Try multiple historical snapshots (noon ET → game time) for an event and return
    odds for ALL books (optionally filtered). Returns (data_dict_filtered, used_iso) or (None, last_err).
    """
    include_books_set = set(b.lower() for b in include_books) if include_books else None
    last_err = None
    for iso in iso_candidates:
        try:
            url = (
                f"{BASE}/events/{event_id}/odds"
                f"?apiKey={API_KEY}&regions={REGION}&markets={MARKETS}&date={iso}&oddsFormat={ODDSFMT}"
            )
            r = requests.get(url, timeout=30)
            if r.status_code in (404, 422):
                last_err = f"{r.status_code} @ {iso}"
                continue
            r.raise_for_status()
            data, _ = unwrap_data(r.json())
            if not isinstance(data, dict):
                last_err = f"unexpected shape @ {iso}"
                continue

            books = data.get("bookmakers", []) or []
            if include_books_set is not None:
                books = [b for b in books if (b.get("key") or "").lower() in include_books_set]

            filtered_books = []
            for b in books:
                ms = [m for m in (b.get("markets") or []) if m.get("key") in ("batter_hits", "batter_hits_alternate")]
                if ms:
                    filtered_books.append({**b, "markets": ms})

            if not filtered_books:
                last_err = f"no relevant books @ {iso}"
                continue

            return {**data, "bookmakers": filtered_books}, iso

        except Exception as e:
            last_err = f"{e} @ {iso}"
    return None, last_err

# ---------- Main: pull Aaron Judge hits lines across ALL books, keep best per side/line ----------
def pull_player_hits_lines_lowest_anybook(
    d: dt.date,
    player_full_name: str = "Aaron Judge",
    team_hint: str = "YANKEES",
    include_books: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Searches snapshots from noon ET → game time for *all* books, filters to the specified player,
    then returns ONLY the best (lowest implied prob / highest decimal odds) line per (event_id, side, line).
    """
    iso_candidates = noon_to_gametime_iso_candidates(d, GAME_TIME_ET, is_dst=True)
    events = get_events_on(d, iso_candidates[0])

    team_events = [
        e for e in events
        if team_hint in norm_name((e.get("home_team","") + " " + e.get("away_team","")))
    ]

    rows: List[Dict[str, Any]] = []
    for ev in team_events:
        eid = ev["id"]
        edata, used_iso = try_get_event_player_hits_anybook(eid, iso_candidates, include_books=include_books)
        if edata is None:
            continue

        for bk in edata.get("bookmakers", []):
            book_key = bk.get("key")
            book     = bk.get("title")
            for m in bk.get("markets", []):
                if m.get("key") not in ("batter_hits", "batter_hits_alternate"):
                    continue
                for o in (m.get("outcomes") or []):
                    if not is_player(o.get("description"), player_full_name):
                        continue
                    side  = o.get("name")
                    line  = o.get("point")
                    price = o.get("price")  # American (ODDSFMT='american')
                    if side is None or line is None or price is None:
                        continue
                    p_be, dec = implied_from_price(price, ODDSFMT)
                    rows.append({
                        "query_date": d.isoformat(),
                        "used_snapshot": used_iso,
                        "event_id": eid,
                        "home_team": ev.get("home_team"),
                        "away_team": ev.get("away_team"),
                        "book_key": book_key,
                        "book": book,
                        "player": player_full_name,
                        "market": m.get("key"),
                        "side": side,
                        "line": float(line),
                        "price_american": float(price),
                        "odds_decimal": dec,
                        "p_breakeven": p_be,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep only the best (lowest implied prob) per event/side/line
    df = keep_lowest_odds_per_line(df)
    return df

# ---------- (Legacy) DK-only puller kept for reference (unused now) ----------
def try_get_event_player_hits_dk_only(event_id: str, iso_candidates: List[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    last_err = None
    for iso in iso_candidates:
        try:
            url = (
                f"{BASE}/events/{event_id}/odds"
                f"?apiKey={API_KEY}&regions={REGION}&markets={MARKETS}&date={iso}&oddsFormat={ODDSFMT}"
            )
            r = requests.get(url, timeout=30)
            if r.status_code in (404, 422):
                last_err = f"{r.status_code} @ {iso}"
                continue
            r.raise_for_status()
            data, _ = unwrap_data(r.json())
            if not isinstance(data, dict):
                last_err = f"unexpected shape @ {iso}"
                continue
            bks = data.get("bookmakers", [])
            if not bks:
                last_err = f"empty bookmakers @ {iso}"
                continue
            dk_only = [b for b in bks if is_draftkings_book(b)]
            if not dk_only:
                last_err = f"no DraftKings @ {iso}"
                continue
            return {**data, "bookmakers": dk_only}, iso
        except Exception as e:
            last_err = f"{e} @ {iso}"
    return None, last_err

def pull_player_hits_lines_draftkings_only(d: dt.date, player_full_name="Aaron Judge", team_hint="YANKEES") -> pd.DataFrame:
    iso_candidates = noon_to_gametime_iso_candidates(d, GAME_TIME_ET, is_dst=True)
    events = get_events_on(d, iso_candidates[0])

    team_events = [
        e for e in events
        if team_hint in norm_name(e.get("home_team","") + " " + e.get("away_team",""))
    ]
    rows: List[Dict[str, Any]] = []
    for ev in team_events:
        eid = ev["id"]
        edata, used_iso = try_get_event_player_hits_dk_only(eid, iso_candidates)
        if edata is None:
            continue
        for bk in edata.get("bookmakers", []):
            for m in bk.get("markets", []):
                if m.get("key") not in ("batter_hits", "batter_hits_alternate"):
                    continue
                for o in m.get("outcomes", []):
                    if not is_player(o.get("description"), player_full_name):
                        continue
                    side  = o.get("name")
                    point = o.get("point")
                    price = o.get("price")
                    if side is None or point is None or price is None:
                        continue
                    p_be, dec = implied_from_price(price, ODDSFMT)
                    rows.append({
                        "query_date": d.isoformat(),
                        "used_snapshot": used_iso,
                        "event_id": eid,
                        "home_team": ev.get("home_team"),
                        "away_team": ev.get("away_team"),
                        "book": bk.get("title") or "DraftKings",
                        "player": player_full_name,
                        "market": m.get("key"),
                        "side": side,
                        "line": float(point),
                        "price_american": float(price),
                        "odds_decimal": dec,
                        "p_breakeven": p_be,
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["event_id", "market", "line", "side"]).reset_index(drop=True)
    return df

# ---------- Model mapping and bet sizing ----------
def model_prob_for_side_line(probs: Dict[str, float], side: str, line: float) -> Optional[float]:
    side = (side or "").strip().lower()
    if side == "over":
        if line == 0.5: return probs["P1plus"]
        if line == 1.5: return probs["P2plus"]
        if line == 2.5: return probs["P3plus"]
    elif side == "under":
        if line == 0.5: return probs["P0"]
        if line == 1.5: return probs["P<=1"]
        if line == 2.5: return probs["P<=2"]
    return None

def kelly_fraction(p: float, dec_odds: float) -> float:
    b = dec_odds - 1.0
    q = 1.0 - p
    numer = b * p - q
    if numer <= 0: return 0.0
    return numer / b

def expected_value_per_dollar(p: float, dec_odds: float) -> float:
    return p * (dec_odds - 1.0) - (1.0 - p)

def hits_pmf_from_results(results: List[int]) -> Dict[int, float]:
    """Empirical pmf over hits from MC results."""
    if not results:
        return {0: 1.0}
    s = pd.Series(results).value_counts().sort_index()
    n = float(len(results))
    return {int(k): float(v / n) for k, v in s.items()}

def bet_win_indicator(h: int, side: str, line: float) -> int:
    side = (side or "").strip().lower()
    if side == "over":
        # wins if hits >= ceil(line)
        return int(h >= int(line + 0.5))
    elif side == "under":
        # wins if hits <= floor(line)
        return int(h <= int(line - 0.5))
    else:
        return 0

def build_payoff_matrix(market_df: pd.DataFrame,
                        pmf: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Returns:
      A : shape (H, M) per-$1 payoffs for each outcome h and bet m
      p : shape (H,)  outcome probabilities
      hs: list of outcome hit counts corresponding to rows of A/p
    """
    hs = sorted(pmf.keys())
    p  = np.array([pmf[h] for h in hs], dtype=float)
    M  = len(market_df)
    A  = np.zeros((len(hs), M), dtype=float)
    dec = market_df["odds_decimal"].astype(float).values

    for j, (_, r) in enumerate(market_df.iterrows()):
        side = r["side"]; line = float(r["line"])
        win_pay = dec[j] - 1.0
        lose_pay = -1.0
        for i, h in enumerate(hs):
            A[i, j] = win_pay if bet_win_indicator(h, side, line) else lose_pay
    return A, p, hs

def project_simplex_with_box(x: np.ndarray, s_cap: float) -> np.ndarray:
    """
    Project x >= 0 with sum(x) <= s_cap (simplex-like with cap).
    If sum <= cap, only clip negatives. Otherwise, Euclidean projection to the capped simplex.
    """
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s <= s_cap:
        return x
    # Project to {x>=0, sum=x_cap}
    # Classic projection: sort, find threshold
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - s_cap))[0][-1]
    theta = (cssv[rho] - s_cap) / (rho + 1.0)
    return np.maximum(x - theta, 0.0)

def joint_kelly_optimize(A: np.ndarray,
                         p: np.ndarray,
                         *,
                         exposure_cap: float = 0.02,
                         max_iter: int = 200,
                         tol: float = 1e-9,
                         step_shrink: float = 0.5) -> np.ndarray:
    """
    Solve: maximize sum_h p[h] * log(1 + (A x)_h)
    subject to x >= 0, sum(x) <= exposure_cap (fractions of bankroll).

    Returns x (fractions of bankroll per bet) at (near) full Kelly.
    """
    H, M = A.shape
    x = np.zeros(M, dtype=float)  # safe start

    for _ in range(max_iter):
        Ax = A @ x
        # Ensure feasibility domain: 1 + Ax_h > 0
        if np.any(1.0 + Ax <= 1e-12):
            # back off x if needed (very conservative)
            alpha = 0.5
            while np.any(1.0 + (A @ (alpha * x)) <= 1e-12) and alpha > 1e-6:
                alpha *= 0.5
            x *= alpha
            Ax = A @ x

        w = p / (1.0 + Ax)            # shape (H,)
        grad = A.T @ w                 # shape (M,)

        # Simple diagonal (negative) Hessian approx for damping:
        # H_diag ≈ sum_h p[h] * (A[h,m]^2) / (1+Ax_h)^2  (note: actual Hessian is negative definite)
        Hdiag = (A * (p / (1.0 + Ax)**2)[:, None]).pow(2).sum(axis=0) if hasattr(A, "pow") \
                else np.sum((A**2) * (p / (1.0 + Ax)**2)[:, None], axis=0)
        # Newton-like step with damping
        delta = grad / (Hdiag + 1e-9)

        # Line search with projection
        t = 1.0
        improved = False
        base = np.sum(p * np.log(1.0 + Ax))
        for _ls in range(20):
            x_trial = project_simplex_with_box(x + t * delta, exposure_cap)
            Ax_trial = A @ x_trial
            if np.all(1.0 + Ax_trial > 1e-12):
                val_trial = np.sum(p * np.log(1.0 + Ax_trial))
                if val_trial > base + 1e-12:
                    x = x_trial
                    improved = True
                    break
            t *= step_shrink

        if not improved:
            # try a small gradient step if Newton didn't move
            t = 1e-2
            x_trial = project_simplex_with_box(x + t * grad, exposure_cap)
            Ax_trial = A @ x_trial
            if np.all(1.0 + Ax_trial > 1e-12):
                val_trial = np.sum(p * np.log(1.0 + Ax_trial))
                if val_trial <= base + 1e-12:
                    break
                x = x_trial
            else:
                break

        # convergence check (norm of projected grad)
        if np.linalg.norm(grad, ord=np.inf) < tol:
            break
    return x


def compare_model_vs_dk_and_size_bets(
    results: List[int],
    target_date: dt.date,
    player_full_name: str = "Aaron Judge",
    team_hint: str = "YANKEES",
    bankroll: float = 1_000.0,
    kelly_scale: float = 0.5,
    save_csv_path: Optional[str] = None,
    verbose: bool = False,
    # control: cap total fraction at risk across all bets for this player
    exposure_cap_fraction: float = 0.02,
) -> pd.DataFrame:
    """
    Pull best prices across ALL books, then compute **joint Kelly** stakes across
    all (side, line) bets for this player, using the hits PMF from 'results'.

    Adds an edge haircut (shrink model prob toward market).
    Returns columns:
      - p_model (after haircut), edge, EV_per_$1
      - kelly_full (joint fraction), kelly_scaled
      - stake_full (= bankroll * kelly_full), stake_scaled (= stake_full * kelly_scale)
    """
    # 1) MC → exact hits pmf
    pmf = hits_pmf_from_results(results)

    # 2) Market pull (best per side/line)
    market_df = pull_player_hits_lines_lowest_anybook(
        target_date,
        player_full_name=player_full_name,
        team_hint=team_hint,
        include_books=None
    )
    if market_df is None or market_df.empty:
        if verbose:
            print(f"[WARN] No hitter-hits lines found for {player_full_name} on {target_date}.")
        return market_df

    # 3) Compute model win prob per bet (+ haircut) and assemble rows
    probs_dict = mc_probabilities(results)  # compute once
    haircut = 0.5  # trust 50% of model edge; tune as needed

    rows: List[Dict[str, Any]] = []
    for _, r in market_df.iterrows():
        side = r["side"]
        line = float(r["line"])
        dec  = float(r["odds_decimal"])
        p_be = float(r["p_breakeven"])  # market-implied probability

        # model win prob for this (side, line)
        p_model = model_prob_for_side_line(probs_dict, side, line)
        if p_model is None:
            continue

        # --- Edge haircut: shrink toward market ---
        p_model = p_be + haircut * (p_model - p_be)
        p_model = float(np.clip(p_model, 0.0, 1.0))  # numeric safety

        edge = p_model - p_be
        ev1  = expected_value_per_dollar(p_model, dec)

        rows.append({
            **r.to_dict(),
            "p_model": p_model,
            "edge": edge,
            "EV_per_$1": ev1,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # 4) Build payoff matrix vs. outcomes; run joint Kelly
    A, p, _hs = build_payoff_matrix(df, pmf)
    x_full = joint_kelly_optimize(A, p, exposure_cap=exposure_cap_fraction)

    # 5) Emit stakes/columns compatible with the rest of your pipeline
    df["kelly_full"]   = x_full
    df["kelly_scaled"] = x_full * float(kelly_scale)
    df["stake_full"]   = np.round(bankroll * df["kelly_full"].values, 2)
    df["stake_scaled"] = np.round(bankroll * df["kelly_scaled"].values, 2)

    # Sort for readability
    df = df.sort_values(["line", "side", "odds_decimal"], ascending=[True, True, False]).reset_index(drop=True)

    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
        if verbose:
            print(f"[OK] Saved joint-Kelly bet card → {save_csv_path}")

    if verbose:
        total_frac = float(df["kelly_full"].sum())
        print(f"[INFO] Joint Kelly across {len(df)} bets: total fraction = {total_frac:.4f} "
              f"(cap {exposure_cap_fraction:.4f}); haircut={haircut:.2f}, kelly_scale={kelly_scale:.2f}")

    return df


    # 4) Build payoff matrix vs. hits outcomes and run joint Kelly
    A, p, hs = build_payoff_matrix(df, pmf)
    # Full-Kelly fractions per bet (subject to cap)
    x_full = joint_kelly_optimize(A, p, exposure_cap=exposure_cap_fraction)

    # 5) Produce output columns consistent with your pipeline
    df["kelly_full"]   = x_full
    df["kelly_scaled"] = x_full * float(kelly_scale)
    df["stake_full"]   = np.round(bankroll * df["kelly_full"].values, 2)
    df["stake_scaled"] = np.round(bankroll * df["kelly_scaled"].values, 2)

    # Optional: sort for readability
    df = df.sort_values(["line", "side", "odds_decimal"], ascending=[True, True, False]).reset_index(drop=True)

    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
        if verbose:
            print(f"[OK] Saved joint-Kelly bet card → {save_csv_path}")

    # Helpful debug (optional)
    if verbose:
        total_frac = float(df["kelly_full"].sum())
        print(f"[INFO] Joint Kelly placed across {len(df)} bets. Total fraction = {total_frac:.4f} "
              f"(cap {exposure_cap_fraction:.4f}).")

    return df


# ===== Settle bets given actual hits =====
def _bet_wins(hits: int, side: str, line: float) -> bool:
    side = (side or "").strip().lower()
    if side == "over":
        needed = int(line + 0.5)
        return hits >= needed
    elif side == "under":
        cap = int(line - 0.5)
        return hits <= cap
    else:
        raise ValueError(f"Unknown side: {side!r}")

def settle_bet_card(
    bet_card: pd.DataFrame,
    hits_actual: int,
    *,
    bankroll_start: float = 1_000.0,
    stake_col: str = "stake_full",
    dedupe: bool = True,
    dedupe_keys: tuple = ("side", "line", "odds_decimal"),
    verbose: bool = False,
) -> pd.DataFrame:
    if bet_card is None or bet_card.empty:
        if verbose:
            print("[WARN] settle_bet_card: empty bet_card.")
        return bet_card

    df = bet_card.copy()

    if dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=list(dedupe_keys), keep="first").reset_index(drop=True)
        after = len(df)
        if verbose and after < before:
            print(f"[INFO] Deduped bets: {before} → {after} (by {dedupe_keys})")

    needed = {"side", "line", "odds_decimal", stake_col}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"bet_card missing required columns: {missing}")

    wins, profits, payouts, stakes_used = [], [], [], []
    for _, r in df.iterrows():
        s   = float(r[stake_col])
        dec = float(r["odds_decimal"])
        side = r["side"]
        line = float(r["line"])

        stake = max(0.0, s)
        win = _bet_wins(hits_actual, side, line)
        profit = stake * (dec - 1.0) if win else -stake
        payout_gross = stake * dec if win else 0.0

        wins.append(win)
        profits.append(round(profit, 2))
        payouts.append(round(payout_gross, 2))
        stakes_used.append(round(stake, 2))

    df["win"] = wins
    df["stake_used"] = stakes_used
    df["profit"] = profits
    df["payout_gross"] = payouts

    total_staked = round(float(df["stake_used"].sum()), 2)
    total_profit = round(float(df["profit"].sum()), 2)
    ending_bankroll = round(bankroll_start + total_profit, 2)
    roi = round((total_profit / total_staked) if total_staked > 0 else 0.0, 4)

    if verbose:
        print("\n===== SETTLEMENT =====")
        print(f"Actual hits: {hits_actual}")
        print(f"Bets placed: {len(df)} (stake column: {stake_col}, dedupe={dedupe})")
        print(f"Total staked: ${total_staked:,.2f}")
        print(f"Net profit:   ${total_profit:,.2f}   (ROI: {roi*100:.2f}%)")
        print(f"Ending bankroll: ${ending_bankroll:,.2f}  (start ${bankroll_start:,.2f})")

    return df

"""
# ---------- Quiet main: print ONLY hit-count probabilities and settlement card ----------
if __name__ == "__main__":
    
    
    # ---- Configure matchup/day and settlement inputs ----
    target_date = dt.date(2025, 7, 1)
    actual_hits_today = 2           # <-- fill with actual hits from the box score
    bankroll = 1_000.0
    kelly_scale = 0.5               # use 0.5 if you want half-Kelly stakes on the card; settle uses 'stake_full' by default

    # ---- Run sim + build bet card quietly ----
    with suppress_stdout_stderr():
        hitter_pack = load_pack("packs/hitter_judge.joblib")
        pitcher_pack = load_pack("packs/pitcher_gausman__judge.joblib")

        nb_pitch_model, pitch_lookup_table, pitch_class_labels, \
        nb_zone_model,  zone_lookup_table,  zone_class_labels = build_models_from_pitcher_df_AUDIT(pitcher_pack)

        results, audit = simulate_total_hits_AUDIT_V2(
            hitter=hitter_pack, pitcher=pitcher_pack, num_trials=10000,
            nb_pitch_model=nb_pitch_model, pitch_lookup_table=pitch_lookup_table, pitch_class_labels=pitch_class_labels,
            nb_zone_model=nb_zone_model, zone_lookup_table=zone_lookup_table, zone_class_labels=zone_class_labels,
            print_every=False, seed=7
        )

        bet_card = compare_model_vs_dk_and_size_bets(
            results=results,
            target_date=target_date,
            player_full_name="Aaron Judge",
            team_hint="YANKEES",
            bankroll=bankroll,
            kelly_scale=kelly_scale,
            save_csv_path=None,
            verbose=False
        )

    # ---- (1) Print probability for each exact number of hits ----
    dist = hits_distribution(results)
    print(dist.to_string(index=False, float_format="%.4f"))

    # ---- (2) Print only the final settlement card (no extra logs) ----
    settled = settle_bet_card(
        bet_card,
        hits_actual=actual_hits_today,
        bankroll_start=bankroll,
        stake_col="stake_full",   # or "stake_scaled" if you want to settle the half-Kelly stakes instead
        dedupe=True,
        verbose=False
    )
    cols = ["market","side","line","price_american","odds_decimal","stake_used","win","profit","payout_gross"]
    print(settled[cols].to_string(index=False))

    """

from datetime import date

def run_player_hits_workflow(
    hitter_last: str,
    pitcher_last: str,
    target_date: date,
    actual_hits: int,
    *,
    num_trials: int = 10_000,
    bankroll: float = 1_000.0,
    kelly_scale: float = 0.5,
    stake_col: str = "stake_scaled",   # "stake_full" for full Kelly
    dedupe: bool = True,
    seed: int = 7,
    print_header: bool = True,
):
    """
    End-to-end:
      1) load packs and build pitcher models
      2) run MC audit sim to get results
      3) compute hit distribution table
      4) pull ALL-BOOKS best-price lines for the date, size with Kelly
      5) settle using actual_hits
      6) print a header: 'HITTER vs. PITCHER MM/DD'

    Returns:
      dist_df   -> columns ['hits','prob'] with .attrs['match_header']
      settled   -> settlement card with .attrs['match_header']
      pnl       -> float net profit (= sum(profit))
    """
    # --- load packs & models ---
    hitter = load_hitter_by_name(hitter_last)
    pitcher, nb_pitch_model, pitch_lookup_table, pitch_class_labels, \
        nb_zone_model, zone_lookup_table, zone_class_labels = build_models_from_pitcher_name(pitcher_last, hitter_last)

    # --- run MC (audited path for stable outputs) ---
    results, _audit = simulate_total_hits_AUDIT_V2(
        hitter=hitter,
        pitcher=pitcher,
        num_trials=num_trials,
        nb_pitch_model=nb_pitch_model, pitch_lookup_table=pitch_lookup_table, pitch_class_labels=pitch_class_labels,
        nb_zone_model=nb_zone_model,   zone_lookup_table=zone_lookup_table,   zone_class_labels=zone_class_labels,
        print_every=False,
        seed=seed,
    )

    # --- distribution: P(hits = k) for all observed k ---
    counts = pd.Series(results).value_counts().sort_index()
    k_max = int(counts.index.max()) if not counts.empty else 0
    n = max(1, len(results))
    dist_df = pd.DataFrame({
        "hits": list(range(0, k_max + 1)),
        "prob": [round(float(counts.get(k, 0) / n), 4) for k in range(0, k_max + 1)],
    })

    # --- Best-price lines + Kelly sizing ---
    hitter_name_full = getattr(hitter, "full_upper", hitter_last.upper())
    pitcher_name_full = getattr(pitcher, "full_upper", pitcher_last.upper())
    team_hint = (getattr(hitter, "team_name", "") or "").upper()

    bet_card = compare_model_vs_dk_and_size_bets(
        results=results,
        target_date=target_date,
        player_full_name=hitter_name_full.title() if hitter_name_full.isupper() else hitter_name_full,
        team_hint=team_hint,
        bankroll=bankroll,
        kelly_scale=kelly_scale,
        save_csv_path=None
    )

    # Graceful empty (no lines found): return dist and empty settlement with pnl=0
    if bet_card is None or bet_card.empty:
        header = f"{hitter_name_full} vs. {pitcher_name_full} {target_date:%m/%d}"
        if print_header:
            print(header)
        dist_df.attrs["match_header"] = header
        empty_settlement = pd.DataFrame(columns=["market","side","line","price_american","odds_decimal","stake_used","win","profit","payout_gross"])
        empty_settlement.attrs["match_header"] = header
        return dist_df, empty_settlement, 0.0

    # --- settle using actual hits ---
    settled = settle_bet_card(
        bet_card,
        hits_actual=actual_hits,
        bankroll_start=bankroll,
        stake_col=stake_col,   # "stake_scaled" for half Kelly
        dedupe=dedupe
    )

    # --- header & attach metadata, print header once ---
    header = f"{hitter_name_full} vs. {pitcher_name_full} {target_date:%m/%d}"
    if print_header:
        print(header)

    dist_df.attrs["match_header"] = header
    settled.attrs["match_header"] = header

    # --- final P/L (sum of net profits) ---
    pnl = float(settled["profit"].sum()) if not settled.empty else 0.0

    # Trim settlement to the requested columns for clean display
    keep_cols = ["market","side","line","price_american","odds_decimal","stake_used","win","profit","payout_gross"]
    settled_out = settled[keep_cols] if all(c in settled.columns for c in keep_cols) else settled

    return dist_df, settled_out, pnl


