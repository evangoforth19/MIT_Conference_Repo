# build_pitcher_packs.py
# Purpose: build full Pitcher objects (linked to hitters) and export sim packs
#          that contain pitcher_data_arch + models needed by Monte Carlo.
#
# This version prints progress as each pack completes (OK/ERR + timing).
# Child processes are kept quiet; the parent streams clean status lines.

from __future__ import annotations

import os
import time
import types
import traceback
import unicodedata
import multiprocessing as mp
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple, List

# ---- keep each process single-threaded for BLAS to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pandas as pd
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================== Utilities ==============================

@contextmanager
def suppress_prints():
    """Redirect stdout/stderr to /dev/null inside the 'with' block."""
    with open(os.devnull, "w") as _devnull:
        _oldout, _olderr = os.sys.stdout, os.sys.stderr
        try:
            os.sys.stdout = os.sys.stderr = _devnull
            yield
        finally:
            os.sys.stdout, os.sys.stderr = _oldout, _olderr


def _slug_ascii(s: str) -> str:
    """Normalize accents (e.g., Sánchez -> sanchez) and lowercase for filenames."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def _task_label(t: Dict[str, Any]) -> str:
    return f"{t['first']} {t['last']} → {t['hitter_key']}"

# ============================== Core loaders ==============================

def load_core_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pitcher_data   = pd.read_pickle("pitcher_data.pkl")
    batter_data    = pd.read_pickle("batter_data.pkl")
    batting        = pd.read_pickle("batting_2025.pkl")
    encoder_data   = pd.read_pickle("encoder_data.pkl")
    bat_stats_2025 = pd.read_pickle("bat_stats_2025.pkl")

    # Import classify_archetype lazily & quietly
    with suppress_prints():
        from General_Initialization import classify_archetype

    # ensure archetype column exists (your pipeline expects it)
    if "Hitter_Archetype" not in batting.columns:
        batting["Hitter_Archetype"] = batting.apply(classify_archetype, axis=1)

    return pitcher_data, batter_data, batting, encoder_data, bat_stats_2025


def make_team_woba_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Team": [
            "CHC","NYY","TOR","LAD","ARI","BOS","DET","NYM","MIL","SEA",
            "PHI","HOU","STL","ATH","ATL","SDP","TBR","BAL","MIN","MIA",
            "TEX","CIN","SFG","CLE","LAA","WSN","KCR","PIT","CHW","COL"
        ],
        "wOBA": [
            0.333,0.337,0.328,0.334,0.329,0.328,0.322,0.317,0.313,0.319,
            0.323,0.318,0.312,0.323,0.311,0.307,0.316,0.314,0.312,0.309,
            0.298,0.313,0.302,0.296,0.311,0.305,0.298,0.285,0.293,0.296
        ],
    })

# ============================== Config ==============================

PACKS_DIR = "packs"

# Map hitter names (keys you use in TASKS) to their saved joblib filenames (exact)
HITTER_PACKS: dict[str, str] = {
    "freeman":      "hitter_freeman.joblib",
    "mcneil":       "hitter_mcneil.joblib",
    "judge":        "hitter_judge.joblib",
    "olson":        "hitter_olson.joblib",
    "semien":       "hitter_semien.joblib",
    "alonso":       "hitter_alonso.joblib",
    "castellanos":  "hitter_castellanos.joblib",
    "ozuna":        "hitter_ozuna.joblib",
    "soto":         "hitter_soto.joblib",
    "riley":        "hitter_riley.joblib",
    "adames":       "hitter_adames.joblib",
    "betts":        "hitter_betts.joblib",
    "devers":       "hitter_devers.joblib",
    "bichette":     "hitter_bichette.joblib",
    "lindor":       "hitter_lindor.joblib",
    "tucker":       "hitter_tucker.joblib",
    "turner":       "hitter_turner.joblib",
    "hoerner":      "hitter_hoerner.joblib",
    "cronenworth":  "hitter_cronenworth.joblib",  # your saved filename
    "steer":        "hitter_steer.joblib",
    "mcmahon":      "hitter_mcmahon.joblib",
    "walker":       "hitter_walker.joblib",
    "nimmo":        "hitter_nimmo.joblib",
    # newly added from your save log:
    "schwarber":    "hitter_schwarber.joblib",
    "rooker":       "hitter_rooker.joblib",
    "raleigh":      "hitter_raleigh.joblib",
    "machado":      "hitter_machado.joblib",
    "ward":         "hitter_ward.joblib",
    "kwan":         "hitter_kwan.joblib",
}

# ============================== Exporter ==============================

def export_pitcher_pack(pitcher: Any, path: str, include_full_df: bool = True) -> None:
    """
    Save everything Monte Carlo needs so it doesn't import GI at sim time:
      - pitcher_data_arch with count_enc
      - fast handedness map
      - starter IP/BF models + ip_std
      - team and winning_pct_value
      - bf_per_out (bullpen distribution)
      - extra-innings distributions + probability
      - identity fields for AtBatSim logging
    """
    df = getattr(pitcher, "pitcher_data_arch", None)

    # Ensure count_enc exists (avoid GI.encode_count later)
    if df is not None and not df.empty and "count_enc" not in df.columns:
        def _enc(b, s):
            table = {
                (0,0):0,(0,1):1,(0,2):2,(1,0):3,(1,1):4,(1,2):5,
                (2,0):6,(2,1):7,(2,2):8,(3,0):9,(3,1):10,(3,2):11
            }
            return table.get((int(b), int(s)), 0)
        df = df.copy()
        df["count_enc"] = df[["balls", "strikes"]].apply(lambda r: _enc(r["balls"], r["strikes"]), axis=1)

    # Fast-handedness lookup
    stand_map = {}
    if df is not None and not df.empty:
        stand_map = (
            df[["batter_name", "stand"]]
            .dropna()
            .assign(batter_name_lower=lambda d: d["batter_name"].str.lower())
            .drop_duplicates("batter_name_lower")
            .set_index("batter_name_lower")["stand"]
            .to_dict()
        )

    # Bullpen BF per out
    try:
        bf_per_out = list(pitcher.bf_list())
    except Exception:
        bf_per_out = []

    # Pull extras distributions from GI now (quietly), not at sim time
    with suppress_prints():
        try:
            from General_Initialization import home_IP_extras, away_IP_extras, prob_extra_innings
            home_extras = list(home_IP_extras)
            away_extras = list(away_IP_extras)
            p_extras    = float(prob_extra_innings)
        except Exception:
            home_extras, away_extras, p_extras = [], [], 0.09  # safe defaults

    pack = {
        "stand_by_batter_lower": stand_map,
        "IPLinReg": getattr(pitcher, "IPLinReg", None),
        "poisson_model": getattr(pitcher, "poisson_model", None),
        "ip_std": float(getattr(pitcher, "ip_std", 0.0) or 0.0),
        "team": getattr(pitcher, "team", None),
        "winning_pct_value": float(getattr(pitcher, "winning_pct_value", 0.5) or 0.5),

        # bullpen / extras
        "bf_per_out": bf_per_out,
        "home_IP_extras": home_extras,
        "away_IP_extras": away_extras,
        "prob_extra_innings": p_extras,

        # identity (for logs)
        "first_lower": getattr(pitcher, "first_lower", "max"),
        "last_lower":  getattr(pitcher, "last_lower",  "fried"),
        "full_lower":  getattr(pitcher, "full_lower",  "max fried"),
        "first_upper": getattr(pitcher, "first_upper", "Max"),
        "last_upper":  getattr(pitcher, "last_upper",  "Fried"),
        "full_upper":  getattr(pitcher, "full_upper",  "Max Fried"),
    }

    if include_full_df and df is not None:
        pack["pitcher_data_arch"] = df

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pack, path, compress=3)
    # Child prints are suppressed; parent will report success.

# ============================== Pitcher/Hitter factories ==============================

def create_link_hitter_for_pitcher(
    first: str, last: str, team_name: str, team_id: int,
    batter_data: pd.DataFrame, encoder_data: pd.DataFrame,
    bat_stats_2025: pd.DataFrame, batting: pd.DataFrame
) -> Any:
    """
    Build a full Hitter purely to link during Pitcher construction (for cluster space).
    """
    with suppress_prints():
        from Hitter_Class import Hitter
    return Hitter(
        first_lower=first.lower(), first_upper=first.title(),
        last_lower=last.lower(),   last_upper=last.title(),
        team_name=team_name, team_id=team_id,
        batter_data=batter_data, encoder_data=encoder_data,
        bat_stats_2025=bat_stats_2025, batting=batting,
        player_id=None,
    )


def create_pitcher_for_storage(
    first: str,
    last: str,
    team_nickname: str,
    pitcher_data: pd.DataFrame,
    batting: pd.DataFrame,
    team_woba: pd.DataFrame,
    linked_hitter,              # Hitter instance OR wrapped pack (SimpleNamespace)
    player_id: Optional[int] = None,
) -> Any:
    """Creates a full Pitcher object (runs its pipeline) with the provided linked hitter."""
    with suppress_prints():
        from Pitcher_Class import Pitcher
        return Pitcher(
            first_lower=first.lower(),
            first_upper=first.title(),
            last_lower=last.lower(),
            last_upper=last.title(),
            team=team_nickname,
            pitcher_data=pitcher_data,
            batting=batting,
            team_woba=team_woba,
            hitter=linked_hitter,          # crucial for cluster encoder space
            player_id=player_id,
        )

# ============================== Worker ==============================

def _build_one_pack(
    task: Dict[str, Any],
    pitcher_data: pd.DataFrame,
    batting: pd.DataFrame,
    team_woba: pd.DataFrame,
    packs_dir: str,
) -> Tuple[bool, str, str, float]:
    """
    Runs in a separate process. Loads the linked hitter pack locally, builds the Pitcher,
    and exports the pack.

    Returns: (ok, out_or_err, label, seconds)
    """
    label = _task_label(task)
    t0 = time.perf_counter()
    try:
        hitter_key = task["hitter_key"]
        hitter_filename = HITTER_PACKS[hitter_key]
        hitter_path = os.path.join(packs_dir, hitter_filename)
        if not os.path.exists(hitter_path):
            return (False, f"Missing hitter pack: {hitter_path}", label, time.perf_counter() - t0)

        # Load the linked hitter pack (may be dict; wrap to attribute access)
        with suppress_prints():
            hitter_obj = joblib.load(hitter_path)
        if isinstance(hitter_obj, dict):
            hitter_obj = types.SimpleNamespace(**hitter_obj)

        # Build Pitcher quietly
        with suppress_prints():
            p = create_pitcher_for_storage(
                first=task["first"],
                last=task["last"],
                team_nickname=task["team_nickname"],
                pitcher_data=pitcher_data,
                batting=batting,
                team_woba=team_woba,
                linked_hitter=hitter_obj,
                player_id=task.get("player_id"),
            )

            last_slug   = _slug_ascii(task["last"])
            hitter_slug = _slug_ascii(hitter_key)
            outname = f"pitcher_{last_slug}__{hitter_slug}.joblib"
            outpath = os.path.join(packs_dir, outname)

            export_pitcher_pack(p, outpath)

        # Double-check file exists
        if not os.path.exists(outpath):
            return (False, f"Expected output missing: {outpath}", label, time.perf_counter() - t0)

        return (True, outpath, label, time.perf_counter() - t0)

    except Exception as e:
        return (False, f"{e}\n{traceback.format_exc()}", label, time.perf_counter() - t0)

# ============================== Main ==============================

if __name__ == "__main__":
    # Load frames
    print("[BOOT] Loading core dataframes…", flush=True)
    pitcher_data, batter_data, batting, encoder_data, bat_stats_2025 = load_core_frames()
    team_woba = make_team_woba_df()
    print("[BOOT] Core dataframes ready.", flush=True)

    # Define tasks (your qualified list). team_nickname can be tweaked if needed.
    # Define tasks (your qualified list). team_nickname can be tweaked if needed.
    TASKS: List[dict[str, Any]] = [
        # Max Fried → Bo Bichette
        dict(first="Max", last="Fried", team_nickname="Yankees", hitter_key="bichette", player_id=608331),  # corrected
    
        # Tyler Anderson → Austin Riley / Marcell Ozuna / Matt Olson
        dict(first="Tyler", last="Anderson", team_nickname="Angels", hitter_key="riley",   player_id=542881),
        dict(first="Tyler", last="Anderson", team_nickname="Angels", hitter_key="ozuna",   player_id=542881),
        dict(first="Tyler", last="Anderson", team_nickname="Angels", hitter_key="olson",   player_id=542881),
    
        # Michael Lorenzen → Cal Raleigh
        dict(first="Michael", last="Lorenzen", team_nickname="Royals", hitter_key="raleigh", player_id=547179),  # corrected
    
        # Freddy Peralta → Nimmo / Lindor / McNeil / Soto / Alonso
        dict(first="Freddy", last="Peralta", team_nickname="Brewers", hitter_key="nimmo",    player_id=642547),  # corrected
        dict(first="Freddy", last="Peralta", team_nickname="Brewers", hitter_key="lindor",   player_id=642547),  # corrected
        dict(first="Freddy", last="Peralta", team_nickname="Brewers", hitter_key="mcneil",   player_id=642547),  # corrected
        dict(first="Freddy", last="Peralta", team_nickname="Brewers", hitter_key="soto",     player_id=642547),  # corrected
        dict(first="Freddy", last="Peralta", team_nickname="Brewers", hitter_key="alonso",   player_id=642547),  # corrected
    
        # Cristopher Sánchez → Jake Cronenworth / Manny Machado
        dict(first="Cristopher", last="Sánchez", team_nickname="Phillies", hitter_key="cronenworth", player_id=650911),  # corrected
        dict(first="Cristopher", last="Sánchez", team_nickname="Phillies", hitter_key="machado",     player_id=650911),  # corrected
    
        # Nick Pivetta → Kyle Schwarber / Nick Castellanos / Trea Turner
        dict(first="Nick", last="Pivetta", team_nickname="Padres", hitter_key="schwarber",   player_id=601713),  # corrected
        dict(first="Nick", last="Pivetta", team_nickname="Padres", hitter_key="castellanos", player_id=601713),  # corrected
        dict(first="Nick", last="Pivetta", team_nickname="Padres", hitter_key="turner",      player_id=601713),  # corrected
    
        # Brady Singer → Rafael Devers
        dict(first="Brady", last="Singer", team_nickname="Reds", hitter_key="devers", player_id=663903),
    
        # Kevin Gausman → Aaron Judge
        dict(first="Kevin", last="Gausman", team_nickname="Blue Jays", hitter_key="judge",  player_id=592332),   # corrected
        ]

    # Ensure packs dir exists
    packs_dir = PACKS_DIR
    os.makedirs(packs_dir, exist_ok=True)

    # Sanity check: referenced hitter packs exist
    print("[CHECK] Verifying referenced hitter packs…", flush=True)
    missing = []
    for t in TASKS:
        key = t["hitter_key"]
        fname = HITTER_PACKS.get(key)
        path = os.path.join(packs_dir, fname or "")
        if not fname or not os.path.exists(path):
            missing.append((key, fname, path))
    if missing:
        print("[WARN] Missing hitter packs for tasks:", flush=True)
        for key, fname, path in missing:
            print(f"  - key={key} file={fname} path={path}", flush=True)
        # You can choose to hard-fail here:
        # raise SystemExit("Abort: missing hitter packs.")
    else:
        print("[CHECK] All hitter packs found.", flush=True)

    # Choose workers
    n_workers = max(1, min(len(TASKS), (mp.cpu_count() or 8)))
    # Allow override via env (optional)
    try:
        n_workers = int(os.environ.get("N_JOBS", n_workers))
    except Exception:
        pass

    print(f"[INFO] Building {len(TASKS)} pitcher packs with {n_workers} parallel workers…", flush=True)
    t0 = time.time()

    # Submit tasks and stream results as they complete
    ok_count = 0
    err_count = 0
    finished = 0
    results_meta: List[Tuple[bool, str, str, float]] = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_build_one_pack, t, pitcher_data, batting, team_woba, packs_dir): _task_label(t)
            for t in TASKS
        }
        for fut in as_completed(futures):
            try:
                ok, out_or_err, label, secs = fut.result()
            except Exception as e:
                ok = False
                out_or_err = f"{e}\n{traceback.format_exc()}"
                label = futures[fut]
                secs = 0.0

            finished += 1
            results_meta.append((ok, out_or_err, label, secs))
            if ok:
                ok_count += 1
                print(f"[OK  {finished:>2}/{len(TASKS)}] {label}  →  {out_or_err}  ({secs:.1f}s)", flush=True)
            else:
                err_count += 1
                print(f"[ERR {finished:>2}/{len(TASKS)}] {label}\n{out_or_err.strip()}", flush=True)

    dt = time.time() - t0

    # Final summary
    print(f"\n[DONE] {ok_count} OK / {err_count} ERR (total {len(TASKS)}) in {dt:.1f}s", flush=True)

    # Optional: list the files we actually see on disk
    built_files = sorted(
        f for f in os.listdir(packs_dir)
        if f.startswith("pitcher_") and f.endswith(".joblib")
    )
    if built_files:
        print("\n[FILES] Built packs:")
        for f in built_files:
            print(f"  - {os.path.join(packs_dir, f)}")
    else:
        print("\n[FILES] No pitcher_*.joblib files found in packs/ (did every task fail?)")
