# Hitter_Objects_parallel.py
import os, sys, traceback, math
import multiprocessing as mp

# ----- keep native libs single-threaded inside each worker -----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Prefer RAM-backed shared memory for joblib scratch (much faster than /tmp)
_JOBLIB_TMP = "/dev/shm/joblib" if os.path.isdir("/dev/shm") else "/tmp/joblib"
os.makedirs(_JOBLIB_TMP, exist_ok=True)
os.environ["JOBLIB_TEMP_FOLDER"] = _JOBLIB_TMP
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 32))

# Force 'spawn' before other heavy imports
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import pandas as pd
import joblib
from joblib import Parallel, delayed

from pybaseball import playerid_lookup
from baseball_utils import *

from Hitter_Class import Hitter
from General_Initialization import classify_archetype  # used to annotate `batting`

# Optional: enforce BLAS=1 even if a wheel ignores env
try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    class _noop:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
    def threadpool_limits(*_, **__):  # type: ignore
        return _noop()

# ---------------- Core loaders (lazy per-process cache) ----------------
_CACHE = None  # (batter_data, pitcher_data, batting, encoder_data, bat_stats_2025)

def load_core_frames():
    batter_data    = pd.read_pickle("batter_data.pkl")
    pitcher_data   = pd.read_pickle("pitcher_data.pkl")  # not used here, but harmless
    batting        = pd.read_pickle("batting_2025.pkl")
    encoder_data   = pd.read_pickle("encoder_data.pkl")
    bat_stats_2025 = pd.read_pickle("bat_stats_2025.pkl")

    # Precompute hitter archetypes on batting
    if "Hitter_Archetype" not in batting.columns:
        batting["Hitter_Archetype"] = batting.apply(classify_archetype, axis=1)

    return batter_data, pitcher_data, batting, encoder_data, bat_stats_2025

def _get_frames_cached():
    global _CACHE
    if _CACHE is None:
        _CACHE = load_core_frames()
    return _CACHE

# ---------------- Pack exporter ----------------
def export_hitter_pack(hitter, path, ensure_metadata: bool = False):
    # Optional metadata refresh
    if ensure_metadata:
        try:
            hitter._get_player_metadata()
        except Exception:
            pass

    # --- hard checks so we fail fast if something is off ---
    enc = getattr(hitter, "cluster_encoder", None)
    if enc is None or not hasattr(enc, "classes_") or len(getattr(enc, "classes_", [])) == 0:
        raise RuntimeError("cluster_encoder missing or unfitted on hitter.")

    gmm_models = getattr(hitter, "gmm_models", None)
    if not isinstance(gmm_models, dict) or len(gmm_models) == 0:
        raise RuntimeError("gmm_models missing on hitter (required by Pitcher_Class).")

    pack = {
        "gmm_models": hitter.gmm_models,
        "importances": hitter.importances,
        "cluster_encoder": hitter.cluster_encoder,
        "stand_encoder": hitter.stand_encoder,
        "arch_encoder": getattr(hitter, "arch_encoder", None),
        "outcome_encoder": hitter.outcome_encoder,
        "nb_outcome_model": hitter.nb_outcome_model,
        "outcome_lookup_table": hitter.outcome_lookup_table,
        "outcome_class_labels": hitter.outcome_class_labels,
        "xba_lookup_table": hitter.xba_lookup_table,
        "global_bip_xba": float(getattr(hitter, "global_bip_xba", 0.300) or 0.300),
        "arch_enc": int(hitter.arch_enc),
        "full_lower": getattr(hitter, "full_lower", None),
        "full_upper": getattr(hitter, "full_upper", None),
        "team_name": getattr(hitter, "team_name", None),
        "xba": float(getattr(hitter, "xba", getattr(hitter, "global_bip_xba", 0.300)) or 0.300),
        "most_recent_spot": int(getattr(hitter, "most_recent_spot", 3) or 3),
        "winning_pct_value": float(getattr(hitter, "winning_pct_value", 0.5) or 0.5),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pack, path, compress=3)

    n_classes = len(getattr(enc, "classes_", []))
    print(f"[OK] Saved hitter pack → {path} | enc_classes={n_classes} | gmm_keys={len(gmm_models)}")

# ---------------- Hitter factory ----------------
def create_hitter_for_storage(
    first: str,
    last: str,
    team_name: str | None,
    team_id: int | None,
    batter_data: pd.DataFrame,
    encoder_data: pd.DataFrame,
    bat_stats_2025: pd.DataFrame,
    batting: pd.DataFrame,
    player_id: int | None = None,
) -> Hitter:
    """
    Creates a Hitter object with optional explicit player_id.
    If player_id is provided, bypasses pybaseball lookup.
    team_name/team_id may be None → we default to 'Unknown'/0.
    """
    tn = team_name if team_name is not None else "Unknown"
    tid = int(team_id) if team_id is not None else 0

    if player_id is None:
        lookup = playerid_lookup(last.lower(), first.lower())
        if lookup.empty or "key_mlbam" not in lookup.columns:
            raise ValueError(f"pybaseball lookup failed for {first} {last}")
        pid = int(lookup["key_mlbam"].values[0])
    else:
        pid = int(player_id)

    with threadpool_limits(limits=1):
        hitter = Hitter(
            first_lower=first.lower(),
            first_upper=first.title(),
            last_lower=last.lower(),
            last_upper=last.title(),
            team_name=tn,
            team_id=tid,
            batter_data=batter_data,
            encoder_data=encoder_data,
            bat_stats_2025=bat_stats_2025,
            batting=batting,
            player_id=pid,
        )
    return hitter

# ---------------- Safe runner (worker entry) ----------------
def safe_build_and_export_parallel(player_spec: dict) -> tuple[bool, str]:
    """
    player_spec requires keys:
      first, last, player_id, pack
    Optional:
      team_name, team_id
    Returns (ok, label)
    """
    label = f"{player_spec.get('first','?')} {player_spec.get('last','?')}"
    try:
        batter_data, pitcher_data, batting, encoder_data, bat_stats_2025 = _get_frames_cached()

        hitter = create_hitter_for_storage(
            first=player_spec["first"],
            last=player_spec["last"],
            team_name=player_spec.get("team_name"),
            team_id=player_spec.get("team_id"),
            batter_data=batter_data,
            encoder_data=encoder_data,
            bat_stats_2025=bat_stats_2025,
            batting=batting,
            player_id=player_spec.get("player_id"),
        )
        export_hitter_pack(hitter, os.path.join("packs", player_spec["pack"]))
        return True, label
    except Exception as e:
        print(f"[SKIP] {label}: {e.__class__.__name__}: {e}")
        if os.environ.get("VERBOSE_ERRORS", ""):
            traceback.print_exc()
        return False, label

# ---------------- Main (parallel build) ----------------
if __name__ == "__main__":
    # ~30 players (team_name/team_id optional here)
    players = [
        
      {"first":"Ozzie","last":"Albies","player_id":645277,"pack":"hitter_albies.joblib","team_name":"Braves","team_id":144},
      {"first":"Jonathan","last":"India","player_id":663697,"pack":"hitter_india.joblib","team_name":"Reds","team_id":113},
      {"first":"Otto","last":"Lopez","player_id":672640,"pack":"hitter_lopez.joblib","team_name":"Marlins","team_id":146},
      {"first":"Nathaniel","last":"Lowe","player_id":663993,"pack":"hitter_lowe.joblib","team_name":"Rangers","team_id":140},
      {"first":"Masyn","last":"Winn","player_id":677950,"pack":"hitter_winn.joblib","team_name":"Cardinals","team_id":138},
      {"first":"Christian","last":"Walker","player_id":592253,"pack":"hitter_walker.joblib","team_name":"Diamondbacks","team_id":109},
      {"first":"Nolan","last":"Schanuel","player_id":694384,"pack":"hitter_schanuel.joblib","team_name":"Angels","team_id":108},
      {"first":"Josh","last":"Smith","player_id":669960,"pack":"hitter_smith.joblib","team_name":"Rangers","team_id":140},
      {"first":"Xander","last":"Bogaerts","player_id":593428,"pack":"hitter_bogaerts.joblib","team_name":"Padres","team_id":135},
      {"first":"William","last":"Contreras","player_id":661388,"pack":"hitter_contreras.joblib","team_name":"Brewers","team_id":158},
      {"first":"Ernie","last":"Clement","player_id":676391,"pack":"hitter_clement.joblib","team_name":"Blue Jays","team_id":141},
      {"first":"Gleyber","last":"Torres","player_id":650402,"pack":"hitter_torres.joblib","team_name":"Yankees","team_id":147},
      {"first":"Bryan","last":"Reynolds","player_id":668804,"pack":"hitter_reynolds.joblib","team_name":"Pirates","team_id":134},
      {"first":"Brendan","last":"Donovan","player_id":680974,"pack":"hitter_donovan.joblib","team_name":"Cardinals","team_id":138},
      {"first":"Dansby","last":"Swanson","player_id":621020,"pack":"hitter_swanson.joblib","team_name":"Cubs","team_id":112},
      {"first":"Lawrence","last":"Butler","player_id":686657,"pack":"hitter_butler.joblib","team_name":"Athletics","team_id":133},
      {"first":"Brice","last":"Turang","player_id":668930,"pack":"hitter_turang.joblib","team_name":"Brewers","team_id":158},
      {"first":"Gavin","last":"Sheets","player_id":657757,"pack":"hitter_sheets.joblib","team_name":"White Sox","team_id":145},
      {"first":"Josh","last":"Naylor","player_id":647304,"pack":"hitter_naylor.joblib","team_name":"Guardians","team_id":114},
      {"first":"Alec","last":"Burleson","player_id":676475,"pack":"hitter_burleson.joblib","team_name":"Cardinals","team_id":138},
      {"first":"Jarren","last":"Duran","player_id":680776,"pack":"hitter_duran.joblib","team_name":"Red Sox","team_id":111},
      {"first":"Andy","last":"Pages","player_id":681624,"pack":"hitter_pages.joblib","team_name":"Dodgers","team_id":119},
      {"first":"CJ","last":"Abrams","player_id":682928,"pack":"hitter_abrams.joblib","team_name":"Nationals","team_id":120},
      {"first":"Gunnar","last":"Henderson","player_id":683002,"pack":"hitter_henderson.joblib","team_name":"Orioles","team_id":110}


    ]

    # Worker sizing for c2d-standard-32
    CPU = os.cpu_count() or 32
    N_WORKERS = 16   # leave a couple for OS/Jupyter
    # If each build is quick (<1s), batching helps; else keep batch_size=1
    TYPICAL_TASK_SEC = 1.5
    BATCH_SIZE = 1 if TYPICAL_TASK_SEC >= 1.0 else max(2, min(16, math.ceil(1.0 / max(0.05, TYPICAL_TASK_SEC))))

    print(f"Building {len(players)} hitter packs with {N_WORKERS} workers…")
    print(f"JOBLIB_TEMP_FOLDER = {_JOBLIB_TMP}, batch_size = {BATCH_SIZE}")

    results = Parallel(
        n_jobs=N_WORKERS,
        backend="loky",
        prefer="processes",
        verbose=10,
        pre_dispatch=f"{2 * max(1, N_WORKERS)}",
        inner_max_num_threads=1,
        batch_size=BATCH_SIZE,
        max_nbytes=None,
        temp_folder=_JOBLIB_TMP,
    )(
        delayed(safe_build_and_export_parallel)(p) for p in players
    )

    successes = sum(1 for ok, _ in results if ok)
    failures  = len(results) - successes

    print(f"\n[SUMMARY] Success: {successes}  |  Failed/Skipped: {failures}")
    if failures:
        failed_labels = [lbl for ok, lbl in results if not ok]
        print("Failed:", ", ".join(failed_labels))
