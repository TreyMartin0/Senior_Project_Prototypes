from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import math
import random
import pandas as pd

# Bands used for personalization based on users audiogram results
BANDS: Dict[str, List[int]] = {
    "low": [250, 500],
    "mid": [1000, 2000],
    "high": [4000, 6000, 8000],
}

#For wide conversion / required freqs
COMMON_FREQS: List[int] = sorted([f for freqs in BANDS.values() for f in freqs])

#Severity buckets that label the severity of hearing loss at each frequency
def severity_bucket(db):
    if db is None or (isinstance(db, float) and math.isnan(db)):
        return "unknown"
    elif db <= 20:
        return "normal"
    elif db <= 40:
        return "mild"
    elif db <= 55:
        return "moderate"
    elif db <= 70:
        return "mod_severe"
    else:
        return "severe"
    
#mapping from severity to how many sessions a week a module appears
BASE_WEIGHTS = {
    "SPEECH_IN_NOISE": 2,
    "CONSONANT_ID": 1,
}

# Prototype modules
MODULES = [
    "HF_FRICATIVES",  # s, f, th, sh, etc.
    "CONSONANT_ID",
    "SPEECH_IN_NOISE",
    "VOICING",
    "VOWEL_CLARITY"
]

#Data Structures
@dataclass
class EarFeatures:
    band_avgs_dbhl: Dict[str, Optional[float]]
    band_severity: Dict[str, str]
    slope_high_minus_mid: Optional[float]
    asymmetry_db: Optional[float] = None
    notch_4k: bool = False

@dataclass
class ProgramProfile:
    userid: int
    audiogramid: int
    date: str
    ear_side: str
    loss_profile: str
    features: EarFeatures
    module_plan: Dict[str, Dict[str, Any]] #weight + start params
    schedule: List[Dict[str, Any]] #week/day blocks

#helper functions
'''
Filters rows for the user, groups by audiogramid, finds most recent test,
and picks the latest one.
'''
def pick_audiogram(df, userid):
    #choose an audiogramid for a client
    sub = df[df["userid"] == userid]
    if sub.empty:
        raise ValueError(f"No rows for userid={userid}")
    
    #pull the lastest audiogram the user got
    g = sub.groupby("audiogramid")["date"].max().reset_index()
    g = g.sort_values(["date", "audiogramid"], ascending=[False, False])
    return int(g.iloc[0]["audiogramid"])


#Converts long format measurements into a wide audiogram vector
def long_conversion(df, userid, audiogramid):
    sub = df[(df["userid"] == userid) & (df["audiogramid"] == audiogramid)]
    if sub.empty:
        raise ValueError(f"No data for userid={userid}, audiogramid={audiogramid}")

    wide = (
        sub[sub["frequency"].isin(COMMON_FREQS)]
        .pivot_table(
            index=["userid", "audiogramid", "date", "side"],
            columns="frequency",
            values="level",
            aggfunc="mean"
        )
        .reset_index()
    )

    return wide



# Load raw JSON files
def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

#Feature extraction
def mean_ignore_nan(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(vals) / len(vals)) if vals else None

# Very simple 4k noise-notch detector.
#True if 4k is ≥10 dB worse than both 2k and 8k.
def detect_4k_notch(row):
    f2 = row.get(2000, float("nan"))
    f4 = row.get(4000, float("nan"))
    f8 = row.get(8000, float("nan"))

    if any(math.isnan(x) for x in [f2, f4, f8]):
        return False

    return (f4 - f2 >= 10) and (f4 - f8 >= 10)

'''
Band averages that averages the threshold for each band, then turns each band
into a bucket. The slope compute high_avg - low_avg, which determines if the hearing
loss slopes worse in the highs.
'''
def extract_ear_features(wide_row):
    band_avgs: Dict[str, Optional[float]] = {}
    band_sev: Dict[str, str] = {}

    for band, freqs in BANDS.items():
        vals = [wide_row.get(f, float("nan")) for f in freqs]
        avg = mean_ignore_nan(vals)
        band_avgs[band] = avg
        band_sev[band] = severity_bucket(avg if avg is not None else float("nan"))

    high = band_avgs["high"]
    mid = band_avgs["mid"]
    slope = None
    if high is not None and mid is not None:
        slope = float(high - mid)

    notch = detect_4k_notch(wide_row)

    return EarFeatures(
        band_avgs_dbhl=band_avgs,
        band_severity=band_sev,
        slope_high_minus_mid=slope,
        notch_4k=notch
    )


'''
Labels the ear as one of:
- Sloping high-frequency loss
- Flat loss
- Mid-dominant
- Mixed loss
In turn this influences what training modules to emphasize.
'''
def classify_loss_profile(features: EarFeatures):
    low = features.band_avgs_dbhl["low"]
    mid = features.band_avgs_dbhl["mid"]
    high = features.band_avgs_dbhl["high"]
    slope = features.slope_high_minus_mid

    # Missingness handling
    if low is None and mid is None and high is None:
        return "unknown"

    # Notch gets special label.
    if features.notch_4k:
        return "noise_notch_4k"

    # Sloping high-frequency loss
    if slope is not None and slope >= 15:
        return "sloping_high_frequency"

    # Flat-ish loss
    if low is not None and mid is not None and high is not None:
        if max(low, mid, high) - min(low, mid, high) <= 10:
            return "flat_loss"

    # Mid-dominant / cookie-bite-ish (rough proxy)
    if mid is not None and high is not None and low is not None:
        if mid - max(low, high) >= 10:
            return "mid_dominant"

    # Otherwise general
    return "mixed_loss"


def compute_asymmetry(left: EarFeatures, right: EarFeatures):
    diffs = []
    for band in ["low", "mid", "high"]:
        l = left.band_avgs_dbhl.get(band)
        r = right.band_avgs_dbhl.get(band)
        if l is None or r is None:
            continue
        diffs.append(abs(l - r))
    return float(sum(diffs)/len(diffs)) if diffs else 0.0



# Module selection + difficulty

'''
Gives weights to modules on how often they should appear per user based
on their hearing loss.
'''
def weights_for_features(profile, features: EarFeatures):
    weights = dict(BASE_WEIGHTS)

    high_sev = features.band_severity["high"]
    mid_sev = features.band_severity["mid"]
    low_sev = features.band_severity["low"]

    # Always include some transfer
    weights.setdefault("SPEECH_IN_NOISE", 2)

    # High-frequency emphasis
    if high_sev in ["moderate", "mod_severe", "severe"]:
        weights["HF_FRICATIVES"] = max(weights.get("HF_FRICATIVES", 0), 3)
        weights["CONSONANT_ID"] = max(weights.get("CONSONANT_ID", 0), 2)

    elif high_sev == "mild":
        weights["HF_FRICATIVES"] = max(weights.get("HF_FRICATIVES", 0), 2)

    # Mid-frequency emphasis
    if mid_sev in ["moderate", "mod_severe", "severe"]:
        weights["VOWEL_CLARITY"] = 2

    # Low-frequency emphasis
    if low_sev in ["moderate", "mod_severe", "severe"]:
        weights["VOICING"] = 2

    # Notch: treat similar to HF but add extra SIN
    if profile == "noise_notch_4k":
        weights["HF_FRICATIVES"] = max(weights.get("HF_FRICATIVES", 0), 3)
        weights["SPEECH_IN_NOISE"] = max(weights.get("SPEECH_IN_NOISE", 0), 3)

    # Optional general auditory processing
    if (high_sev in ["moderate", "mod_severe", "severe"]) or (mid_sev in ["moderate", "mod_severe", "severe"]):
        weights["TEMPORAL_GAPS"] = 1

    # Keep only non-zero modules
    weights = {m: w for m, w in weights.items() if w > 0}

    return weights

#Sets starting difficulty for each module
def starting_difficulty(features: EarFeatures, sin_baseline_snr50):
    params: Dict[str, Dict[str, Any]] = {}

    # Speech in noise: start easier than baseline
    if sin_baseline_snr50 is not None and not math.isnan(sin_baseline_snr50):
        start_snr = float(sin_baseline_snr50 + 4.0)  # easier by +4 dB
    else:
        start_snr = 10.0

    params["SPEECH_IN_NOISE"] = {
        "snr_db": start_snr,
        "noise_type": "speech_shaped"  # placeholder label for your app
    }

    # HF fricatives: tier determines contrast difficulty (easy/med/hard)
    hf_avg = features.band_avgs_dbhl.get("high")
    if hf_avg is None:
        tier = "easy"
    elif hf_avg >= 55:
        tier = "easy"
    elif hf_avg >= 40:
        tier = "medium"
    else:
        tier = "hard"

    params["HF_FRICATIVES"] = {"tier": tier, "talker_variability": "low"}

    # Consonant ID: start level by overall severity
    overall = mean_ignore_nan([features.band_avgs_dbhl.get("low"),
                              features.band_avgs_dbhl.get("mid"),
                              features.band_avgs_dbhl.get("high")]) or 30.0
    level = 1 if overall >= 45 else 2 if overall >= 30 else 3
    params["CONSONANT_ID"] = {"level": level}

    # Vowel clarity
    params["VOWEL_CLARITY"] = {"level": 1}

    # Voicing
    params["VOICING"] = {"level": 1}

    # Temporal gaps
    params["TEMPORAL_GAPS"] = {"gap_ms": 20}  # placeholder knob

    # Environmental sounds
    params["ENV_SOUND_CATEGORIZATION"] = {"level": 1}

    # Ensure only modules with weights (later) are used—caller can filter.
    return params


# Schedule generation

def build_module_pool(weights):
    pool = []
    for m, w in weights.items():
        pool.extend([m] * int(w))
    random.shuffle(pool)
    return pool


"""
    Generate a plan: week/day with blocks.
    Structure: warmup (2m), target (6m), transfer (2m).
    Ensures SPEECH_IN_NOISE appears as transfer unless absent.
"""
def generate_schedule(
    weights,
    start_params,
    weeks: int = 4,
    days_per_week: int = 5,
    session_minutes: int = 10,
    seed: int = 7
):
    random.seed(seed)

    # Ensure transfer exists
    if "SPEECH_IN_NOISE" not in weights:
        weights["SPEECH_IN_NOISE"] = 1

    pool = build_module_pool(weights)
    cooldown = {}  # module -> remaining cooldown sessions

    def pick_module(allowed: List[str]) -> str:
        # Prefer modules not on cooldown
        candidates = [m for m in allowed if cooldown.get(m, 0) == 0]
        if not candidates:
            candidates = allowed[:]  # fallback
        return random.choice(candidates)

    schedule: List[Dict[str, Any]] = []
    pool_idx = 0


    for w in range(1, weeks + 1):
        for d in range(1, days_per_week + 1):
            # Refresh pool index cyclically
            def next_from_pool() -> str:
                nonlocal pool_idx
                if not pool:
                    return "SPEECH_IN_NOISE"
                m = pool[pool_idx % len(pool)]
                pool_idx += 1
                return m

            # Decrement cooldowns
            for m in list(cooldown.keys()):
                cooldown[m] = max(0, cooldown[m] - 1)

            warmup = pick_module([m for m in weights.keys() if m != "SPEECH_IN_NOISE"] or ["SPEECH_IN_NOISE"])
            target = next_from_pool()
            transfer = "SPEECH_IN_NOISE"

            # Avoid target == warmup too often
            if target == warmup and len(weights) > 1:
                target = next_from_pool()

            # Put target on short cooldown (avoid repeating over and over)
            cooldown[target] = max(cooldown.get(target, 0), 1)

            blocks = [
                {"module": warmup, "minutes": 2, "params": start_params.get(warmup, {})},
                {"module": target, "minutes": 6, "params": start_params.get(target, {})},
                {"module": transfer, "minutes": 2, "params": start_params.get(transfer, {})},
            ]

            schedule.append({"week": w, "day": d, "session_minutes": session_minutes, "blocks": blocks})

    return schedule


# Top-level: build a program per ear
"""
Returns two ProgramProfile objects (left/right) when possible.
 """
def build_program_for_client(
    df,
    userid,
    weeks: int = 4,
    days_per_week: int = 5,
    sin_baseline_snr50: Optional[float] = None
):
    audiogramid = pick_audiogram(df, userid)
    wide = long_conversion(df, userid, audiogramid)

    # Build ear features
    ear_rows = {}
    for _, row in wide.iterrows():
        side = str(row["side"]).lower()
        ear_rows[side] = row

    programs: List[ProgramProfile] = []

    # Extract features for each ear
    ear_features: Dict[str, EarFeatures] = {}
    for side, row in ear_rows.items():
        feats = extract_ear_features(row)
        ear_features[side] = feats

    # Compute asymmetry if both ears present
    if "left" in ear_features and "right" in ear_features:
        asym = compute_asymmetry(ear_features["left"], ear_features["right"])
        ear_features["left"].asymmetry_db = asym
        ear_features["right"].asymmetry_db = asym

    # Build program per ear
    for side, feats in ear_features.items():
        profile = classify_loss_profile(feats)
        weights = weights_for_features(profile, feats)
        params = starting_difficulty(feats, sin_baseline_snr50=sin_baseline_snr50)

        # Filter params to only modules used
        params = {m: p for m, p in params.items() if m in weights}

        schedule = generate_schedule(
            weights=weights,
            start_params=params,
            weeks=weeks,
            days_per_week=days_per_week,
        )

        # Module plan includes weights + params so your UI can show both
        module_plan = {m: {"weight": weights[m], "start_params": params.get(m, {})} for m in weights}

        date_val = wide["date"].max()
        date_str = date_val.strftime("%Y-%m-%d") if pd.notna(date_val) else "unknown"

        programs.append(
            ProgramProfile(
                userid=int(userid),
                audiogramid=int(audiogramid),
                date=date_str,
                ear_side=side,
                loss_profile=profile,
                features=feats,
                module_plan=module_plan,
                schedule=schedule
            )
        )

    return programs


# Demo / CLI entry

def main():
    # Point this at merged+filtered long-format CSV
    df= pd.DataFrame(pd.DataFrame(load_json("test_data.json")))
    df["date"] = pd.to_datetime(df["date"], unit="ms", errors="coerce")

    # Pick a few clients to generate programs for
    sample_clients = sorted(df["userid"].unique())[:3]

    all_programs: List[Dict[str, Any]] = []
    for cid in sample_clients:
        programs = build_program_for_client(
            df=df,
            userid=int(cid), 
            weeks=4,
            days_per_week=5,
            sin_baseline_snr50=None
        )
        for p in programs:
            # Convert dataclasses cleanly to JSON
            d = asdict(p)
            all_programs.append(d)

    # Write output
    with open("training_programs.json", "w", encoding="utf-8") as f:
        json.dump(all_programs, f, indent=2)

    print(f"Wrote {len(all_programs)} ear-level programs to training_programs.json")
    # Print one example
    print(json.dumps(all_programs[0], indent=2)[:2000])


if __name__ == "__main__":
    main()