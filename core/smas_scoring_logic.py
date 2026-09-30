from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks


# ============================================================
# CONFIG
# ============================================================

JSON_PATH = "path/to/athlete.json"
EVENTS_CSV = "path/to/smasframe.csv"
IMAGE_DIR = "path/to/imagefolder"
OUTPUT_DIR = "/path/to/output"

IMAGE_FRAME_OFFSET = 1
EMA_ALPHA = 0.3

# Static webpage browsing.
BROWSE_ALL_FRAMES = False
FRAME_WINDOW = 5

JOINT_MAP = {
    "left_hip": "l_upleg",
    "right_hip": "r_upleg",
    "left_knee": "l_lowleg",
    "right_knee": "r_lowleg",
    "left_ankle": "l_foot",
    "right_ankle": "r_foot",
    "left_foot": "l_foot",
    "right_foot": "r_foot",
    "left_toe": "l_toe_tip",
    "right_toe": "r_toe_tip",
    "left_heel": "l_heel_tip",
    "right_heel": "r_heel_tip",
}

POSITION_ORDER = ["toeoff", "mvp", "lateswing", "touchdown", "midstance"]
POSITION_LABELS = {
    "toeoff": "Toe Off",
    "mvp": "MVP",
    "lateswing": "Late Swing",
    "touchdown": "Touchdown",
    "midstance": "Midstance",
}
POSITION_ALIASES = {
    "toeoff": "toeoff", "toe": "toeoff", "to": "toeoff",
    "mvp": "mvp", "maximumverticalprojection": "mvp",
    "maxverticalprojection": "mvp",
    "lateswing": "lateswing", "late": "lateswing",
    "touchdown": "touchdown", "touch": "touchdown", "td": "touchdown",
    "midstance": "midstance", "mid": "midstance",
}

# New algorithm settings.
NEW_MVP_PROMINENCE_PCT = 0.12
NEW_TOEOFF_PROMINENCE_PCT = 0.12
NEW_MIDSTANCE_PROMINENCE_PCT = 0.12
NEW_PEAK_DISTANCE = 10
NEW_PEAK_WIDTH = 1
NEW_PAIR_MIN_GAP = 2
NEW_PAIR_MAX_GAP = 40
BILATERAL_MERGE_WINDOW = 6
NEW_BOUNDARY_PEAK_WINDOW = 4
NEW_END_PEAK_SEARCH_WINDOW = 15
NEW_END_PEAK_BASELINE_WINDOW = 10
NEW_END_PEAK_PROMINENCE_MULTIPLIER = 0.60

# Old algorithm settings.
OLD_TOEOFF_PROMINENCE_PCT = 0.03
OLD_TOEOFF_NOISE_FLOOR_PCT = 0.005
OLD_TOEOFF_CONFIRM_K = 3
OLD_TOEOFF_MAX_SEARCH_PCT = 0.30
OLD_TOEOFF_REFINE_RADIUS = 4
OLD_HEEL_PROMINENCE_PCT = 0.03
OLD_TOE_PROMINENCE_PCT = 0.03
OLD_TOUCHDOWN_HEEL_OFFSET = 2
OLD_TOE_HEEL_AGREE_WINDOW = 5
OLD_MAX_STRIDE_SEARCH = 45

# Drawing colors.
COLOR_VERTICAL = (0, 190, 255)
COLOR_ACTUAL = (255, 80, 50)
COLOR_REFERENCE = (185, 90, 255)
COLOR_TEXT = (0, 0, 0)
COLOR_BOX = (255, 255, 255)
COLOR_POINT = (255, 210, 0)
COLOR_DARK = (20, 20, 20)

# SMAS thresholds.
TOEOFF_TARGET_ANGLE_DEG = 45
TOEOFF_TOLERANCE_DEG = 5
MVP_HEEL_KNEE_HEIGHT_FRAC = 0.05
LATE_SWING_KNEE_BEHIND_HIP_FRAC = 0.04
TOUCHDOWN_SHIN_AHEAD_FRAC = 0.04
TOUCHDOWN_THIGH_GAP_TARGET_DEG = 20
TOUCHDOWN_THIGH_GAP_GRACE_DEG = 5
TOUCHDOWN_THIGH_GAP_MAX_DEG = TOUCHDOWN_THIGH_GAP_TARGET_DEG + TOUCHDOWN_THIGH_GAP_GRACE_DEG
MIDSTANCE_KNEE_AHEAD_FOOT_FRAC = 0.03
TOUCHDOWN_MAX_FOOT_SPACE_MULT = 1.25
MIN_PIXEL_THRESHOLD = 8

METRIC_LABELS = {
    "toeoff_angle": "Toe off: back-heel angle",
    "mvp_shin_height": "MVP: back shin not higher than parallel",
    "late_swing_knee_behind_hip": "Late Swing: trailing knee behind hip",
    "touchdown_thigh_separation": "Touchdown: thigh gap angle",
    "touchdown_shin_angle": "Touchdown: shin angle",
    "touchdown_foot_space": "Touchdown: foot space",
    "midstance_knee_over_foot": "Midstance: knee over foot",
}
POSITION_METRICS = {
    "toeoff": ["toeoff_angle"],
    "mvp": ["mvp_shin_height"],
    "lateswing": ["late_swing_knee_behind_hip"],
    "touchdown": [
        "touchdown_thigh_separation",
        "touchdown_shin_angle",
        "touchdown_foot_space",
    ],
    "midstance": ["midstance_knee_over_foot"],
}


# ============================================================
# LOADING
# ============================================================

def clean_joint_name(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_position_name(name):
    clean = (
        str(name).strip().lower()
        .replace(" ", "").replace("_", "")
        .replace("-", "").replace("/", "")
    )
    return POSITION_ALIASES.get(clean, clean)


def load_motion_json(json_path):
    """Original joint_names/joints loader retained for standalone use."""
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if "joint_names" not in data or "joints" not in data:
        raise ValueError("JSON must contain joint_names and joints.")

    joint_names = data["joint_names"]
    joints = data["joints"]
    fps = float(data.get("fps", 1.0))

    lengths = []
    for name in joint_names:
        if name not in joints or "pixel_uv" not in joints[name]:
            raise ValueError(f"Missing pixel_uv for joint: {name}")
        lengths.append(len(joints[name]["pixel_uv"]))

    n_frames = min(lengths)
    rows = []

    for frame in range(n_frames):
        row = {
            "frame": frame,
            "time_seconds": frame / fps,
            "image_number": frame + IMAGE_FRAME_OFFSET,
            "file_name": f"{frame + IMAGE_FRAME_OFFSET:05d}.jpg",
        }
        for name in joint_names:
            key = clean_joint_name(name)
            p = joints[name]["pixel_uv"][frame]
            if p is None or len(p) < 2:
                row[f"{key}_x"] = np.nan
                row[f"{key}_y"] = np.nan
            else:
                row[f"{key}_x"] = float(p[0])
                row[f"{key}_y"] = float(p[1])
        rows.append(row)

    df = pd.DataFrame(rows)

    for canonical, actual in JOINT_MAP.items():
        actual = clean_joint_name(actual)
        for axis in ["x", "y"]:
            source = f"{actual}_{axis}"
            if source not in df.columns:
                raise ValueError(f"Required JSON joint column missing: {source}")
            df[f"{canonical}_{axis}"] = df[source]

    df["midhip_x"] = (df["left_hip_x"] + df["right_hip_x"]) / 2
    df["midhip_y"] = (df["left_hip_y"] + df["right_hip_y"]) / 2
    return df, joint_names, fps, data


def load_events_csv(csv_path):
    events = pd.read_csv(csv_path)
    events.columns = [str(c).strip().lower() for c in events.columns]

    if "frame" not in events.columns or "position" not in events.columns:
        raise ValueError("Events CSV requires frame and position columns.")

    events["frame"] = events["frame"].astype(int)
    events["position_norm"] = events["position"].map(normalize_position_name)
    events = events[events["position_norm"].isin(POSITION_ORDER)]
    events = events.sort_values("frame").reset_index(drop=True)

    if "gait_cycle" in events.columns:
        events["gait_cycle"] = events["gait_cycle"].astype(int)
        return events
    if "cycle" in events.columns:
        events["gait_cycle"] = events["cycle"].astype(int)
        return events

    cycle_ids = []
    current_cycle = 1
    seen = set()
    last_frame = None

    for _, row in events.iterrows():
        position = row["position_norm"]
        frame = int(row["frame"])
        if position == "toeoff" and seen:
            current_cycle += 1
            seen = set()
        if position in seen:
            current_cycle += 1
            seen = set()
        if last_frame is not None and frame < last_frame:
            current_cycle += 1
            seen = set()
        cycle_ids.append(current_cycle)
        seen.add(position)
        last_frame = frame

    events["gait_cycle"] = cycle_ids
    return events


# ============================================================
# FEATURES
# ============================================================

def fill_nan(values):
    return (
        pd.Series(np.asarray(values, dtype=float))
        .interpolate(limit_direction="both")
        .ffill().bfill()
        .to_numpy(dtype=float)
    )


def ema(values, alpha=EMA_ALPHA):
    values = fill_nan(values)
    if len(values) == 0:
        return values.copy()
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def normalize_one_signal(values):
    values = fill_nan(values)
    q75 = np.percentile(values, 75)
    top = values[values > q75]
    reference = float(np.mean(top)) if len(top) else float(np.mean(values))
    return reference - values, reference


def derivative_n(values, n=1):
    out = fill_nan(values)
    for _ in range(n):
        out = np.gradient(out)
    return out


def detrend_signal(values):
    values = fill_nan(values)
    if len(values) < 2:
        return values.copy()
    idx = np.arange(len(values))
    slope, intercept = np.polyfit(idx, values, 1)
    trend = slope * idx + intercept
    return values - (trend - intercept)


def calculate_three_point_angle(ax, ay, bx, by, cx, cy):
    bax = np.asarray(ax, float) - np.asarray(bx, float)
    bay = np.asarray(ay, float) - np.asarray(by, float)
    bcx = np.asarray(cx, float) - np.asarray(bx, float)
    bcy = np.asarray(cy, float) - np.asarray(by, float)
    dot = bax * bcx + bay * bcy
    denom = np.sqrt(bax**2 + bay**2) * np.sqrt(bcx**2 + bcy**2)
    result = np.full(len(dot), np.nan)
    valid = np.isfinite(dot) & np.isfinite(denom) & (denom > 0)
    result[valid] = np.degrees(np.arccos(np.clip(dot[valid] / denom[valid], -1, 1)))
    return result


def add_feature_columns_for_prediction(df, alpha=EMA_ALPHA):
    df = df.copy()
    features = {}
    keypoints = sorted({
        col[:-2] for col in df.columns
        if col.endswith("_x") and f"{col[:-2]}_y" in df.columns
    })

    for kp in keypoints:
        x = fill_nan(df[f"{kp}_x"])
        y = fill_nan(df[f"{kp}_y"])
        y_norm, y_ref = normalize_one_signal(y)
        y_norm_ema = ema(y_norm, alpha)
        y_velocity_raw = derivative_n(y_norm_ema, 1)
        y_velocity = ema(y_velocity_raw, alpha)
        y_detrended = detrend_signal(y)
        y_detrended_ema = ema(y_detrended, alpha)
        y_detrended_norm, _ = normalize_one_signal(y_detrended)

        features[f"{kp}_x_ema"] = ema(x, alpha)
        features[f"{kp}_y_reference_normalized"] = y_norm
        features[f"{kp}_y_reference_normalized_ref"] = np.full(len(df), y_ref)
        features[f"{kp}_y_reference_normalized_ema"] = y_norm_ema
        features[f"{kp}_y_velocity_raw"] = y_velocity_raw
        features[f"{kp}_y_velocity"] = y_velocity
        features[f"{kp}_y_acceleration"] = derivative_n(y_velocity, 1)
        features[f"{kp}_y_detrended_ema"] = y_detrended_ema
        features[f"{kp}_y_detrended_norm_ema"] = ema(y_detrended_norm, alpha)

    df = pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)
    df["left_leg_angle"] = calculate_three_point_angle(
        df["left_hip_x"], df["left_hip_y"],
        df["left_knee_x"], df["left_knee_y"],
        df["left_foot_x"], df["left_foot_y"],
    )
    df["right_leg_angle"] = calculate_three_point_angle(
        df["right_hip_x"], df["right_hip_y"],
        df["right_knee_x"], df["right_knee_y"],
        df["right_foot_x"], df["right_foot_y"],
    )
    return df


# ============================================================
# PREDICTION
# ============================================================

def prominent_maxima(frames, values, prominence_pct):
    frames = np.asarray(frames, dtype=int)
    values = fill_nan(values)
    if len(values) == 0 or len(frames) != len(values):
        return []
    signal_range = float(np.ptp(values))
    if signal_range <= 0:
        return []

    required = prominence_pct * signal_range
    candidates = []

    if len(values) >= 3:
        peaks, props = find_peaks(
            values,
            prominence=required,
            distance=NEW_PEAK_DISTANCE,
            width=NEW_PEAK_WIDTH,
        )
        for j, idx in enumerate(peaks):
            candidates.append({
                "index": int(idx),
                "frame": int(frames[idx]),
                "prominence": float(props["prominences"][j]),
            })

    if len(values) >= 2:
        boundary_end = min(len(values), NEW_BOUNDARY_PEAK_WINDOW + 1)
        following = values[1:boundary_end]
        following = following[np.isfinite(following)]
        if np.isfinite(values[0]) and len(following):
            prominence = float(values[0] - np.min(following))
            if values[0] > np.max(following) and prominence >= required:
                candidates.append({"index": 0, "frame": int(frames[0]), "prominence": prominence})

    if len(values) >= 3:
        search_start = max(0, len(values) - NEW_END_PEAK_SEARCH_WINDOW)
        region = values[search_start:]
        candidate_idx = search_start + int(np.argmax(region))
        baseline_start = max(0, candidate_idx - NEW_END_PEAK_BASELINE_WINDOW)
        preceding = values[baseline_start:candidate_idx]
        preceding = preceding[np.isfinite(preceding)]
        if np.isfinite(values[candidate_idx]) and len(preceding):
            candidate_value = float(values[candidate_idx])
            local_prominence = float(candidate_value - np.min(preceding))
            after = values[candidate_idx + 1:]
            after = after[np.isfinite(after)]
            no_higher_after = len(after) == 0 or candidate_value >= np.max(after)
            if (
                local_prominence >= required * NEW_END_PEAK_PROMINENCE_MULTIPLIER
                and no_higher_after
            ):
                candidates.append({
                    "index": int(candidate_idx),
                    "frame": int(frames[candidate_idx]),
                    "prominence": local_prominence,
                })

    best = {}
    for candidate in candidates:
        idx = candidate["index"]
        if idx not in best or candidate["prominence"] > best[idx]["prominence"]:
            best[idx] = candidate

    ordered = sorted(best.values(), key=lambda x: (-x["prominence"], x["index"]))
    selected = []
    for candidate in ordered:
        if all(abs(candidate["index"] - chosen["index"]) >= NEW_PEAK_DISTANCE for chosen in selected):
            selected.append(candidate)

    return [
        {"frame": int(c["frame"]), "prominence": float(c["prominence"])}
        for c in sorted(selected, key=lambda x: x["index"])
    ]


def merge_bilateral(candidates):
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda x: x["frame"])
    groups = [[candidates[0]]]
    for candidate in candidates[1:]:
        if candidate["frame"] - groups[-1][-1]["frame"] <= BILATERAL_MERGE_WINDOW:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    return [int(max(group, key=lambda x: x["prominence"])["frame"]) for group in groups]


def detect_new_mvp_frames(df):
    frames = df["frame"].to_numpy(int)
    candidates = []
    for side in ["left", "right"]:
        candidates.extend(prominent_maxima(frames, df[f"{side}_toe_y_velocity"], NEW_MVP_PROMINENCE_PCT))
    return merge_bilateral(candidates)


def detect_new_toeoff_frames(df):
    frames = df["frame"].to_numpy(int)
    candidates = []
    for side in ["left", "right"]:
        candidates.extend(prominent_maxima(frames, df[f"{side}_knee_y_reference_normalized_ema"], NEW_TOEOFF_PROMINENCE_PCT))
    return merge_bilateral(candidates)


def detect_new_midstance_frames(df):
    frames = df["frame"].to_numpy(int)
    candidates = []
    for side in ["left", "right"]:
        candidates.extend(prominent_maxima(frames, df[f"{side}_heel_y_reference_normalized_ema"], NEW_MIDSTANCE_PROMINENCE_PCT))
    return merge_bilateral(candidates)


def detect_lateswing_frame(df, mvp, touchdown):
    window = df[(df["frame"] >= mvp) & (df["frame"] <= touchdown)].copy()
    if window.empty:
        return None
    window["largest_leg_angle"] = window[["left_leg_angle", "right_leg_angle"]].max(axis=1)
    window = window.dropna(subset=["largest_leg_angle"])
    if window.empty:
        return None
    return int(window.loc[window["largest_leg_angle"].idxmax(), "frame"])


def detect_new_algorithm(df):
    toeoffs = sorted(detect_new_toeoff_frames(df))
    mvps = sorted(detect_new_mvp_frames(df))
    midstances = sorted(detect_new_midstance_frames(df))

    pairs = []
    used_toeoffs = set()
    for mvp in mvps:
        eligible = [
            toeoff for toeoff in toeoffs
            if toeoff not in used_toeoffs
            and NEW_PAIR_MIN_GAP <= mvp - toeoff <= NEW_PAIR_MAX_GAP
        ]
        if eligible:
            toeoff = max(eligible)
            used_toeoffs.add(toeoff)
            pairs.append((toeoff, mvp))

    cycles = []
    max_frame = int(df["frame"].max())
    for i, (toeoff, mvp) in enumerate(pairs):
        predicted_touchdown = int(mvp + (mvp - toeoff))
        next_toeoff = pairs[i + 1][0] if i + 1 < len(pairs) else None

        if predicted_touchdown <= max_frame:
            touchdown = predicted_touchdown
            lateswing = detect_lateswing_frame(df, mvp, touchdown)
            valid_mids = [
                frame for frame in midstances
                if frame >= touchdown and (next_toeoff is None or frame < next_toeoff)
            ]
            midstance = valid_mids[0] if valid_mids else None
        else:
            touchdown = None
            lateswing = None
            midstance = None

        cycles.append({
            "toeoff": int(toeoff),
            "mvp": int(mvp),
            "lateswing": lateswing,
            "touchdown": touchdown,
            "midstance": midstance,
        })
    return cycles


def old_toeoff_single(frames, values):
    values = fill_nan(values)
    if len(values) < 10 or np.ptp(values) == 0:
        return []
    signal_range = np.ptp(values)
    valleys, _ = find_peaks(-values, prominence=OLD_TOEOFF_PROMINENCE_PCT * signal_range, distance=5)

    d1 = np.zeros_like(values)
    for i in range(2, len(values) - 2):
        d1[i] = values[i + 2] - values[i - 2]
    d2 = np.zeros_like(values)
    for i in range(1, len(values) - 1):
        d2[i] = d1[i + 1] - d1[i - 1]

    d1 /= np.max(np.abs(d1)) + 1e-8
    d2 /= np.max(np.abs(d2)) + 1e-8
    dy = np.diff(values)
    noise_floor = OLD_TOEOFF_NOISE_FLOOR_PCT * signal_range
    max_search = max(6, int(OLD_TOEOFF_MAX_SEARCH_PCT * len(values)))
    predictions = []

    for valley in valleys:
        exit_idx = None
        search_end = min(valley + max_search, len(values) - OLD_TOEOFF_CONFIRM_K)
        for i in range(valley, search_end):
            window = dy[i:i + OLD_TOEOFF_CONFIRM_K]
            if len(window) == OLD_TOEOFF_CONFIRM_K and np.mean(window) > noise_floor:
                exit_idx = i
                break
        if exit_idx is None:
            continue
        start = max(exit_idx - OLD_TOEOFF_REFINE_RADIUS, 2)
        end = min(exit_idx + OLD_TOEOFF_REFINE_RADIUS, len(values) - 2)
        scores = [(0.5 * d2[j] + 0.3 * d1[j], j) for j in range(start, end)]
        predictions.append(int(frames[max(scores)[1] if scores else exit_idx]))
    return predictions


def deduplicate(frames, window=BILATERAL_MERGE_WINDOW):
    frames = sorted(int(f) for f in frames)
    if not frames:
        return []
    groups = [[frames[0]]]
    for frame in frames[1:]:
        if frame - groups[-1][-1] <= window:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return [int(round(np.mean(group))) for group in groups]


def detect_old_algorithm(df):
    frames = df["frame"].to_numpy(int)
    toeoff_predictions = []
    for side in ["left", "right"]:
        toeoff_predictions.extend(old_toeoff_single(frames, df[f"{side}_toe_y_detrended_norm_ema"]))
    toeoffs = deduplicate(toeoff_predictions)

    heel_minima = []
    toe_maxima = []
    for side in ["left", "right"]:
        heel = fill_nan(df[f"{side}_heel_y_detrended_ema"])
        if np.ptp(heel) > 0:
            idx, _ = find_peaks(-heel, prominence=OLD_HEEL_PROMINENCE_PCT * np.ptp(heel), distance=5)
            heel_minima.extend(int(frames[i]) for i in idx)
        toe = fill_nan(df[f"{side}_toe_y_detrended_norm_ema"])
        if np.ptp(toe) > 0:
            idx, _ = find_peaks(toe, prominence=OLD_TOE_PROMINENCE_PCT * np.ptp(toe), distance=5)
            toe_maxima.extend(int(frames[i]) for i in idx)

    heel_minima = sorted(heel_minima)
    toe_maxima = sorted(toe_maxima)
    cycles = []
    for i, toeoff in enumerate(toeoffs):
        end = toeoffs[i + 1] - 1 if i + 1 < len(toeoffs) else min(int(frames.max()), toeoff + OLD_MAX_STRIDE_SEARCH)
        heels = [f for f in heel_minima if toeoff + 3 <= f <= end]
        if not heels:
            continue
        heel_min = heels[0]
        heel_td = heel_min - OLD_TOUCHDOWN_HEEL_OFFSET
        toes = [f for f in toe_maxima if toeoff < f <= end]
        toe_max = toes[0] if toes else None
        touchdown = int(round((toe_max + heel_td) / 2)) if toe_max is not None and abs(toe_max - heel_td) <= OLD_TOE_HEEL_AGREE_WINDOW else int(heel_td)
        cycles.append({
            "toeoff": int(toeoff),
            "mvp": int(round((toeoff + touchdown) / 2)),
            # Old algorithm only: Late Swing is defined as two frames
            # before the old-algorithm Touchdown prediction.
            "lateswing": max(int(frames.min()), int(touchdown) - 2),
            "touchdown": int(touchdown),
            "midstance": int(heel_min),
        })
    return cycles


# ============================================================
# EVALUATION
# ============================================================

def cycles_to_map(cycles):
    result = {position: [] for position in POSITION_ORDER}
    for cycle in cycles:
        for position in POSITION_ORDER:
            if cycle.get(position) is not None:
                result[position].append(int(cycle[position]))
    return result


def actual_items(events, position):
    rows = events[events["position_norm"] == position].sort_values(["gait_cycle", "frame"])
    return [{"gait_cycle": int(r["gait_cycle"]), "frame": int(r["frame"])} for _, r in rows.iterrows()]


def prediction_rows(actual, predicted, algorithm, position):
    predicted = sorted(int(f) for f in predicted)
    rows = []
    for i, item in enumerate(actual):
        pred = predicted[i] if i < len(predicted) else None
        error = None if pred is None else pred - item["frame"]
        rows.append({
            "algorithm": algorithm,
            "position": position,
            "gait_cycle": item["gait_cycle"],
            "actual_frame": item["frame"],
            "predicted_frame": pred,
            "error": error,
            "abs_error": None if error is None else abs(error),
            "correct": error == 0 if error is not None else False,
        })
    return rows


def summarize_rows(rows):
    valid = [r for r in rows if r["predicted_frame"] is not None]
    total = len(rows)
    if not valid:
        return {"n_ground_truth": total, "n_predicted": 0, "accuracy": 0.0 if total else None, "mae": None, "rmse": None}
    errors = np.array([r["error"] for r in valid], float)
    return {
        "n_ground_truth": total,
        "n_predicted": len(valid),
        "accuracy": sum(r["correct"] for r in valid) / total,
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }


def build_prediction_performance_payload(df, events):
    algorithms = {
        "old": ("Old Algorithm", cycles_to_map(detect_old_algorithm(df))),
        "new": ("New Algorithm", cycles_to_map(detect_new_algorithm(df))),
    }
    sections = []
    for algorithm_id, (algorithm_name, event_map) in algorithms.items():
        for position in POSITION_ORDER:
            rows = prediction_rows(actual_items(events, position), event_map[position], algorithm_name, position)
            sections.append({
                "section_id": f"{algorithm_id}_{position}",
                "title": f"{algorithm_name}: {POSITION_LABELS[position]}",
                "algorithm": algorithm_name,
                "position": position,
                "predicted_frames": event_map[position],
                "rows": rows,
                "summary": summarize_rows(rows),
            })
    return {
        "notes": [
            "Old algorithm retained for comparison.",
            "New algorithm predicts all five positions.",
            "Only new-algorithm frames are used for scoring.",
        ],
        "sections": sections,
    }


def build_new_predicted_events_df(events, performance):
    section_maps = {}
    for section in performance["sections"]:
        if section["section_id"].startswith("new_"):
            section_maps[section["position"]] = {
                int(r["gait_cycle"]): r["predicted_frame"]
                for r in section["rows"]
                if r["predicted_frame"] is not None
            }

    rows = []
    for _, row in events.iterrows():
        new_row = row.copy()
        cycle = int(row["gait_cycle"])
        position = row["position_norm"]
        gt = int(row["frame"])
        pred = section_maps.get(position, {}).get(cycle)
        new_row["ground_truth_frame"] = gt
        if pred is None:
            new_row["frame"] = gt
            new_row["frame_source"] = "ground_truth_fallback"
        else:
            new_row["frame"] = int(pred)
            new_row["frame_source"] = "new_algorithm"
        rows.append(new_row)

    result = pd.DataFrame(rows)
    result["frame"] = result["frame"].astype(int)
    return result


# ============================================================
# SCORING + GEOMETRY
# ============================================================

@dataclass
class MetricResult:
    metric_id: str
    metric_label: str
    position: str
    frame: int
    gait_cycle: int
    score: int
    passed: bool
    penalty: bool
    value_text: str
    threshold_text: str
    reasoning: str
    annotated_image: str | None = None


def point(row, name):
    x = row.get(f"{name}_x", np.nan)
    y = row.get(f"{name}_y", np.nan)
    return None if pd.isna(x) or pd.isna(y) else (float(x), float(y))


def midpoint(p1, p2):
    return None if p1 is None or p2 is None else ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def get_midhip(row):
    return point(row, "midhip") or midpoint(point(row, "left_hip"), point(row, "right_hip"))


def get_ankle_or_foot(row, side):
    return point(row, f"{side}_ankle") or midpoint(point(row, f"{side}_toe"), point(row, f"{side}_heel")) or point(row, f"{side}_foot")


def get_toe(row, side):
    """Return toe-tip when available, then toe, then legacy foot alias."""
    if side is None:
        return None
    return (
        point(row, f"{side}_toe_tip")
        or point(row, f"{side}_toe")
        or point(row, f"{side}_foot")
    )


def angle_between_vectors_degrees(v1, v2):
    """Return the smaller 2D angle between two vectors in degrees."""
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def get_heel(row, side):
    return point(row, f"{side}_heel")


def estimate_running_direction(df):
    slope, _ = np.polyfit(np.arange(len(df)), df["midhip_x"].to_numpy(float), 1)
    return "left_to_right" if slope > 0 else "right_to_left"


def ahead(candidate, reference, direction):
    return reference - candidate if direction == "right_to_left" else candidate - reference


def choose_side(row, direction, front, joint):
    left = point(row, f"left_{joint}")
    right = point(row, f"right_{joint}")
    if left is None or right is None:
        return None
    if direction == "left_to_right":
        return ("left" if left[0] > right[0] else "right") if front else ("left" if left[0] < right[0] else "right")
    return ("left" if left[0] < right[0] else "right") if front else ("left" if left[0] > right[0] else "right")


def choose_stance_side(row):
    candidates = []
    for side in ["left", "right"]:
        pts = [get_ankle_or_foot(row, side), get_heel(row, side), get_toe(row, side)]
        pts = [p for p in pts if p is not None]
        if pts:
            candidates.append((side, max(p[1] for p in pts)))
    return max(candidates, key=lambda x: x[1])[0] if candidates else None


def safe_height(row):
    ys = [
        p[1] for name in [
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_foot", "right_foot", "left_toe", "right_toe",
            "left_heel", "right_heel"
        ] if (p := point(row, name)) is not None
    ]
    return max(max(ys) - min(ys), 1) if len(ys) >= 2 else 300


def threshold(row, frac):
    return max(MIN_PIXEL_THRESHOLD, safe_height(row) * frac)


def fail_result(metric_id, position, frame, cycle, message):
    return MetricResult(metric_id, METRIC_LABELS[metric_id], position, int(frame), int(cycle), 1, False, True, "Value: unavailable", "Threshold: unavailable", message)


def metric(metric_id, row, cycle, direction):
    frame = int(row["frame"])

    if metric_id == "toeoff_angle":
        side = choose_side(row, direction, False, "heel")
        mid = get_midhip(row)
        heel = point(row, f"{side}_heel") if side else None
        if mid is None or heel is None:
            return fail_result(metric_id, "toeoff", frame, cycle, "Missing midhip or back heel keypoint.")
        angle = math.degrees(math.atan2(abs(heel[0] - mid[0]), abs(heel[1] - mid[1])))
        low = TOEOFF_TARGET_ANGLE_DEG - TOEOFF_TOLERANCE_DEG
        high = TOEOFF_TARGET_ANGLE_DEG + TOEOFF_TOLERANCE_DEG
        passed = low <= angle <= high
        return MetricResult(metric_id, METRIC_LABELS[metric_id], "toeoff", frame, cycle, 0 if passed else 1, passed, not passed, f"Angle: {angle:.1f}°", f"Pass range: {low:.0f}° to {high:.0f}°", "Back-heel angle comparison.")

    if metric_id == "mvp_shin_height":
        side = choose_side(row, direction, False, "heel")
        knee = point(row, f"{side}_knee") if side else None
        heel = point(row, f"{side}_heel") if side else None
        if knee is None or heel is None:
            return fail_result(metric_id, "mvp", frame, cycle, "Missing back knee or heel keypoint.")
        limit = threshold(row, MVP_HEEL_KNEE_HEIGHT_FRAC)
        value = heel[1] - knee[1]
        passed = value > limit
        return MetricResult(metric_id, METRIC_LABELS[metric_id], "mvp", frame, cycle, 0 if passed else 1, passed, not passed, f"Back heel clearance below knee line: {value:.1f}px", f"Pass if clearance > {limit:.1f}px", "Heel clearance below knee.")

    if metric_id == "late_swing_knee_behind_hip":
        side = choose_side(
            row,
            direction,
            False,
            "knee",
        )

        hip = (
            point(row, f"{side}_hip")
            if side
            else None
        )

        knee = (
            point(row, f"{side}_knee")
            if side
            else None
        )

        if hip is None or knee is None:
            return fail_result(
                metric_id,
                "lateswing",
                frame,
                cycle,
                "Missing trailing hip or knee keypoint.",
            )

        # Positive means the knee is in front of the hip
        # in the athlete's running direction.
        knee_ahead = ahead(
            knee[0],
            hip[0],
            direction,
        )

        limit = threshold(
            row,
            LATE_SWING_KNEE_BEHIND_HIP_FRAC,
        )

        # Pass when the trailing knee has moved sufficiently
        # in front of the trailing hip.
        passed = knee_ahead >= limit

        if passed:
            reasoning = (
                f"The trailing knee is {knee_ahead:.1f}px in front of "
                f"the trailing hip line, meeting the required "
                f"{limit:.1f}px minimum."
            )
        else:
            reasoning = (
                f"The trailing knee is {knee_ahead:.1f}px in front of "
                f"the trailing hip line. The required minimum is "
                f"{limit:.1f}px. A negative value means the knee is "
                f"still behind the hip."
            )

        return MetricResult(
            metric_id,
            METRIC_LABELS[metric_id],
            "lateswing",
            frame,
            cycle,
            0 if passed else 1,
            passed,
            not passed,
            f"Trailing knee in front of hip line: {knee_ahead:.1f}px",
            f"Pass if knee-ahead distance >= {limit:.1f}px",
            reasoning,
        )

    if metric_id == "touchdown_thigh_separation":
        # Measure the opening between the thighs using the two knee points
        # relative to the shared midhip reference. This replaces the old
        # horizontal pixel-distance proxy.
        mid = get_midhip(row)
        left_knee = point(row, "left_knee")
        right_knee = point(row, "right_knee")
        if mid is None or left_knee is None or right_knee is None:
            return fail_result(
                metric_id,
                "touchdown",
                frame,
                cycle,
                "Missing midhip, left-knee, or right-knee keypoint.",
            )

        left_vector = (left_knee[0] - mid[0], left_knee[1] - mid[1])
        right_vector = (right_knee[0] - mid[0], right_knee[1] - mid[1])
        thigh_gap_angle = angle_between_vectors_degrees(left_vector, right_vector)

        if not np.isfinite(thigh_gap_angle):
            return fail_result(
                metric_id,
                "touchdown",
                frame,
                cycle,
                "Could not calculate the thigh-gap angle.",
            )

        passed = thigh_gap_angle <= TOUCHDOWN_THIGH_GAP_MAX_DEG
        return MetricResult(
            metric_id,
            METRIC_LABELS[metric_id],
            "touchdown",
            frame,
            cycle,
            0 if passed else 1,
            passed,
            not passed,
            f"Thigh-gap angle: {thigh_gap_angle:.1f}°",
            (
                f"Pass if angle <= {TOUCHDOWN_THIGH_GAP_MAX_DEG:.0f}° "
                f"({TOUCHDOWN_THIGH_GAP_TARGET_DEG:.0f}° target + "
                f"{TOUCHDOWN_THIGH_GAP_GRACE_DEG:.0f}° grace)"
            ),
            "Angle between the left and right midhip-to-knee vectors.",
        )

    if metric_id == "touchdown_shin_angle":
        front = choose_side(row, direction, True, "foot")
        knee = point(row, f"{front}_knee") if front else None
        foot = get_ankle_or_foot(row, front) if front else None
        if knee is None or foot is None:
            return fail_result(metric_id, "touchdown", frame, cycle, "Missing front knee or foot keypoint.")
        value = ahead(foot[0], knee[0], direction)
        limit = threshold(row, TOUCHDOWN_SHIN_AHEAD_FRAC)
        passed = value <= limit
        return MetricResult(metric_id, METRIC_LABELS[metric_id], "touchdown", frame, cycle, 0 if passed else 1, passed, not passed, f"Ankle/foot ahead of knee: {value:.1f}px", f"Pass if ahead distance <= {limit:.1f}px", "Foot ahead of knee.")

    if metric_id == "touchdown_foot_space":
        front = choose_side(row, direction, True, "foot")
        mid = get_midhip(row)
        foot = get_ankle_or_foot(row, front) if front else None
        toe = get_toe(row, front) if front else None
        heel = get_heel(row, front) if front else None
        if mid is None or foot is None:
            return fail_result(metric_id, "touchdown", frame, cycle, "Missing front foot or midhip keypoint.")
        foot_length = math.hypot(toe[0] - heel[0], toe[1] - heel[1]) if toe is not None and heel is not None else threshold(row, 0.08)
        value = ahead(foot[0], mid[0], direction) / max(foot_length, 1)
        passed = value <= TOUCHDOWN_MAX_FOOT_SPACE_MULT
        return MetricResult(metric_id, METRIC_LABELS[metric_id], "touchdown", frame, cycle, 0 if passed else 1, passed, not passed, f"Foot ahead of midhip: {value:.2f} foot-lengths", f"Pass if <= {TOUCHDOWN_MAX_FOOT_SPACE_MULT:.2f} foot-lengths", "Foot spacing.")

    if metric_id == "midstance_knee_over_foot":
        side = choose_stance_side(row)
        knee = point(row, f"{side}_knee") if side else None
        # Midstance explicitly uses toe-tip when available, otherwise toe.
        toe = get_toe(row, side) if side else None
        if knee is None or toe is None:
            return fail_result(
                metric_id,
                "midstance",
                frame,
                cycle,
                "Missing stance knee or stance toe-tip/toe keypoint.",
            )
        value = ahead(knee[0], toe[0], direction)
        limit = threshold(row, MIDSTANCE_KNEE_AHEAD_FOOT_FRAC)
        passed = value <= limit
        return MetricResult(
            metric_id,
            METRIC_LABELS[metric_id],
            "midstance",
            frame,
            cycle,
            0 if passed else 1,
            passed,
            not passed,
            f"Stance knee ahead of toe line: {value:.1f}px",
            f"Pass if knee-ahead distance <= {limit:.1f}px",
            "Stance knee relative to the stance toe-tip/toe vertical line.",
        )

    raise ValueError(metric_id)


def score_events(df, events, label):
    by_frame = {int(r["frame"]): r for _, r in df.iterrows()}
    direction = estimate_running_direction(df)
    results = []
    for _, event in events.iterrows():
        row = by_frame.get(int(event["frame"]))
        if row is None:
            continue
        for metric_id in POSITION_METRICS[event["position_norm"]]:
            result = metric(metric_id, row, int(event["gait_cycle"]), direction)
            results.append(metric_to_dict(result, None))
    penalties = sorted({r["metric_id"] for r in results if r["penalty"]})
    return {"label": label, "final_score": len(penalties), "unique_penalty_metric_ids": penalties, "event_results": results}


# ============================================================
# IMAGE ANNOTATION
# ============================================================

def load_font(size=18):
    for name in ["Arial.ttf", "DejaVuSans.ttf", "/Library/Fonts/Arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_point(draw, p, color=COLOR_POINT, label=None, font=None, radius=6):
    if p is None:
        return
    x, y = p
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline=COLOR_DARK)
    if label:
        draw.text((x + 8, y - 18), label, fill=color, font=font or load_font(14))


def draw_text_box(draw, lines):
    font = load_font(16)
    font_big = load_font(20)
    line_height = 24
    width = 650
    height = 18 + line_height * len(lines)
    draw.rectangle([8, 8, 8 + width, 8 + height], fill=COLOR_BOX, outline=COLOR_DARK)
    for i, line in enumerate(lines):
        draw.text((16, 16 + i * line_height), line, fill=COLOR_TEXT, font=font_big if i == 0 else font)


def find_image_for_frame(frame, df_by_frame, image_dir):
    image_dir = Path(image_dir)
    row = df_by_frame.get(int(frame))
    if row is not None:
        file_name = row.get("file_name")
        if file_name:
            for candidate in [image_dir / str(file_name), image_dir / Path(str(file_name)).name]:
                if candidate.exists():
                    return candidate

    for name in [
        f"{int(frame) + IMAGE_FRAME_OFFSET:05d}.jpg",
        f"{int(frame):05d}.jpg",
        f"{int(frame):04d}.jpg",
        f"{int(frame):03d}.jpg",
        f"{int(frame)}.jpg",
        f"{int(frame):05d}.png",
        f"{int(frame)}.png",
    ]:
        candidate = image_dir / name
        if candidate.exists():
            return candidate
    return None


def annotate_metric(result, row, image_path, output_path, direction):
    if image_path is None:
        return None
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = load_font(16)

    if result.metric_id == "toeoff_angle":
        side = choose_side(row, direction, False, "heel")
        mid = get_midhip(row)
        heel = get_heel(row, side) if side else None
        if mid and heel:
            draw.line([(mid[0], 0), (mid[0], height)], fill=COLOR_VERTICAL, width=4)
            draw.line([mid, heel], fill=COLOR_ACTUAL, width=4)
            ref_len = min(width, height) * 0.28
            sx = 1 if heel[0] >= mid[0] else -1
            sy = 1 if heel[1] >= mid[1] else -1
            ref = (mid[0] + sx * ref_len / math.sqrt(2), mid[1] + sy * ref_len / math.sqrt(2))
            draw.line([mid, ref], fill=COLOR_REFERENCE, width=4)
            draw_point(draw, mid, COLOR_VERTICAL, "midhip", font)
            draw_point(draw, heel, COLOR_ACTUAL, f"{side} heel", font)

    elif result.metric_id == "mvp_shin_height":
        side = choose_side(row, direction, False, "heel")
        knee = point(row, f"{side}_knee") if side else None
        heel = get_heel(row, side) if side else None
        if knee and heel:
            draw.line([(0, knee[1]), (width, knee[1])], fill=COLOR_REFERENCE, width=4)
            draw.line([knee, heel], fill=COLOR_ACTUAL, width=4)
            draw_point(draw, knee, COLOR_REFERENCE, f"{side} knee", font)
            draw_point(draw, heel, COLOR_ACTUAL, f"{side} heel", font)

    elif result.metric_id == "late_swing_knee_behind_hip":
        side = choose_side(row, direction, False, "knee")
        hip = point(row, f"{side}_hip") if side else None
        knee = point(row, f"{side}_knee") if side else None
        if hip and knee:
            draw.line([(hip[0], 0), (hip[0], height)], fill=COLOR_REFERENCE, width=4)
            draw.line([hip, knee], fill=COLOR_ACTUAL, width=4)
            draw.line([(hip[0], knee[1]), knee], fill=COLOR_VERTICAL, width=3)
            draw_point(draw, hip, COLOR_REFERENCE, f"{side} hip", font)
            draw_point(draw, knee, COLOR_ACTUAL, f"{side} knee", font)

    elif result.metric_id == "touchdown_thigh_separation":
        mid = get_midhip(row)
        left_knee = point(row, "left_knee")
        right_knee = point(row, "right_knee")
        if mid and left_knee and right_knee:
            # Draw the two thigh vectors whose included angle is scored.
            draw.line([mid, left_knee], fill=COLOR_VERTICAL, width=4)
            draw.line([mid, right_knee], fill=COLOR_ACTUAL, width=4)
            draw_point(draw, mid, COLOR_REFERENCE, "midhip", font)
            draw_point(draw, left_knee, COLOR_VERTICAL, "left knee", font)
            draw_point(draw, right_knee, COLOR_ACTUAL, "right knee", font)

    elif result.metric_id == "touchdown_shin_angle":
        front = choose_side(row, direction, True, "foot")
        knee = point(row, f"{front}_knee") if front else None
        foot = get_ankle_or_foot(row, front) if front else None
        if knee and foot:
            draw.line([(knee[0], 0), (knee[0], height)], fill=COLOR_REFERENCE, width=4)
            draw.line([knee, foot], fill=COLOR_ACTUAL, width=4)
            draw_point(draw, knee, COLOR_REFERENCE, f"front {front} knee", font)
            draw_point(draw, foot, COLOR_ACTUAL, f"front {front} foot", font)

    elif result.metric_id == "touchdown_foot_space":
        front = choose_side(row, direction, True, "foot")
        mid = get_midhip(row)
        foot = get_ankle_or_foot(row, front) if front else None
        toe = get_toe(row, front) if front else None
        heel = get_heel(row, front) if front else None
        if mid and foot:
            draw.line([(mid[0], 0), (mid[0], height)], fill=COLOR_REFERENCE, width=4)
            draw.line([(mid[0], foot[1]), foot], fill=COLOR_VERTICAL, width=3)
            if toe and heel:
                draw.line([heel, toe], fill=COLOR_ACTUAL, width=5)
            draw_point(draw, mid, COLOR_REFERENCE, "midhip", font)
            draw_point(draw, foot, COLOR_ACTUAL, f"front {front} foot", font)

    elif result.metric_id == "midstance_knee_over_foot":
        side = choose_stance_side(row)
        knee = point(row, f"{side}_knee") if side else None
        toe = get_toe(row, side) if side else None
        if knee and toe:
            draw.line([(toe[0], 0), (toe[0], height)], fill=COLOR_REFERENCE, width=4)
            draw.line([knee, toe], fill=COLOR_ACTUAL, width=4)
            draw.line([knee, (toe[0], knee[1])], fill=COLOR_VERTICAL, width=3)
            draw_point(draw, knee, COLOR_ACTUAL, f"{side} knee", font)
            draw_point(draw, toe, COLOR_REFERENCE, f"{side} toe-tip/toe", font)

    draw_text_box(draw, [
        f"{POSITION_LABELS[result.position]} frame {result.frame}",
        result.value_text,
        result.threshold_text,
        "Penalty: YES" if result.penalty else "Penalty: NO",
    ])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return Path(output_path)


def relpath_for_html(path, report_dir):
    if path is None:
        return None
    path = Path(path).resolve()
    if report_dir is None:
        return path.as_uri()
    try:
        return os.path.relpath(str(path), start=str(report_dir)).replace("\\", "/")
    except ValueError:
        return path.as_uri()


def metric_to_dict(result, report_dir):
    return {
        "metric_id": result.metric_id,
        "metric_label": result.metric_label,
        "position": result.position,
        "frame": int(result.frame),
        "gait_cycle": int(result.gait_cycle),
        "score": int(result.score),
        "passed": bool(result.passed),
        "penalty": bool(result.penalty),
        "value_text": result.value_text,
        "threshold_text": result.threshold_text,
        "reasoning": result.reasoning,
        "annotated_image": relpath_for_html(result.annotated_image, report_dir) if result.annotated_image else None,
    }


def get_frame_candidates(center_frame, all_frames):
    all_frames = sorted(int(f) for f in all_frames)
    if BROWSE_ALL_FRAMES:
        return all_frames
    return [f for f in all_frames if center_frame - FRAME_WINDOW <= f <= center_frame + FRAME_WINDOW]


def get_selected_heel_info(row, position, direction):
    if position in ["toeoff", "mvp", "lateswing"]:
        side = choose_side(row, direction, False, "heel")
    elif position == "touchdown":
        side = choose_side(row, direction, True, "heel")
    else:
        side = choose_stance_side(row)
    heel = get_heel(row, side) if side else None
    if heel is None:
        return {"side": side, "x": None, "y": None, "text": "Selected heel: unavailable"}
    return {"side": side, "x": float(heel[0]), "y": float(heel[1]), "text": f"Selected heel: {side}, x={heel[0]:.2f}, y={heel[1]:.2f}"}


# ============================================================
# PAYLOAD + WEBPAGE
# ============================================================

def build_scoring_payload(df, predicted_events, performance, ground_truth_events):
    direction = estimate_running_direction(df)
    by_frame = {int(r["frame"]): r for _, r in df.iterrows()}
    all_frames = sorted(by_frame)
    # Keep annotations separate for each athlete, even when this core module
    # is called repeatedly by the batch script.
    report_dir = None
    annotated_dir = Path(IMAGE_DIR).resolve().parent / "_smas_annotated_frames"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    cycles = []
    for cycle_id, cycle_events in predicted_events.groupby("gait_cycle"):
        cycle = {"gait_cycle": int(cycle_id), "positions": {}}
        for position in POSITION_ORDER:
            rows = cycle_events[cycle_events["position_norm"] == position]
            if rows.empty:
                cycle["positions"][position] = {
                    "position": position,
                    "label": POSITION_LABELS[position],
                    "event_frame": None,
                    "frame_candidates": [],
                    "frames": {},
                    "missing": True,
                    "message": f"No {POSITION_LABELS[position]} frame found for gait cycle {cycle_id}.",
                }
                continue

            event = rows.iloc[0]
            center_frame = int(event["frame"])
            candidates = get_frame_candidates(center_frame, all_frames)
            position_payload = {
                "position": position,
                "label": POSITION_LABELS[position],
                "event_frame": center_frame,
                "frame_source": event.get("frame_source", "unknown"),
                "frame_candidates": candidates,
                "frames": {},
                "missing": False,
            }

            for frame in candidates:
                row = by_frame.get(frame)
                if row is None:
                    continue
                image_path = find_image_for_frame(frame, by_frame, IMAGE_DIR)
                frame_metrics = {}
                for metric_id in POSITION_METRICS[position]:
                    result = metric(metric_id, row, int(cycle_id), direction)
                    if image_path is not None:
                        output_path = annotated_dir / f"cycle_{int(cycle_id):02d}_{position}_frame_{frame:05d}_{metric_id}.jpg"
                        result.annotated_image = annotate_metric(result, row, image_path, output_path, direction)
                    frame_metrics[metric_id] = metric_to_dict(result, report_dir)

                position_payload["frames"][str(frame)] = {
                    "raw_image": relpath_for_html(image_path, report_dir),
                    "selected_heel": get_selected_heel_info(row, position, direction),
                    "metrics": frame_metrics,
                }

            cycle["positions"][position] = position_payload
        cycles.append(cycle)

    payload = {
        "metadata": {
            "running_direction": direction,
            "scoring_algorithm": "New Algorithm",
            "image_frame_offset": IMAGE_FRAME_OFFSET,
            "browse_all_frames": BROWSE_ALL_FRAMES,
            "frame_window": FRAME_WINDOW,
        },
        "metric_labels": METRIC_LABELS,
        "position_labels": POSITION_LABELS,
        "position_metrics": POSITION_METRICS,
        "cycles": cycles,
        "prediction_performance": performance,
        "score_comparison": {
            "ground_truth": score_events(df, ground_truth_events, "Ground Truth"),
            "predicted": score_events(df, predicted_events, "New Algorithm"),
        },
    }
    return payload


def write_webpage(payload, output_path):
    data = json.dumps(payload).replace("</", "<\\/")
    html = r'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SMAS Scoring Web Report</title>
<style>
:root{--bg:#f5f7fb;--card:#fff;--ink:#1c2430;--muted:#697386;--border:#d8dee9;--nav:#17202a;--nav2:#243447;--accent:#1f6feb;--danger:#d93025;--ok:#1769aa;--warn:#b35c00}
*{box-sizing:border-box}body{margin:0;font-family:Arial,Helvetica,sans-serif;background:var(--bg);color:var(--ink)}
.header{padding:22px 28px;background:var(--nav);color:#fff}.header h1{margin:0}.header p{color:#d7dde8}
.tab-bar{display:flex;flex-wrap:wrap;gap:6px;padding:12px 18px;background:var(--nav2);position:sticky;top:0;z-index:99}
.tab-button,button{border:0;border-radius:7px;padding:9px 13px;cursor:pointer;background:var(--accent);color:#fff;font-weight:600}.tab-button{background:#40546d}.tab-button.active{background:#fff;color:var(--ink)}button.secondary{background:#637083}button.fake-store{background:#343a40}
.tab-content{display:none;padding:22px}.tab-content.active{display:block}.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:18px}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin:10px 0 12px}.result{border-top:1px solid var(--border);margin-top:12px;padding-top:12px}
select,input{padding:8px 10px;border:1px solid var(--border);border-radius:7px;background:#fff}.badge{display:inline-block;border-radius:999px;padding:4px 9px;font-size:12px;font-weight:700}.badge.pass{background:#e6f2ff;color:var(--ok)}.badge.fail{background:#fde8e7;color:var(--danger)}
.metric-result{border:1px solid var(--border);border-radius:10px;padding:12px;margin:10px 0;background:#fbfcff}.metric-result.fail{border-left:5px solid var(--danger)}.metric-result.pass{border-left:5px solid var(--ok)}
.annotation{width:100%;max-width:1100px;margin-top:10px;border-radius:8px;border:1px solid var(--border);display:block}.summary-table{width:100%;border-collapse:collapse;font-size:14px}.summary-table th,.summary-table td{padding:10px;border-bottom:1px solid var(--border);vertical-align:top;text-align:left}.summary-table th{background:#edf2f7}.small{font-size:13px;color:var(--muted)}.final-score{font-size:34px;font-weight:800}.notice{background:#fff8e6;border-left:5px solid var(--warn);padding:12px;border-radius:8px}
</style></head><body>
<div class="header"><h1>SMAS Scoring Web Report</h1><p>Predicted frames, visual annotations, frame selection, recalculation, overrides, and final summary.</p></div>
<div class="tab-bar" id="tabBar"></div><div id="app"></div>
<script>
const DATA=__DATA__;
let selectedFrames={};let calculated={};let overrides={};let finalScoreOverride=null;
function init(){buildTabs();buildCycleTabs();buildSummaryTab();buildPredictionTab();openTab('summary')}
function buildTabs(){const b=document.getElementById('tabBar');[['summary','Final Report'],['prediction','Prediction Performance']].forEach(([id,label])=>{const x=document.createElement('button');x.id='button_'+id;x.className='tab-button';x.innerText=label;x.onclick=()=>openTab(id);b.appendChild(x)});DATA.cycles.forEach(c=>{const id='cycle_'+c.gait_cycle,x=document.createElement('button');x.id='button_'+id;x.className='tab-button';x.innerText='Gait Cycle '+c.gait_cycle;x.onclick=()=>openTab(id);b.appendChild(x)})}
function openTab(id){document.querySelectorAll('.tab-content').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab-button').forEach(x=>x.classList.remove('active'));document.getElementById(id).classList.add('active');document.getElementById('button_'+id).classList.add('active');if(id==='summary')renderSummary()}
function getCycle(id){return DATA.cycles.find(c=>c.gait_cycle===Number(id))}
function buildCycleTabs(){const app=document.getElementById('app');DATA.cycles.forEach(c=>{const tab=document.createElement('div');tab.id='cycle_'+c.gait_cycle;tab.className='tab-content';let h=`<h2>Gait Cycle ${c.gait_cycle}</h2><div class="notice">Choose a frame, then click <strong>Calculate</strong> to use that frame's precomputed score and annotated image.</div>`;Object.keys(DATA.position_labels).forEach(p=>{const d=c.positions[p],card=`cycle_${c.gait_cycle}_${p}`;if(d.missing){h+=`<div class="card"><h3>${d.label}</h3><p>${d.message}</p></div>`;return}const opts=d.frame_candidates.map(f=>`<option value="${f}" ${f===d.event_frame?'selected':''}>Frame ${f}${f===d.event_frame?' — event frame':''}</option>`).join('');h+=`<div class="card" id="${card}"><h3>${d.label}</h3><p class="small">Predicted/scoring frame: <strong>${d.event_frame}</strong> (${d.frame_source||'unknown'})</p><div class="controls"><button class="secondary" onclick="stepFrame(${c.gait_cycle},'${p}',-1)">← Prev</button><select id="${card}_select" onchange="setSelectedFrame(${c.gait_cycle},'${p}',this.value)">${opts}</select><button class="secondary" onclick="stepFrame(${c.gait_cycle},'${p}',1)">Next →</button><button onclick="calculatePosition(${c.gait_cycle},'${p}')">Calculate</button></div><div id="${card}_preview"></div><div id="${card}_result" class="result"></div></div>`});tab.innerHTML=h;app.appendChild(tab);Object.keys(DATA.position_labels).forEach(p=>{const d=c.positions[p];if(!d||d.missing)return;selectedFrames[`${c.gait_cycle}_${p}`]=d.event_frame;renderPreview(c.gait_cycle,p)})})}
function setSelectedFrame(c,p,f){selectedFrames[`${c}_${p}`]=Number(f);renderPreview(c,p)}
function stepFrame(c,p,delta){const d=getCycle(c).positions[p],frames=d.frame_candidates,key=`${c}_${p}`;let i=frames.indexOf(Number(selectedFrames[key]??d.event_frame));i=Math.max(0,Math.min(frames.length-1,(i<0?0:i)+delta));selectedFrames[key]=frames[i];document.getElementById(`cycle_${c}_${p}_select`).value=frames[i];renderPreview(c,p)}
function renderPreview(c,p){const d=getCycle(c).positions[p],f=selectedFrames[`${c}_${p}`],fd=d.frames[String(f)],el=document.getElementById(`cycle_${c}_${p}_preview`);el.innerHTML=fd&&fd.raw_image?`<p class="small">Selected frame: <strong>${f}</strong></p><img class="annotation" src="${fd.raw_image}">`:`<p class="small">Raw image not found for frame ${f}.</p>`}
function overrideKey(m){return `${m.gait_cycle}|${m.position}|${m.frame}|${m.metric_id}`}
function calculatePosition(c,p){const d=getCycle(c).positions[p],f=selectedFrames[`${c}_${p}`],fd=d.frames[String(f)],el=document.getElementById(`cycle_${c}_${p}_result`);if(!fd){el.innerHTML='<p>No frame data found.</p>';return}calculated[`${c}_${p}`]=f;let h=`<h4>Calculated Results — Frame ${f}</h4><p class="small"><strong>${fd.selected_heel?.text||'Selected heel: unavailable'}</strong></p>`;Object.values(fd.metrics).forEach(m=>{const key=overrideKey(m),ov=overrides[key]??'';h+=`<div class="metric-result ${m.penalty?'fail':'pass'}"><span class="badge ${m.penalty?'fail':'pass'}">${m.penalty?'PENALTY +1':'PASS'}</span> <strong>${m.metric_label}</strong><p><strong>${m.value_text}</strong><br>${m.threshold_text}</p><p>${m.reasoning}</p><div class="controls"><label><strong>Override:</strong></label><select onchange="setOverride('${key}',this.value)"><option value="" ${ov===''?'selected':''}>Use algorithm score</option><option value="pass" ${ov==='pass'?'selected':''}>Override to pass</option><option value="fail" ${ov==='fail'?'selected':''}>Override to fail</option></select></div>${m.annotated_image?`<img class="annotation" src="${m.annotated_image}">`:'<p class="small">Annotated image unavailable.</p>'}</div>`});el.innerHTML=h;renderSummary()}
function setOverride(k,v){if(v==='')delete overrides[k];else overrides[k]=v;renderSummary()}
function activePenalty(m){const v=overrides[overrideKey(m)];return v==='pass'?false:v==='fail'?true:m.penalty}
function activeResults(){let out=[];DATA.cycles.forEach(c=>Object.keys(c.positions).forEach(p=>{const d=c.positions[p];if(d.missing)return;const f=calculated[`${c.gait_cycle}_${p}`]??d.event_frame,fd=d.frames[String(f)];if(fd)Object.values(fd.metrics).forEach(m=>out.push(m))}));return out}
function summary(){const by={};Object.keys(DATA.metric_labels).forEach(id=>by[id]={metric_id:id,metric_label:DATA.metric_labels[id],penalty:false,failures:[]});activeResults().forEach(m=>{if(activePenalty(m)){by[m.metric_id].penalty=true;by[m.metric_id].failures.push(m)}});const rows=Object.values(by),algorithmScore=rows.filter(r=>r.penalty).length;return{rows,algorithmScore,displayedScore:finalScoreOverride===null?algorithmScore:finalScoreOverride}}
function buildSummaryTab(){const t=document.createElement('div');t.id='summary';t.className='tab-content';t.innerHTML='<h2>Final Report Summary</h2><div id="summary_content"></div>';document.getElementById('app').appendChild(t)}
function renderSummary(){const el=document.getElementById('summary_content');if(!el)return;const s=summary();let h=`<div class="card"><h3>Final Score</h3><div class="final-score">${s.displayedScore}</div><p class="small">Algorithm unique-metric score: <strong>${s.algorithmScore}</strong>.</p><div class="controls"><label><strong>Override final score:</strong></label><input type="number" min="0" value="${finalScoreOverride??''}" onchange="finalScoreOverride=this.value===''?null:Number(this.value);renderSummary()"><button class="fake-store" onclick="alert('Placeholder only: values are not connected to a database yet.')">Store Values</button></div></div><div class="card"><h3>Penalized Metrics</h3><table class="summary-table"><tr><th>Metric</th><th>Penalty?</th><th>Where it failed</th></tr>`;s.rows.forEach(r=>{h+=`<tr><td>${r.metric_label}</td><td><span class="badge ${r.penalty?'fail':'pass'}">${r.penalty?'+1':'0'}</span></td><td>${r.failures.length?r.failures.map(m=>`Cycle ${m.gait_cycle}, ${DATA.position_labels[m.position]}, frame ${m.frame}`).join('<br>'):'No active failures'}</td></tr>`});h+='</table></div>';el.innerHTML=h}
function buildPredictionTab(){const t=document.createElement('div');t.id='prediction';t.className='tab-content';let h='<h2>Prediction Performance</h2>';DATA.prediction_performance.sections.forEach(s=>{h+=`<div class="card"><h3>${s.title}</h3><table class="summary-table"><tr><th>Cycle</th><th>Actual</th><th>Predicted</th><th>Error</th></tr>`;s.rows.forEach(r=>h+=`<tr><td>${r.gait_cycle}</td><td>${r.actual_frame}</td><td>${r.predicted_frame??'N/A'}</td><td>${r.error??'N/A'}</td></tr>`);h+='</table></div>'});t.innerHTML=h;document.getElementById('app').appendChild(t)}
document.addEventListener('DOMContentLoaded',init);
</script></body></html>'''.replace('__DATA__', data)
    Path(output_path).write_text(html, encoding='utf-8')


# ============================================================
# MAIN
# ============================================================

def run():
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    df, joint_names, fps, raw_json = load_motion_json(JSON_PATH)
    df = add_feature_columns_for_prediction(df)
    ground_truth = load_events_csv(EVENTS_CSV)
    performance = build_prediction_performance_payload(df, ground_truth)
    predicted = build_new_predicted_events_df(ground_truth, performance)
    payload = build_scoring_payload(df, predicted, performance, ground_truth)
    payload["athlete"] = raw_json.get("athlete", "Unknown athlete")
    payload["fps"] = fps
    payload["joint_names"] = joint_names

    df.to_csv(output / "motion_features.csv", index=False)
    predicted.to_csv(output / "new_algorithm_events.csv", index=False)
    (output / "smas_scoring_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_webpage(payload, output / "smas_scoring_report.html")
    print(f"Saved report: {output / 'smas_scoring_report.html'}")
    print(f"Saved predictions: {output / 'new_algorithm_events.csv'}")


if __name__ == "__main__":
    run()
