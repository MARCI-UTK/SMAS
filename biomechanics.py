from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FORWARD_LEAN_MIN = -2.0
FORWARD_LEAN_MAX = 15.0

# Provisional thresholds. Validate these against coach-labelled examples.
TRUNK_PELVIS_SEPARATION_MAX = 20.0
SIGNED_EXTENSION_BEND_REVIEW_MIN = 12.0
TRAILING_HIP_BEHIND_PELVIS_REVIEW_MIN = 0.10


def norm(v):
    v = np.asarray(v, dtype=float)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def horizontal(v, up):
    up = norm(up)
    v = np.asarray(v, dtype=float)
    return norm(v - np.dot(v, up) * up)


def signed_angle(a, b, normal):
    a, b, normal = norm(a), norm(b), norm(normal)
    return math.degrees(
        math.atan2(np.dot(normal, np.cross(a, b)), np.dot(a, b))
    )


def signed_angle_2d(v1, v2):
    """Signed turn from v1 to v2 in image coordinates, in degrees."""
    x1, y1 = float(v1[0]), float(v1[1])
    x2, y2 = float(v2[0]), float(v2[1])
    return math.degrees(math.atan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2))


def load_font(size=28):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _valid_2d(value):
    return (
        value is not None
        and len(value) >= 2
        and value[0] is not None
        and value[1] is not None
        and np.isfinite(value[0])
        and np.isfinite(value[1])
    )


def analyze(smpl_path: Path, lower_path: Path) -> dict:
    smpl = json.loads(Path(smpl_path).read_text())
    lower = json.loads(Path(lower_path).read_text())

    up = norm(smpl["world_up"])
    sprint = horizontal(smpl["sprint_dir"], up)
    run_sign = 1.0 if float(smpl["sprint_dir"][0]) >= 0 else -1.0

    lower_joints = lower["joints"]
    lhip_3d = np.asarray(lower_joints["l_upleg"]["world_m"], float)
    rhip_3d = np.asarray(lower_joints["r_upleg"]["world_m"], float)

    lhip_2d = lower_joints["l_upleg"]["pixel_uv"]
    rhip_2d = lower_joints["r_upleg"]["pixel_uv"]
    lknee_2d = lower_joints["l_lowleg"]["pixel_uv"]
    rknee_2d = lower_joints["r_lowleg"]["pixel_uv"]

    names = smpl["joint_names"]
    idx = {name: i for i, name in enumerate(names)}

    # Only analyze frames present in both datasets. A length mismatch is
    # reported to the analyst by the pipeline's alignment warning instead of
    # failing the whole run.
    n_frames = min(
        len(smpl["frames"]),
        len(lhip_3d), len(rhip_3d),
        len(lhip_2d), len(rhip_2d), len(lknee_2d), len(rknee_2d),
    )

    frames = []
    for i, frame in enumerate(smpl["frames"][:n_frames]):
        spine = norm(frame["spine_up"])
        chest = horizontal(frame["chest_fwd"], up)

        vertical_component = float(np.dot(spine, up))
        forward_component = float(np.dot(spine, sprint))
        forward_lean = math.degrees(
            math.atan2(forward_component, vertical_component)
        )

        trunk_yaw = signed_angle(sprint, chest, up)

        pelvis_right = horizontal(rhip_3d[i] - lhip_3d[i], up)
        pelvis_forward = horizontal(np.cross(up, pelvis_right), up)
        if np.dot(pelvis_forward, sprint) < 0:
            pelvis_forward = -pelvis_forward

        pelvis_yaw = signed_angle(sprint, pelvis_forward, up)
        trunk_pelvis = signed_angle(pelvis_forward, chest, up)

        kpts = frame["kpts_2d"]
        pelvis_2d = kpts[idx["pelvis"]]
        spine2_2d = kpts[idx["spine2"]]
        neck_2d = kpts[idx["neck"]]

        torso_len = 1.0
        signed_extension_bend = None
        trailing_ratio = None
        trailing_side = None

        if _valid_2d(pelvis_2d) and _valid_2d(neck_2d):
            torso_len = max(math.dist(pelvis_2d, neck_2d), 1.0)

        if _valid_2d(pelvis_2d) and _valid_2d(spine2_2d) and _valid_2d(neck_2d):
            lower_segment = (
                spine2_2d[0] - pelvis_2d[0],
                spine2_2d[1] - pelvis_2d[1],
            )
            upper_segment = (
                neck_2d[0] - spine2_2d[0],
                neck_2d[1] - spine2_2d[1],
            )
            # Flip with running direction so positive means the same visual
            # extension direction for athletes running either way.
            signed_extension_bend = run_sign * signed_angle_2d(
                lower_segment,
                upper_segment,
            )

        side_candidates = []
        for side, hip_point, knee_point in [
            ("left", lhip_2d[i], lknee_2d[i]),
            ("right", rhip_2d[i], rknee_2d[i]),
        ]:
            if _valid_2d(hip_point) and _valid_2d(knee_point):
                # Lower value means farther behind in the sprint direction.
                knee_progress = run_sign * float(knee_point[0])
                side_candidates.append((knee_progress, side, hip_point))

        if side_candidates and _valid_2d(pelvis_2d):
            _, trailing_side, trailing_hip = min(side_candidates, key=lambda item: item[0])
            trailing_ratio = (
                run_sign * (float(pelvis_2d[0]) - float(trailing_hip[0]))
                / torso_len
            )

        needs_review = bool(
            signed_extension_bend is not None
            and trailing_ratio is not None
            and signed_extension_bend > SIGNED_EXTENSION_BEND_REVIEW_MIN
            and trailing_ratio > TRAILING_HIP_BEHIND_PELVIS_REVIEW_MIN
        )

        frames.append({
            "frame": i,
            "forward_lean_deg": round(forward_lean, 2),
            "trunk_yaw_deg": round(trunk_yaw, 2),
            "pelvis_yaw_deg": round(pelvis_yaw, 2),
            "trunk_pelvis_rotation_deg": round(trunk_pelvis, 2),
            "signed_extension_bend_deg": (
                round(float(signed_extension_bend), 2)
                if signed_extension_bend is not None
                else None
            ),
            "trailing_hip_behind_pelvis_ratio": (
                round(float(trailing_ratio), 3)
                if trailing_ratio is not None
                else None
            ),
            "trailing_side": trailing_side,
            "trailing_hip_2d": (
                [float(trailing_hip[0]), float(trailing_hip[1])]
                if side_candidates and trailing_side is not None
                else None
            ),
            "lumbar_needs_review": needs_review,
            "kpts_2d": kpts,
            "sprint_horizontal": sprint.tolist(),
            "chest_horizontal": chest.tolist(),
            "pelvis_horizontal": pelvis_forward.tolist(),
            "world_up": up.tolist(),
        })

    return {
        "frames": frames,
        "joint_names": names,
        "skeleton": smpl["skeleton"],
    }


def metric(
    metric_id,
    label,
    cycle,
    position,
    frame,
    penalty,
    value,
    threshold,
    reasoning,
    *,
    review_required=False,
    review_suggested=False,
):
    return {
        "metric_id": metric_id,
        "metric_label": label,
        "gait_cycle": cycle,
        "position": position,
        "frame": int(frame),
        "penalty": bool(penalty),
        "value_text": value,
        "threshold_text": threshold,
        "reasoning": reasoning,
        "annotated_image": None,
        "review_required": bool(review_required),
        "review_suggested": bool(review_suggested),
    }


def metrics_for_position(frame_data: dict, cycle: int, position: str) -> list[dict]:
    frame = frame_data["frame"]
    output = []

    if position == "mvp":
        trunk = frame_data["trunk_yaw_deg"]
        pelvis = frame_data["pelvis_yaw_deg"]
        separation = frame_data["trunk_pelvis_rotation_deg"]
        penalty = abs(separation) > TRUNK_PELVIS_SEPARATION_MAX
        output.append(metric(
            "trunk_pelvic_rotation",
            "Trunk and pelvic rotation",
            cycle,
            position,
            frame,
            penalty,
            (
                f"Chest vs sprint: {trunk:.1f}°; pelvis vs sprint: "
                f"{pelvis:.1f}°; chest vs pelvis: {separation:.1f}°"
            ),
            (
                "Penalty if the absolute chest-to-pelvis rotation exceeds "
                f"{TRUNK_PELVIS_SEPARATION_MAX:.0f}°."
            ),
            (
                "The score uses chest rotation relative to the pelvis. "
                "Chest and pelvis rotation relative to sprint direction are "
                "shown as supporting values."
            ),
        ))

    if position in {"lateswing", "touchdown"}:
        bend = frame_data.get("signed_extension_bend_deg")
        trailing = frame_data.get("trailing_hip_behind_pelvis_ratio")
        lean = frame_data.get("forward_lean_deg")
        suggested = bool(frame_data.get("lumbar_needs_review"))

        bend_text = "unavailable" if bend is None else f"{bend:.1f}°"
        trailing_text = "unavailable" if trailing is None else f"{trailing:.2f}"

        if suggested:
            reasoning = "Possible extension pattern detected. Review the annotated frame."
        else:
            reasoning = (
                "The temporary review rule was not triggered, but this remains "
                "an analyst-reviewed measurement because the available keypoints "
                "cannot directly measure anterior pelvic tilt."
            )

        output.append(metric(
            "lumbar_extension_review",
            "Lumbar extension / anterior pelvic tilt",
            cycle,
            position,
            frame,
            False,  # Never adds a point until the analyst selects Penalty.
            (
                f"Forward lean: {lean:.1f}°; signed trunk curvature: "
                f"{bend_text}; trailing hip behind pelvis: {trailing_text} "
                "torso lengths"
            ),
            (
                "Review suggested when signed extension bend exceeds "
                f"{SIGNED_EXTENSION_BEND_REVIEW_MIN:.0f}° and trailing-hip "
                "position exceeds "
                f"{TRAILING_HIP_BEHIND_PELVIS_REVIEW_MIN:.2f} torso lengths."
            ),
            reasoning,
            review_required=True,
            review_suggested=suggested,
        ))

    if position == "touchdown":
        lean = frame_data["forward_lean_deg"]
        penalty = lean < FORWARD_LEAN_MIN or lean > FORWARD_LEAN_MAX
        output.append(metric(
            "forward_lean",
            "Forward lean",
            cycle,
            position,
            frame,
            penalty,
            f"Forward lean: {lean:.1f}°",
            f"Pass range: {FORWARD_LEAN_MIN:.0f}° to {FORWARD_LEAN_MAX:.0f}°",
            (
                "The pelvis-to-neck trunk line was compared with vertical. "
                + (
                    "The angle is outside the allowed range."
                    if penalty
                    else "The angle is within the allowed range."
                )
            ),
        ))

    return output


def draw_upper_annotation(
    source_image: Path,
    destination: Path,
    metric_result: dict,
    frame_data: dict,
    joint_names: list[str],
):
    image = Image.open(source_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = load_font(24)
    small = load_font(18)
    tiny = load_font(15)
    idx = {name: i for i, name in enumerate(joint_names)}
    k = frame_data["kpts_2d"]

    def pt(name):
        if name not in idx:
            return None
        value = k[idx[name]]
        return tuple(value) if _valid_2d(value) else None

    orange = (255, 130, 0)
    gray = (75, 85, 99)
    dark = (28, 28, 28)
    sprint_blue = (0, 120, 255)
    left_hip_color = (0, 185, 220)
    right_hip_color = (205, 55, 145)
    red = (190, 35, 35)
    green = (28, 120, 65)
    white = (255, 255, 255)

    def dot(point, fill, label=None):
        if not point:
            return
        x, y = point
        draw.ellipse([x-7, y-7, x+7, y+7], fill=white, outline=fill, width=3)
        if label:
            draw.text((x + 10, y - 18), label, fill=fill, font=tiny, stroke_width=3, stroke_fill=white)

    metric_id = metric_result["metric_id"]

    if metric_id == "trunk_pelvic_rotation":
        ls, rs = pt("left_shoulder"), pt("right_shoulder")
        lh, rh = pt("left_hip"), pt("right_hip")
        if ls and rs:
            draw.line([ls, rs], fill=orange, width=6)
        if lh and rh:
            draw.line([lh, rh], fill=gray, width=6)
        for p, label, color in [
            (ls, "L shoulder", orange), (rs, "R shoulder", orange),
            (lh, "L hip", left_hip_color), (rh, "R hip", right_hip_color),
        ]:
            dot(p, color, label)

        # Top-down orientation inset. Arrows show sprint, pelvis, and chest
        # directions projected onto the horizontal plane.
        inset_w = min(430, max(310, image.width // 3))
        inset_h = 250
        x0 = image.width - inset_w - 18
        y0 = 155
        draw.rounded_rectangle(
            [x0, y0, x0 + inset_w, y0 + inset_h],
            radius=12,
            fill=white,
            outline=gray,
            width=3,
        )
        draw.text((x0 + 14, y0 + 10), "Top-down orientation", fill=dark, font=small)

        center = (x0 + inset_w // 2, y0 + 125)
        arrow_len = min(115, inset_w // 3)
        up = np.asarray(frame_data["world_up"], float)
        sprint = norm(frame_data["sprint_horizontal"])
        lateral = norm(np.cross(up, sprint))

        def plane_xy(vector):
            vector = norm(vector)
            return np.array([
                np.dot(vector, sprint),
                np.dot(vector, lateral),
            ], float)

        def arrow(vector, color, label, y_label, line_width=6):
            v = plane_xy(vector)
            end = (
                center[0] + arrow_len * v[0],
                center[1] - arrow_len * v[1],
            )
            draw.line([center, end], fill=color, width=line_width)
            angle = math.atan2(center[1] - end[1], end[0] - center[0])
            head = 13
            for offset in (2.55, -2.55):
                p = (
                    end[0] - head * math.cos(angle + offset),
                    end[1] + head * math.sin(angle + offset),
                )
                draw.line([end, p], fill=color, width=max(line_width - 1, 3))
            draw.text((x0 + 14, y0 + y_label), label, fill=color, font=tiny)

        arrow(frame_data["sprint_horizontal"], sprint_blue, "Sprint direction", 175, line_width=10)
        arrow(frame_data["pelvis_horizontal"], gray, "Pelvis direction", 197, line_width=6)
        arrow(frame_data["chest_horizontal"], orange, "Chest direction", 219, line_width=6)
        dot(center, dark)

    elif metric_id == "lumbar_extension_review":
        pelvis, spine2, neck = pt("pelvis"), pt("spine2"), pt("neck")
        trailing_side = frame_data.get("trailing_side")
        trailing_value = frame_data.get("trailing_hip_2d")
        trailing_hip = tuple(trailing_value) if _valid_2d(trailing_value) else None

        if pelvis and spine2 and neck:
            draw.line([pelvis, spine2], fill=gray, width=7)
            draw.line([spine2, neck], fill=orange, width=7)
            draw.line(
                [pelvis, (pelvis[0], max(0, pelvis[1] - 360))],
                fill=(110, 110, 110),
                width=3,
            )
            dot(pelvis, gray, "pelvis")
            dot(spine2, orange, "spine2")
            dot(neck, orange, "neck")

        if pelvis and trailing_hip:
            draw.line([trailing_hip, pelvis], fill=red, width=6)
            draw.line(
                [trailing_hip, (pelvis[0], trailing_hip[1])],
                fill=red,
                width=3,
            )
            dot(trailing_hip, red, f"trailing {trailing_side} hip")

    elif metric_id == "forward_lean":
        pelvis, neck = pt("pelvis"), pt("neck")
        if pelvis and neck:
            draw.line([pelvis, neck], fill=orange, width=7)
            draw.line(
                [pelvis, (pelvis[0], max(0, pelvis[1] - 360))],
                fill=gray,
                width=4,
            )
            dot(pelvis, gray, "pelvis")
            dot(neck, orange, "neck")

    box_h = 135
    draw.rounded_rectangle(
        [15, 15, image.width - 15, 15 + box_h],
        radius=10,
        fill=white,
        outline=gray,
        width=2,
    )
    draw.text((28, 24), metric_result["metric_label"], fill=dark, font=font)
    draw.text((28, 62), metric_result["value_text"], fill=dark, font=small)

    if metric_result.get("review_required"):
        result_text = (
            "Review suggested" if metric_result.get("review_suggested")
            else "Analyst review required"
        )
        result_color = red if metric_result.get("review_suggested") else orange
    else:
        result_text = "Penalty +1" if metric_result["penalty"] else "Pass"
        result_color = red if metric_result["penalty"] else green

    draw.text((28, 103), result_text, fill=result_color, font=small)

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, quality=94)
