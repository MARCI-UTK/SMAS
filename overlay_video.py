from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def create_skeleton_overlay(
    smpl_json: Path,
    frames_dir: Path,
    output_video: Path,
    fps: float,
) -> Path:
    data = json.loads(Path(smpl_json).read_text())

    overlay_frames = output_video.parent / "_overlay_frames"
    if overlay_frames.exists():
        shutil.rmtree(overlay_frames)
    overlay_frames.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(Path(frames_dir).glob("*.jpg"))
    n = min(len(frame_files), int(data["n_frames"]))

    for i in range(n):
        image = Image.open(frame_files[i]).convert("RGB")
        draw = ImageDraw.Draw(image)
        points = data["frames"][i]["kpts_2d"]

        for a, b in data["skeleton"]:
            pa = points[a]
            pb = points[b]
            if (
                pa and pb
                and pa[0] is not None and pa[1] is not None
                and pb[0] is not None and pb[1] is not None
            ):
                draw.line(
                    [tuple(pa), tuple(pb)],
                    fill=(255, 130, 0),
                    width=5,
                )

        if points:
            pelvis = points[0]
            if pelvis and pelvis[0] is not None and pelvis[1] is not None:
                draw.ellipse(
                    [
                        pelvis[0] - 8,
                        pelvis[1] - 8,
                        pelvis[0] + 8,
                        pelvis[1] + 8,
                    ],
                    fill=(255, 255, 255),
                    outline=(255, 130, 0),
                    width=3,
                )

        image.save(
            overlay_frames / f"{i + 1:06d}.jpg",
            quality=92,
        )

    output_video.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(overlay_frames / "%06d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{fps:.6f}",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return output_video
