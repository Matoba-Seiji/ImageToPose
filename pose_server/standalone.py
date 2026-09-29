"""Entry point for the bundled, Maya-independent pose server."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pose_server.server import main


def run():
    if getattr(sys, "frozen", False):
        models = Path(sys.executable).resolve().parent / "models"
        os.environ.setdefault(
            "MEDIAPIPE_POSE_MODEL", str(models / "pose_landmarker_full.task")
        )
        os.environ.setdefault(
            "MEDIAPIPE_HAND_MODEL", str(models / "hand_landmarker.task")
        )
    main()


if __name__ == "__main__":
    run()
