"""Create a transferable archive of the complete standalone pose runtime."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


REQUIRED = (
    Path("jade_pose_server.exe"),
    Path("models/pose_landmarker_full.task"),
    Path("models/hand_landmarker.task"),
)
HAND_ONLY_REQUIRED = (Path("jade_pose_server.exe"), Path("models/hand_landmarker.task"))


def package(runtime: Path, archive: Path, hand_only: bool = False) -> None:
    runtime = runtime.resolve()
    archive = archive.resolve()
    for relative in HAND_ONLY_REQUIRED if hand_only else REQUIRED:
        if not (runtime / relative).is_file():
            raise FileNotFoundError(runtime / relative)
    if not (runtime / "_internal").is_dir():
        raise FileNotFoundError(runtime / "_internal")
    if hand_only and (runtime / "models/pose_landmarker_full.task").exists():
        raise ValueError("Hand-only runtime still contains the full-body model")
    if archive.is_relative_to(runtime):
        raise ValueError("Archive cannot be placed inside the runtime directory")

    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(runtime.rglob("*")):
            if path.is_file():
                bundle.write(path, (Path("jade_pose_server") / path.relative_to(runtime)).as_posix())

    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        "{}  {}\n".format(digest.hexdigest(), archive.name), encoding="ascii"
    )
    print("Archive: {}".format(archive))
    print("SHA256: {}".format(digest.hexdigest()))


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Package jade_pose_server runtime")
    parser.add_argument("--runtime", type=Path, default=project / "dist/jade_pose_server")
    parser.add_argument("--archive", type=Path, default=project / "dist/jade_pose_server.zip")
    parser.add_argument("--hand-only", action="store_true")
    args = parser.parse_args()
    package(args.runtime, args.archive, hand_only=args.hand_only)


if __name__ == "__main__":
    main()
