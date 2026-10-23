#!/usr/bin/env python3
"""Download suite-specific OpenVLA-OFT checkpoints used by cross-suite evals."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


FULL_OFT_REPOS = {
    "libero_object": "moojink/openvla-7b-oft-finetuned-libero-object",
    "libero_goal": "moojink/openvla-7b-oft-finetuned-libero-goal",
    "libero_10": "moojink/openvla-7b-oft-finetuned-libero-10",
}
OFT_W_REPO = "Sylvest/openvla-7b-oft-finetuned-libero-without-wrist"


def download_repo(repo_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] repo={repo_id}")
    print(f"[download] local_dir={local_dir}")
    if allow_patterns:
        print(f"[download] allow_patterns={allow_patterns}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        resume_download=True,
    )
    print(f"[done] {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        default="/mnt/beegfs/home/caijiasheng25/models",
        help="Directory where checkpoints should be stored.",
    )
    parser.add_argument(
        "--suites",
        default="libero_object,libero_goal,libero_10",
        help="Comma-separated suite names to download.",
    )
    parser.add_argument("--skip-full-oft", action="store_true")
    parser.add_argument("--skip-oft-w", action="store_true")
    args = parser.parse_args()

    model_root = Path(args.model_root).expanduser()
    suites = [item.strip() for item in args.suites.split(",") if item.strip()]
    unknown = [suite for suite in suites if suite not in FULL_OFT_REPOS]
    if unknown:
        raise SystemExit(f"Unknown suites: {unknown}; allowed={sorted(FULL_OFT_REPOS)}")

    if not args.skip_full_oft:
        for suite in suites:
            repo_id = FULL_OFT_REPOS[suite]
            suffix = repo_id.rsplit("/", 1)[1]
            download_repo(repo_id, model_root / suffix)

    if not args.skip_oft_w:
        download_repo(
            OFT_W_REPO,
            model_root / "openvla-7b-oft-finetuned-libero-without-wrist",
            allow_patterns=[f"{suite}/*" for suite in suites],
        )


if __name__ == "__main__":
    main()
