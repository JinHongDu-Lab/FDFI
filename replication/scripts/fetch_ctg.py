#!/usr/bin/env python3
"""Fetch the exact UCI CTG spreadsheet used by the source notebook."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00193/CTG.xls"
SHA256 = "d6aa62e82625e59f9d5b05bc8fc52af9ea67bdf2a23f1faca8511d5618fcb421"
DESTINATION = Path(__file__).resolve().parents[1] / "data" / "CTG.xls"


def main() -> int:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".xls.part")
    if DESTINATION.exists():
        actual = hashlib.sha256(DESTINATION.read_bytes()).hexdigest()
        if actual == SHA256:
            print(f"Using existing verified CTG input: {DESTINATION}")
            return 0
        print(
            f"CTG destination conflict: {DESTINATION} exists with SHA-256 {actual}; "
            f"expected {SHA256}. Refusing to overwrite it.",
            file=sys.stderr,
        )
        return 1
    try:
        temporary.unlink(missing_ok=True)
        urllib.request.urlretrieve(URL, temporary)
        actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if actual != SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch: expected {SHA256}, got {actual}")
        temporary.replace(DESTINATION)
        print(f"Saved verified CTG input: {DESTINATION}")
        return 0
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        print(f"CTG download failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
