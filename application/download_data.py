#!/usr/bin/env python3
"""Download and verify the public ConvoKit release of the CaSiNo corpus."""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


DATA_URL = (
    "https://zissou.infosci.cornell.edu/convokit/datasets/"
    "casino-corpus/casino-corpus.zip"
)
EXPECTED_SHA256 = "71448281f192ffcecfa05877e0aecb0b2f7c68db763b8cf8dc5f9d1010ece387"
EXPECTED_MEMBERS = {
    "casino-corpus/corpus.json",
    "casino-corpus/index.json",
    "casino-corpus/speakers.json",
    "casino-corpus/conversations.json",
    "casino-corpus/utterances.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_archive(path: Path) -> None:
    observed = sha256(path)
    if observed != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {EXPECTED_SHA256}, got {observed}."
        )
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        missing = EXPECTED_MEMBERS - members
        if missing:
            raise RuntimeError(f"Archive is missing expected files: {sorted(missing)}")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"Archive member failed CRC validation: {corrupt}")


def validate_extracted_members(archive_path: Path, raw_dir: Path) -> None:
    """Require every extracted input to match the verified archive byte for byte."""
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(EXPECTED_MEMBERS):
            extracted_path = raw_dir / member
            if not extracted_path.is_file():
                raise RuntimeError(f"Missing extracted corpus member: {extracted_path}")
            if extracted_path.read_bytes() != archive.read(member):
                raise RuntimeError(
                    "Extracted corpus member differs from the verified archive: "
                    f"{extracted_path}"
                )


def main() -> int:
    application_dir = Path(__file__).resolve().parent
    raw_dir = application_dir / "data" / "raw"
    archive_path = raw_dir / "casino-corpus.zip"
    corpus_dir = raw_dir / "casino-corpus"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        validate_archive(archive_path)
        print(f"Using verified archive: {archive_path}")
    else:
        with tempfile.TemporaryDirectory(prefix="casino-download-") as temp_dir:
            temporary_archive = Path(temp_dir) / "casino-corpus.zip"
            print(f"Downloading {DATA_URL}")
            urllib.request.urlretrieve(DATA_URL, temporary_archive)
            validate_archive(temporary_archive)
            shutil.move(str(temporary_archive), archive_path)
        print(f"Saved verified archive: {archive_path}")

    expected_paths = [raw_dir / member for member in EXPECTED_MEMBERS]
    if not any(path.exists() for path in expected_paths):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(raw_dir)
        validate_extracted_members(archive_path, raw_dir)
        print(f"Extracted corpus: {corpus_dir}")
    elif not all(path.exists() for path in expected_paths):
        raise RuntimeError(
            "The extracted corpus is incomplete. Move or remove only "
            f"{corpus_dir} and rerun this downloader."
        )
    else:
        validate_extracted_members(archive_path, raw_dir)
        print(f"Using verified extracted corpus: {corpus_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
