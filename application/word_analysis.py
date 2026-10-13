#!/usr/bin/env python3
"""Recompute the supplementary opening-word contrasts from the public corpus."""

import argparse
import json
from pathlib import Path

import analyze
import core
import plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path,
                        default=analyze.APPLICATION_DIR / "data/raw/casino-corpus")
    parser.add_argument("--output-dir", type=Path,
                        default=analyze.APPLICATION_DIR.parent / "runs/casino-words")
    args = parser.parse_args()
    archive = args.corpus_dir.parent / "casino-corpus.zip"
    if not archive.is_file() or core.file_sha256(archive) != core.ARCHIVE_SHA256:
        raise ValueError("Run application/download_data.py to obtain the verified corpus")
    member_hashes = core.verify_extracted_corpus(archive, args.corpus_dir)
    records, _ = analyze.load_integrative_potential_records(args.corpus_dir)
    sample, sample_hash = analyze.frozen_main_sample([r for r in records if r["eligible_extreme"]])
    words = analyze.differential_word_statistics(sample,
        minimum_document_frequency=analyze.MINIMUM_WORD_DOCUMENT_FREQUENCY, bootstrap_draws=2000)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    words.to_csv(args.output_dir / "word_log_odds.csv", index=False)
    plot.configure_style()
    plot.save_figure(plot.make_manuscript_word_figure(words), args.output_dir, "casino_word_contrasts_v5")
    metadata = {"sample_sha256": sample_hash, "corpus_member_sha256": member_hashes,
                "bootstrap_draws": 2000, "minimum_document_frequency": analyze.MINIMUM_WORD_DOCUMENT_FREQUENCY,
                "word_table_sha256": core.file_sha256(args.output_dir / "word_log_odds.csv")}
    (args.output_dir / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved word summaries and figure to {args.output_dir}")


if __name__ == "__main__":
    main()
