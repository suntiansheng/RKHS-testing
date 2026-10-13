"""Validate the complete GCM run and publish matched NORD/GCM curves.

The original summaries remain immutable. NORD is refreshed from the same
score vectors as GCM, retaining the original 170-cell null-screen membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import sklearn
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "simulation"))
from io_utils import atomic_write_csv, atomic_write_json, file_hash
from summarize import attach_size_gate, size_audit, wilson_interval

KEYS = ["design", "scenario", "p", "delta", "learner", "method", "alpha"]
CURVE_KEYS = ["design", "scenario", "p", "learner", "method"]


def paired_nord_summary(gcm, raw):
    nord = gcm.drop(columns=[c for c in gcm if c.startswith("size_")]).copy()
    nord["method"] = "orthogonal_nosplit"
    nord["rejection_rate"] = nord.pop("paired_nord_rejection_rate")
    nord["monte_carlo_se"] = np.sqrt(
        nord.rejection_rate * (1 - nord.rejection_rate) / nord.n_valid)
    bounds = [wilson_interval(round(row.rejection_rate * row.n_valid), row.n_valid)
              for row in nord.itertuples()]
    nord[["wilson_lower", "wilson_upper"]] = bounds
    observed = raw.groupby([k for k in KEYS if k != "method"]).nord_reject.mean()
    reported = nord.set_index([k for k in KEYS if k != "method"]).rejection_rate
    np.testing.assert_allclose(reported.sort_index(), observed.sort_index(), atol=1e-14)
    return nord


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/gcm_comparison")
    args = parser.parse_args()
    run = args.run
    validation = json.loads((run / "validation.json").read_text())
    if not validation["complete"]:
        raise ValueError("Publication requires the full 115,200-case grid")
    protocol = json.loads((run / "protocol.json").read_text())
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    for name, expected in protocol["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Frozen simulation source changed: {name}")
    for name, expected in validation["source_files"].items():
        if file_hash(run / name) != expected:
            raise ValueError(f"Result hash mismatch: {name}")
    checkpoints = sorted((run / "checkpoints").glob("*.json"))
    if len(checkpoints) != 1200:
        raise ValueError("Expected 600 null and 600 power checkpoints")
    for path in checkpoints:
        meta = json.loads(path.read_text())
        expected_rows = 16 if meta["key"].endswith("_null") else 176
        if (meta["fingerprint"] != fingerprint or meta["rows"] != expected_rows
                or file_hash(run / "shards" / (meta["key"] + ".csv.gz")) != meta["sha256"]):
            raise ValueError(f"Invalid checkpoint: {path.name}")
    raw = pd.read_csv(run / "raw.csv.gz")
    gcm = pd.read_csv(run / "summary.csv")
    sizes = pd.read_csv(run / "null_size.csv")
    if (len(raw) != 115200 or len(gcm) != 384 or len(sizes) != 32
            or raw.duplicated(KEYS + ["rep"]).any()
            or not raw.status.eq("ok").all()
            or not raw.score_n.eq(400).all()
            or not gcm.n_valid.eq(300).all()
            or not gcm.n_failed.eq(0).all()):
        raise ValueError("Incomplete, duplicate, or failed simulation cells")
    if not raw.groupby(KEYS).rep.apply(lambda s: set(s) == set(range(1,301))).all():
        raise ValueError("Each cell must contain replications 1 through 300")
    denominator = raw.score_sd * np.sqrt((raw.score_n - 1) / raw.score_n)
    statistic = np.sqrt(raw.score_n) * raw.score_mean / denominator
    np.testing.assert_allclose(raw.statistic, statistic, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(raw.p_value, 2 * norm.sf(abs(statistic)), atol=1e-12)
    np.testing.assert_array_equal(raw.reject, (raw.p_value <= raw.alpha).astype(int))
    np.testing.assert_array_equal(raw.nord_reject,
                                 (raw.nord_p_value <= raw.alpha).astype(int))
    observed = raw.groupby(KEYS).reject.mean()
    np.testing.assert_allclose(gcm.set_index(KEYS).rejection_rate.sort_index(),
                               observed.sort_index(), atol=1e-14)
    nord = paired_nord_summary(gcm, raw)

    # Refresh only the displayed NORD null counts, preserving all 170 members
    # of the original family. Yan and the 40-cell split family are untouched.
    archived = pd.read_csv(ROOT / "results/frozen_summary.csv")
    archived_null = archived[archived.delta.eq(0)].copy()
    replacement = nord[nord.delta.eq(0)].set_index(CURVE_KEYS)
    # Keep only size inputs: nuisance diagnostics and confidence intervals from
    # the old fit must not be carried beside refreshed rejection counts.
    updated_null = archived_null[KEYS + ["n_valid", "n_failed", "rejection_rate"]].copy()
    for index, row in updated_null.iterrows():
        key = tuple(row[k] for k in CURVE_KEYS)
        if key in replacement.index:
            updated_null.loc[index, "rejection_rate"] = replacement.loc[key, "rejection_rate"]
    if len(updated_null) != 170:
        raise ValueError("Original null-screen family membership changed")
    audit = size_audit(updated_null, family="main_170_cells_paired_nord_update")
    nord = attach_size_gate(nord, audit)
    old_flags = archived_null.set_index(CURVE_KEYS).size_holm_flag_0_05.sort_index()
    new_flags = audit.set_index(CURVE_KEYS).holm_flag_0_05.sort_index()
    changed_flags = int((old_flags != new_flags).sum())

    paired = []
    for values, group in raw.groupby([k for k in KEYS if k != "method"], sort=True):
        difference = group.nord_reject - group.reject
        estimate = difference.mean()
        se = difference.std(ddof=1) / np.sqrt(len(group))
        paired.append(dict(zip([k for k in KEYS if k != "method"], values)) | dict(
            n_valid=len(group), nord_rejection_rate=group.nord_reject.mean(),
            gcm_rejection_rate=group.reject.mean(), nord_minus_gcm=estimate,
            paired_se=se, lower=estimate - norm.ppf(.975)*se,
            upper=estimate + norm.ppf(.975)*se))

    tables = {
        "gcm_summary.csv": gcm,
        "gcm_paired_nord_summary.csv": nord,
        "gcm_null_size.csv": sizes,
        "gcm_main_null_audit.csv": audit,
        "gcm_paired_differences.csv": pd.DataFrame(paired),
    }
    for name, frame in tables.items():
        atomic_write_csv(ROOT / "results" / name, frame)
    phase_progress = {}
    for name in ("null_phase_progress.json", "progress.json"):
        if (run / name).exists():
            progress = json.loads((run / name).read_text())
            phase_progress[progress["phase"]] = progress
    report = dict(validation, publication_complete=True,
        checkpoint_count=len(checkpoints), changed_original_null_decisions=changed_flags,
        gcm_null_min=float(sizes.rejection_rate.min()),
        gcm_null_max=float(sizes.rejection_rate.max()),
        gcm_null_flags=int(sizes.holm_flag_0_05.sum()),
        paired_nord_null_min=float(nord[nord.delta.eq(0)].rejection_rate.min()),
        paired_nord_null_max=float(nord[nord.delta.eq(0)].rejection_rate.max()),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
            numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
            sklearn=sklearn.__version__, torch=torch.__version__, device="cpu"),
        published_sha256={name:file_hash(ROOT/"results"/name) for name in tables},
        run_metadata_sha256={name:file_hash(run/name) for name in
            ("protocol.json", "calibrations.json", "publication_plan.json")
            if (run/name).exists()},
        phase_progress=phase_progress,
        measured_phase_seconds=sum(p["elapsed_seconds"] for p in phase_progress.values()),
        publication_source_sha256=file_hash(Path(__file__)),
        note="Figures use matched rerun NORD and GCM. Other curves use archived estimates. "
             "Historical null diagnostics are retained in the output tables.")
    atomic_write_json(run / "publication_validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
