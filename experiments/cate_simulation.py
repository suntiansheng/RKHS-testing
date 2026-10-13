"""Run the CATE experiment in independently resumable simulation shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "simulation/configs/cate.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/cate_hierarchical")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--rep-stop", type=int)
    parser.add_argument("--learners", help="Comma-separated subset of configured learners")
    parser.add_argument("--deltas", help="Comma-separated subset of configured amplitudes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    stop = args.rep_stop or int(config["replications"])
    if args.jobs < 1 or args.shard_size < 1 or not 1 <= stop <= int(config["replications"]):
        parser.error("Require positive jobs/shard-size and a valid replication limit")
    learners = args.learners.split(",") if args.learners else config["learners"]
    deltas = [float(a) for a in args.deltas.split(",")] if args.deltas else config["deltas"]
    if len(set(learners)) != len(learners) or not set(learners) <= set(config["learners"]):
        parser.error("Learners must be a distinct subset of the configured learners")
    if len(set(deltas)) != len(deltas) or not set(deltas) <= set(config["deltas"]):
        parser.error("Amplitudes must be a distinct subset of the configured grid")
    output = args.output.resolve()
    jobs = []
    for amplitude in deltas:
        for start in range(1, stop + 1, args.shard_size):
            end = min(stop, start + args.shard_size - 1)
            for learner in learners:
                key = f"a{amplitude:g}_{learner}_r{start:04d}_{end:04d}"
                jobs.append(dict(key=key, amplitude=amplitude, learner=learner,
                                 rep_start=start, rep_stop=end))
    if args.dry_run:
        print(json.dumps(dict(shards=len(jobs), tasks=len(deltas)*len(learners)*stop,
                              jobs=args.jobs, first_shards=jobs[:5]), indent=2))
        return 0

    for directory in (output, output / "shards", output / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "simulation/simcore.py",
               ROOT / "simulation/run_simulation.py", config_path]
    protocol = dict(config=config, sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
                    shards=jobs, python=sys.executable)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol, source, interpreter, or shard grid changed; use another output directory")
    write_json(protocol_path, protocol)
    env = os.environ.copy()
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    def run(job: dict) -> dict:
        command = [sys.executable, str(ROOT / "simulation/run_simulation.py"),
                   "--config", str(config_path), "--output", str(output / "shards" / job["key"]),
                   "--learners", job["learner"], "--deltas", str(job["amplitude"]),
                   "--rep-start", str(job["rep_start"]), "--rep-stop", str(job["rep_stop"]),
                   "--workers", "1", "--device", "cpu", "--checkpoint-every", "10"]
        started = time.monotonic()
        log_path = output / "logs" / f"{job['key']}.log"
        with log_path.open("a") as stream:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, check=False)
        return dict(key=job["key"], returncode=process.returncode,
                    elapsed_seconds=time.monotonic()-started, log=str(log_path))

    started = time.monotonic()
    finished = []
    print(f"CATE grid: {len(jobs)} shards, {len(deltas)*len(learners)*stop} learner/data tasks; {args.jobs} processes", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending = [executor.submit(run, job) for job in jobs]
        for future in as_completed(pending):
            result = future.result()
            finished.append(result)
            progress = dict(shards_requested=len(jobs), shards_finished=len(finished),
                            shards_failed=sum(r["returncode"] != 0 for r in finished),
                            elapsed_seconds=time.monotonic()-started,
                            complete=len(finished)==len(jobs) and all(r["returncode"]==0 for r in finished),
                            finished=finished)
            write_json(output / "progress.json", progress)
            print(f"{len(finished)}/{len(jobs)} {result['key']} exit={result['returncode']} elapsed={progress['elapsed_seconds']:.1f}s", flush=True)
    return 0 if all(r["returncode"] == 0 for r in finished) else 1


if __name__ == "__main__":
    raise SystemExit(main())
