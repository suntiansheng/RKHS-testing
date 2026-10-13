#!/usr/bin/env python3
"""Plot the CaSiNo satisfaction results and descriptive word contrasts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_VERSION = "casino-integrative-potential-figures-v1"
PNG_DPI = 300
RESULTS_FIGSIZE = (6.25, 5.0)

GROUP_ORDER = ("aligned", "maximal")
GROUP_D = {"aligned": 0, "maximal": 1}
GROUP_COLORS = {"aligned": "#D55E00", "maximal": "#0072B2"}
GROUP_LABELS = {
    "aligned": "Aligned priorities ($M=36$)",
    "maximal": "Maximal integrative potential ($M=42$)",
}
LEARNER_COLORS = {
    "ridge": "#0072B2",
    "lasso": "#009E73",
    "random_forest": "#CC79A7",
}

INPUT_FILES = {
    "sample": "integrative_potential_sample.csv",
    "tests": "integrative_potential_rkhs_results.csv",
    "selection": "integrative_potential_selection_sensitivity.csv",
    "nuisance": "integrative_potential_nuisance_predictions.csv",
    "words": "integrative_potential_word_log_odds.csv",
}


def configure_style() -> None:
    """Set typography for a six-inch-wide manuscript figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _read_csv(results_dir: Path, key: str, required: Iterable[str]) -> pd.DataFrame:
    path = results_dir / INPUT_FILES[key]
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen input is missing: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{path.name} is empty")
    return frame


def _finite_numeric(frame: pd.DataFrame, columns: Iterable[str], filename: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{filename}.{column} must contain only finite numbers")
        frame[column] = values


def _boolean_series(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    if not normalized.isin(mapping).all():
        raise ValueError(f"{label} must be boolean (true/false or 1/0)")
    return normalized.map(mapping).astype(bool)


def load_inputs(results_dir: Path) -> tuple[pd.DataFrame, ...]:
    """Read and validate the five frozen CSV inputs."""
    sample = _read_csv(
        results_dir,
        "sample",
        ("dialogue_id", "D", "population", "mean_satisfaction"),
    )
    tests = _read_csv(
        results_dir,
        "tests",
        ("learner", "regime", "outcome", "p_value", "bootstrap_draws"),
    )
    selection = _read_csv(results_dir, "selection", ("selection", "p_value"))
    nuisance = _read_csv(
        results_dir,
        "nuisance",
        ("dialogue_id", "D", "propensity_ridge_crossfit"),
    )
    words = _read_csv(
        results_dir,
        "words",
        (
            "term",
            "log_odds_maximal_minus_aligned",
            "z_score",
            "abs_z_score",
            "bootstrap_ci_lower",
            "bootstrap_ci_upper",
            "favored_population",
        ),
    )

    sample_name = INPUT_FILES["sample"]
    _finite_numeric(sample, ("D", "mean_satisfaction"), sample_name)
    sample["D"] = sample["D"].astype(int)
    sample["population"] = sample["population"].astype(str).str.strip().str.lower()
    if sample["dialogue_id"].astype(str).duplicated().any():
        raise ValueError(f"{sample_name}.dialogue_id must be unique")
    if set(sample["population"]) != set(GROUP_ORDER):
        raise ValueError(f"{sample_name}.population must contain aligned and maximal")
    expected_d = sample["population"].map(GROUP_D).to_numpy(dtype=int)
    if not np.array_equal(sample["D"].to_numpy(dtype=int), expected_d):
        raise ValueError(f"{sample_name} must map aligned to D=0 and maximal to D=1")
    if not sample["mean_satisfaction"].between(1, 5).all():
        raise ValueError(f"{sample_name}.mean_satisfaction must lie in [1, 5]")

    tests_name = INPUT_FILES["tests"]
    _finite_numeric(tests, ("p_value", "bootstrap_draws"), tests_name)
    if (
        not tests["p_value"].between(0, 1, inclusive="right").all()
        or (tests["p_value"] <= 0).any()
    ):
        raise ValueError(f"{tests_name}.p_value must lie in (0, 1]")
    if (tests["bootstrap_draws"] < 1).any():
        raise ValueError(f"{tests_name}.bootstrap_draws must be positive")
    tests["learner"] = tests["learner"].astype(str).str.strip()
    tests["regime"] = tests["regime"].astype(str).str.strip()
    tests["outcome"] = tests["outcome"].astype(str).str.strip()
    if "bandwidth_spec" in tests:
        tests["bandwidth_spec"] = tests["bandwidth_spec"].astype(str).str.strip()
        tests = tests.loc[tests["bandwidth_spec"] == "fixed_h_1"].copy()
        if tests.empty:
            raise ValueError(f"{tests_name} contains no fixed_h_1 rows")
    if tests.duplicated(["outcome", "learner", "regime"]).any():
        raise ValueError(
            f"{tests_name} must have unique (outcome, learner, regime) rows"
        )
    if "is_primary" in tests:
        tests["is_primary"] = _boolean_series(
            tests["is_primary"], f"{tests_name}.is_primary"
        )

    selection_name = INPUT_FILES["selection"]
    _finite_numeric(selection, ("p_value",), selection_name)
    if (selection["p_value"] <= 0).any() or (selection["p_value"] > 1).any():
        raise ValueError(f"{selection_name}.p_value must lie in (0, 1]")
    selection["selection"] = selection["selection"].astype(str).str.strip()
    if selection["selection"].duplicated().any():
        raise ValueError(f"{selection_name}.selection must be unique")
    if "is_primary" in selection:
        selection["is_primary"] = _boolean_series(
            selection["is_primary"], f"{selection_name}.is_primary"
        )

    nuisance_name = INPUT_FILES["nuisance"]
    _finite_numeric(nuisance, ("D", "propensity_ridge_crossfit"), nuisance_name)
    nuisance["D"] = nuisance["D"].astype(int)
    nuisance["dialogue_id"] = nuisance["dialogue_id"].astype(str)
    sample_ids = sample["dialogue_id"].astype(str)
    if nuisance["dialogue_id"].duplicated().any():
        raise ValueError(f"{nuisance_name}.dialogue_id must be unique")
    if set(nuisance["dialogue_id"]) != set(sample_ids):
        raise ValueError(f"{nuisance_name} dialogue IDs must exactly match the sample")
    if not nuisance["D"].isin((0, 1)).all():
        raise ValueError(f"{nuisance_name}.D must contain only 0 and 1")
    d_by_id = dict(zip(sample_ids, sample["D"].astype(int)))
    if any(d_by_id[row.dialogue_id] != int(row.D) for row in nuisance.itertuples()):
        raise ValueError(f"{nuisance_name}.D disagrees with the sample")
    if not nuisance["propensity_ridge_crossfit"].between(0, 1).all():
        raise ValueError(
            f"{nuisance_name}.propensity_ridge_crossfit must lie in [0, 1]"
        )

    words_name = INPUT_FILES["words"]
    numeric_word_columns = (
        "log_odds_maximal_minus_aligned",
        "z_score",
        "abs_z_score",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
    )
    _finite_numeric(words, numeric_word_columns, words_name)
    words["term"] = words["term"].astype(str).str.strip()
    words["favored_population"] = (
        words["favored_population"].astype(str).str.strip().str.lower()
    )
    if words["term"].eq("").any() or words["term"].duplicated().any():
        raise ValueError(f"{words_name}.term must be nonempty and unique")
    if not set(words["favored_population"]).issubset(GROUP_ORDER):
        raise ValueError(f"{words_name}.favored_population must be aligned or maximal")
    if set(words["favored_population"]) != set(GROUP_ORDER):
        raise ValueError(f"{words_name} must contain terms favoring both populations")
    if not np.allclose(words["abs_z_score"], np.abs(words["z_score"]), atol=1e-10):
        raise ValueError(f"{words_name}.abs_z_score must equal abs(z_score)")
    if (words["bootstrap_ci_lower"] > words["bootstrap_ci_upper"]).any():
        raise ValueError(
            f"{words_name} has a lower confidence limit above its upper limit"
        )
    sign_population = np.where(
        words["log_odds_maximal_minus_aligned"] >= 0, "maximal", "aligned"
    )
    if not np.array_equal(sign_population, words["favored_population"].to_numpy()):
        raise ValueError(
            f"{words_name}.favored_population must agree with the log-odds sign"
        )
    for population in GROUP_ORDER:
        if (words["favored_population"] == population).sum() < 10:
            raise ValueError(
                f"{words_name} needs at least 10 terms favoring {population}"
            )

    sample["dialogue_id"] = sample["dialogue_id"].astype(str)
    return sample, tests, selection, nuisance, words


def _display_name(value: str) -> str:
    return {
        "no_split": "Full sample",
        "ridge": "Ridge",
        "lasso": "Lasso",
        "random_forest": "Forest",
        "aligned": "Aligned",
        "maximal": "Maximal",
    }.get(value, value.replace("_", " "))


def _panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", pad=5)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(values)
    probability = np.arange(1, len(ordered) + 1, dtype=float) / len(ordered)
    return ordered, probability


def draw_satisfaction(ax: plt.Axes, sample: pd.DataFrame) -> None:
    for population in GROUP_ORDER:
        values = sample.loc[
            sample["population"] == population, "mean_satisfaction"
        ].to_numpy(dtype=float)
        x, y = _ecdf(values)
        ax.step(
            x,
            y,
            where="post",
            color=GROUP_COLORS[population],
            linestyle="--" if population == "aligned" else "-",
            label=f"{_display_name(population).title()} (n={len(values)}, mean={values.mean():.2f})",
        )
    ax.set_xlim(0.95, 5.05)
    ax.set_ylim(0, 1.02)
    ax.set_xticks((1, 2, 3, 4, 5))
    ax.set_xlabel("Mean dyad satisfaction (1–5)")
    ax.set_ylabel("Empirical cumulative probability")
    _panel_title(ax, "a", "Satisfaction distributions")
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.legend(loc="upper left", handlelength=2.2)


def draw_test_pvalues(ax: plt.Axes, tests: pd.DataFrame) -> None:
    selected = tests.loc[
        tests["outcome"].eq("mean_satisfaction") & tests["regime"].eq("no_split")
    ].set_index("learner")
    learners = [name for name in LEARNER_COLORS if name in selected.index]
    if not learners:
        raise ValueError("No full-sample mean-satisfaction tests are available.")
    for index, learner in enumerate(learners):
        row = selected.loc[learner]
        primary = bool(row.get("is_primary", learner == "ridge"))
        ax.scatter(
            float(row["p_value"]),
            index,
            s=42 if primary else 27,
            color=LEARNER_COLORS[learner],
            edgecolor="#111111" if primary else "white",
            linewidth=0.8 if primary else 0.35,
            zorder=3,
        )
    minimum_floor = float((1 / (selected["bootstrap_draws"] + 1)).min())
    xmin = min(minimum_floor, float(selected["p_value"].min())) * 0.70
    ax.set_xscale("log")
    ax.set_xlim(max(xmin, 1e-7), 1.05)
    ax.axvline(0.05, color="#777777", linestyle="--", linewidth=0.9)
    ax.set_yticks(range(len(learners)), [_display_name(item) for item in learners])
    ax.invert_yaxis()
    ax.set_xlabel("Multiplier p-value")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)


def draw_selection_stability(ax: plt.Axes, selection: pd.DataFrame) -> None:
    frame = selection.copy()
    if "is_primary" in frame:
        primary = frame["is_primary"].astype(bool)
    else:
        primary = frame["selection"].str.lower().eq("main")
    if primary.sum() != 1:
        raise ValueError(
            f"{INPUT_FILES['selection']} must identify exactly one primary selection"
        )
    frame["_primary"] = primary
    frame["_original_order"] = np.arange(len(frame))
    frame = frame.sort_values(
        ["_primary", "_original_order"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    x = np.arange(len(frame))
    ax.plot(x, frame["p_value"], color="#8A8A8A", linewidth=0.8, zorder=1)
    ax.scatter(
        x[~frame["_primary"].to_numpy()],
        frame.loc[~frame["_primary"], "p_value"],
        s=21,
        color="#0072B2",
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    ax.scatter(
        x[frame["_primary"].to_numpy()],
        frame.loc[frame["_primary"], "p_value"],
        s=39,
        color="#D55E00",
        edgecolor="#111111",
        linewidth=0.7,
        zorder=3,
        label="designated selection",
    )
    ax.set_yscale("log")
    ymin = max(min(float(frame["p_value"].min()), 0.05) * 0.65, 1e-7)
    ax.set_ylim(ymin, 1.05)
    ax.axhline(0.05, color="#777777", linestyle="--", linewidth=0.9)
    labels = ["main"] + [str(index) for index in range(1, len(frame))]
    ax.set_xticks(x, labels)
    ax.set_xlabel("Fixed outcome-blind selection (main, alternatives)")
    ax.set_ylabel("RKHS p-value")
    _panel_title(ax, "c", "Participant-selection stability")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6, which="major")


def _auc_from_ranks(d: np.ndarray, score: np.ndarray) -> float:
    n1 = int(np.sum(d == 1))
    n0 = int(np.sum(d == 0))
    if n0 == 0 or n1 == 0:
        raise ValueError("AUC requires observations from both populations")
    ranks = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    return float((ranks[d == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def draw_overlap(ax: plt.Axes, nuisance: pd.DataFrame) -> None:
    d = nuisance["D"].to_numpy(dtype=int)
    propensity = nuisance["propensity_ridge_crossfit"].to_numpy(dtype=float)
    for population in GROUP_ORDER:
        values = propensity[d == GROUP_D[population]]
        x, y = _ecdf(values)
        ax.step(
            x,
            y,
            where="post",
            color=GROUP_COLORS[population],
            linestyle="--" if population == "aligned" else "-",
            label=_display_name(population).title(),
        )
    auc = _auc_from_ranks(d, propensity)
    ax.text(
        0.98,
        0.04,
        f"AUC = {auc:.3f}\nrange = [{propensity.min():.3f}, {propensity.max():.3f}]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.25,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"Out-of-fold estimated $P(D=1\mid X)$")
    ax.set_ylabel("Empirical cumulative probability")
    _panel_title(ax, "d", "Ridge propensity overlap")
    ax.grid(color="#E5E5E5", linewidth=0.6)
    ax.legend(loc="upper left")


def _word_selection(words: pd.DataFrame, per_side: int) -> pd.DataFrame:
    parts = []
    for population in GROUP_ORDER:
        group = words[words["favored_population"].eq(population)].sort_values(
            ["abs_z_score", "term"], ascending=[False, True], kind="stable"
        )
        parts.append(group.head(per_side))
    return pd.concat(parts, ignore_index=True)


def draw_word_forest(ax: plt.Axes, words: pd.DataFrame, per_side: int = 6) -> None:
    selected = _word_selection(words, per_side).sort_values(
        ["log_odds_maximal_minus_aligned", "term"], kind="stable"
    )
    selected = selected.reset_index(drop=True)
    y = np.arange(len(selected))
    estimate = selected["log_odds_maximal_minus_aligned"].to_numpy(dtype=float)
    lower = selected["bootstrap_ci_lower"].to_numpy(dtype=float)
    upper = selected["bootstrap_ci_upper"].to_numpy(dtype=float)
    for index, row in selected.iterrows():
        color = GROUP_COLORS[str(row["favored_population"])]
        ax.plot(
            (lower[index], upper[index]),
            (y[index], y[index]),
            color=color,
            linewidth=1.25,
        )
        ax.scatter(
            estimate[index],
            y[index],
            s=24,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    limit_values = np.concatenate((lower, upper, np.array([0.0])))
    span = max(float(limit_values.max() - limit_values.min()), 0.5)
    ax.set_xlim(limit_values.min() - 0.06 * span, limit_values.max() + 0.06 * span)
    ax.axvline(0, color="#777777", linewidth=0.8)
    ax.set_yticks(y, selected["term"])
    ax.set_xlabel(
        "Jeffreys-smoothed document-presence log odds\n(maximal minus aligned)"
    )
    _panel_title(ax, "b", "Direction-balanced top 6 terms by $|z|$")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)


def make_results_figure(
    sample: pd.DataFrame,
    tests: pd.DataFrame,
    selection: pd.DataFrame,
    nuisance: pd.DataFrame,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=RESULTS_FIGSIZE)
    draw_satisfaction(axes[0, 0], sample)
    draw_test_pvalues(axes[0, 1], tests)
    draw_selection_stability(axes[1, 0], selection)
    draw_overlap(axes[1, 1], nuisance)
    for ax, title in zip(
        axes.flat,
        [
            "(a) Satisfaction distributions",
            "(b) Full-sample nuisance choices",
            "(c) Participant selection",
            "(d) Fitted propensities",
        ],
    ):
        ax.set_title(title, loc="left", pad=7)
    axes[0, 0].set_xlabel("Mean dyad satisfaction (1–5)")
    axes[0, 1].set_xlabel("Multiplier p-value")
    axes[1, 0].set_xlabel("Reference and alternative selections")
    axes[1, 0].set_ylabel("Multiplier p-value")
    fig.subplots_adjust(
        left=0.11, right=0.985, bottom=0.11, top=0.94, wspace=0.45, hspace=0.57
    )
    return fig


def make_language_figure(words: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    draw_word_forest(ax, words)
    ax.set_title("Selected opening-word contrasts", loc="left")
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.19, top=0.90)
    return fig


def make_manuscript_word_figure(words: pd.DataFrame) -> plt.Figure:
    """Compact opening-word comparison used in the supplement."""
    fig, ax = plt.subplots(figsize=(6.25, 3.0))
    draw_word_forest(ax, words)
    ax.set_title("", loc="left")
    ax.set_xlabel("Opening-word log odds (maximal potential minus aligned priorities)")
    ax.tick_params(axis="y", labelsize=9)
    fig.subplots_adjust(left=0.15, right=0.975, bottom=0.18, top=0.94)
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    common_metadata = {
        "Title": stem,
        "Author": "CaSiNo integrative-potential analysis",
        "Subject": "Publication figure",
        "Creator": PLOT_VERSION,
    }
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.1,
        metadata={
            **common_metadata,
            "Producer": PLOT_VERSION,
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.1,
        metadata={"Software": PLOT_VERSION},
    )
    plt.close(fig)
    return pdf_path, png_path


def create_publication_figures(results_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    configure_style()
    sample, tests, selection, nuisance, words = load_inputs(results_dir)
    tests = tests.loc[tests["regime"].eq("no_split")].copy()
    outputs: list[Path] = []
    outputs.extend(
        save_figure(
            make_results_figure(sample, tests, selection, nuisance),
            output_dir,
            "integrative_potential_results",
        )
    )
    outputs.extend(
        save_figure(
            make_language_figure(words),
            output_dir,
            "integrative_potential_language",
        )
    )
    return tuple(outputs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing the five frozen integrative-potential CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure destination (default: RESULTS_DIR/figures).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else results_dir / "figures"
    )
    outputs = create_publication_figures(results_dir, output_dir)
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
