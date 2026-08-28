"""Shared plotting helpers for the soma/bouton pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

from analysis.dendrites_pipeline.dendrites_pipeline import is_significant_row
from analysis.shared.shared_boxplots import plot_boxplot_series, plot_grouped_boxplot_series

try:  # Keep the state palette aligned with the main dendrite pipeline.
    from analysis.dendrites_pipeline.dendrites_pipeline import (  # type: ignore
        state_display_color as _main_state_display_color,
        state_family_label as _main_state_family_label,
    )
except Exception:  # pragma: no cover - fallback for import edge cases.
    _main_state_display_color = None
    _main_state_family_label = None

STATE_FAMILY_COLORS = {
    "all": "#4c78a8",
    "active_awake": "#4c78a8",
    "active": "#4c78a8",
    "quiet_awake": "#f58518",
    "quiet": "#f58518",
    "nrem": "#54a24b",
    "rem": "#e45756",
}

COMPARTMENT_ACCENTS = {
    "soma": "#0f766e",
    "bouton": "#7c3aed",
}

STATE_PLOT_ORDER = [
    "quiet_awake_blank",
    "nrem_blank",
    "rem_blank",
    "quiet_awake_movies",
    "nrem_movies",
    "rem_movies",
    "quiet_awake_gratings",
    "nrem_gratings",
    "rem_gratings",
    "quiet_awake_zebras",
    "nrem_zebras",
    "rem_zebras",
    "quiet_awake",
    "nrem",
    "rem",
    "active_awake",
]

STATE_PRETTY_LABELS = {
    "quiet_awake_blank": "Quiet Awake Blank",
    "nrem_blank": "Nrem Blank",
    "rem_blank": "Rem Blank",
    "quiet_awake_movies": "Quiet Awake Movies",
    "nrem_movies": "Nrem Movies",
    "rem_movies": "Rem Movies",
    "quiet_awake_gratings": "Quiet Awake Gratings",
    "nrem_gratings": "Nrem Gratings",
    "rem_gratings": "Rem Gratings",
    "quiet_awake_zebras": "Quiet Awake Zebras",
    "nrem_zebras": "Nrem Zebras",
    "rem_zebras": "Rem Zebras",
    "quiet_awake": "Quiet Awake",
    "nrem": "Nrem",
    "rem": "Rem",
    "active_awake": "Active Awake",
}


def canonical_state_label(label):
    text = str(label or "").strip().lower()
    text = text.replace(" ", "_").replace("-", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def pretty_state_label(label):
    key = canonical_state_label(label)
    return STATE_PRETTY_LABELS.get(key, key.replace("_", " " ).title())


def ordered_state_labels(states, *, include_missing_canonical=False):
    seen = {canonical_state_label(state) for state in states}
    seen.discard("")
    ordered = [state for state in STATE_PLOT_ORDER if state in seen]
    extras = sorted(seen.difference(STATE_PLOT_ORDER))
    if include_missing_canonical:
        return list(STATE_PLOT_ORDER) + extras
    return ordered + extras


def _read_frame(rows: Any) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    if isinstance(rows, dict):
        return pd.DataFrame([rows])
    return pd.DataFrame(list(rows))


def _state_family(state: Any) -> str:
    text = "" if state is None else str(state).strip().lower()
    if not text:
        return "all"
    if _main_state_family_label is not None:
        try:
            return str(_main_state_family_label(text))
        except Exception:
            pass
    if text.startswith("active_awake") or text.startswith("active") or text in {"running", "moving"}:
        return "active_awake"
    if text.startswith("quiet") or text in {"still", "wake", "rest"}:
        return "quiet_awake"
    if text.startswith("nrem"):
        return "nrem"
    if text.startswith("rem"):
        return "rem"
    if text in {"all", "total", "overall"}:
        return "all"
    return text


def state_display_color(state: Any) -> str:
    if _main_state_display_color is not None:
        try:
            return str(_main_state_display_color(state))
        except Exception:
            pass
    return STATE_FAMILY_COLORS.get(_state_family(state), "#6b7280")


def _ordered_unique(values: pd.Series) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values.astype(str).tolist():
        key = value.strip().lower()
        if key and key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _display_label_map(frame: pd.DataFrame, state_col: str, label_col: str) -> dict[str, str]:
    if label_col not in frame.columns:
        label_col = state_col
    label_map: dict[str, str] = {}
    for _, row in frame[[state_col, label_col]].dropna().drop_duplicates(subset=[state_col]).iterrows():
        label_map[str(row[state_col]).strip().lower()] = str(row[label_col])
    return label_map


def _day_group_column(frame: pd.DataFrame) -> str | None:
    if "day_id" in frame.columns:
        return "day_id"
    if "date" in frame.columns:
        return "date"
    if "expid" in frame.columns:
        return "expid"
    return None


def _collapse_to_day_level(
    frame: pd.DataFrame,
    *,
    state_col: str,
    value_col: str,
    label_col: str | None = None,
    group_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    day_col = _day_group_column(frame)
    if day_col is None:
        return frame
    cols = [state_col, day_col]
    if label_col and label_col in frame.columns and label_col not in cols:
        cols.append(label_col)
    for column in group_cols:
        if column in frame.columns and column not in cols:
            cols.append(column)
    return frame.groupby(cols, as_index=False)[value_col].mean()


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, svg]


def _style_state_ticks(ax: plt.Axes, labels: list[str], states: list[str]) -> None:
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for tick, state in zip(ax.get_xticklabels(), states):
        tick.set_color(state_display_color(state))
        tick.set_fontweight("bold")


def _unique_roi_count(frame: pd.DataFrame) -> int:
    for column in ("unit_id", "global_soma_id", "global_bouton_id", "roi_key", "roi_id", "soma_id", "bouton_id", "subject_id", "roi_index"):
        if column in frame.columns:
            values = frame[column].dropna().astype(str)
            values = values[values.str.strip() != ""]
            if not values.empty:
                return int(values.nunique())
    return int(len(frame))


def _plot_boxplot(
    frame: pd.DataFrame,
    *,
    state_col: str,
    label_col: str,
    value_col: str,
    output_dir: Path,
    stem: str,
    title: str,
    ylabel: str,
    accent_color: str,
    comparison_rows: list[Mapping[str, Any]] | None = None,
    sample_size_column: str | None = None,
    horizontal: bool = False,
) -> list[Path]:
    if frame.empty:
        return []

    frame = frame.copy()
    frame[state_col] = frame[state_col].map(canonical_state_label)

    if label_col not in frame.columns:
        label_col = state_col

    frame[label_col] = frame[label_col].astype(str)

    states = ordered_state_labels(frame[state_col].dropna().unique(), include_missing_canonical=False)

    values_by_state: list[np.ndarray] = []
    labels: list[str] = []
    top_labels: list[str] = []
    present_states: list[str] = []
    colors: list[str] = []
    annotation_rows: list[dict[str, Any]] = []
    position_lookup: dict[str, int] = {}

    for state in states:
        state_frame = frame.loc[frame[state_col] == state]
        values = pd.to_numeric(state_frame[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        if sample_size_column is not None:
            if sample_size_column not in state_frame.columns:
                raise ValueError(f"plot_state_correlation requires {sample_size_column} for n labels")
            sample_values = state_frame[sample_size_column].dropna().astype(str)
            sample_values = sample_values[sample_values.str.strip() != ""]
            sample_size = int(sample_values.nunique())
        else:
            sample_size = _unique_roi_count(state_frame)
        present_states.append(state)
        position_lookup[state] = len(present_states)
        labels.append(pretty_state_label(state))
        top_labels.append(f"n={sample_size}")
        values_by_state.append(values)
        colors.append(state_display_color(state))

    if not values_by_state:
        return []

    if comparison_rows:
        for row in comparison_rows:
            if not isinstance(row, Mapping):
                continue
            if not is_significant_row(dict(row), p_key="shuffle_p"):
                continue
            x1 = row.get("x1")
            x2 = row.get("x2")
            if x1 is None or x2 is None:
                state_a = canonical_state_label(row.get("state_a") or row.get("state_a_display"))
                state_b = canonical_state_label(row.get("state_b") or row.get("state_b_display"))
                x1 = position_lookup.get(state_a)
                x2 = position_lookup.get(state_b)
            if x1 is None or x2 is None:
                continue
            annotation_rows.append({"x1": float(x1), "x2": float(x2), "shuffle_p": row.get("shuffle_p")})

    return plot_boxplot_series(
        values_by_state,
        labels,
        present_states,
        colors,
        Path(output_dir),
        top_labels=top_labels,
        stem=stem,
        title=title,
        ylabel=ylabel,
        xlabel="State",
        title_color=accent_color,
        label_color_fn=state_display_color,
        edge_color=accent_color,
        comparison_rows=annotation_rows,
        horizontal=horizontal,
    )


def _state_comparison_rows_for_plot(
    comparison_rows: list[Mapping[str, Any]] | None,
    *,
    compartment: str | None = None,
) -> list[Mapping[str, Any]] | None:
    if not comparison_rows:
        return None
    filtered: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    target_compartment = str(compartment).strip().lower() if compartment is not None else None
    for row in comparison_rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("comparison") or "") != "state_pair":
            continue
        row_compartment = str(row.get("compartment") or "all").strip().lower()
        if target_compartment is None:
            if row_compartment != "all":
                continue
        elif row_compartment != target_compartment:
            continue
        state_a = str(row.get("state_a") or row.get("state_a_display") or "").strip().lower()
        state_b = str(row.get("state_b") or row.get("state_b_display") or "").strip().lower()
        if not state_a or not state_b:
            continue
        pair_key = tuple(sorted((state_a, state_b)) + [row_compartment])
        if pair_key in seen:
            continue
        seen.add(pair_key)
        filtered.append(row)
    return filtered or None


def _plot_state_metric(
    rows: Any,
    output_root: Any,
    *,
    value_col: str,
    title_prefix: str,
    ylabel: str,
    stem_prefix: str,
    comparison_rows: list[Mapping[str, Any]] | None = None,
    cohort_label: str = "all",
    state_order: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    if output_root is None:
        raise ValueError("state metric plot needs an output root")
    frame = _read_frame(rows)
    if frame.empty:
        return []
    state_col = "state" if "state" in frame.columns else "state_display"
    label_col = "state_display" if "state_display" in frame.columns else state_col
    if value_col not in frame.columns:
        raise ValueError(f"state metric plot could not find a {value_col} column")
    if "compartment" not in frame.columns and "channel" in frame.columns:
        frame = frame.copy()
        frame["compartment"] = frame["channel"].map({0: "bouton", 1: "soma", "0": "bouton", "1": "soma"})
    generated: list[Path] = []
    output_dir = Path(output_root) / "figures" / "state_activity" / cohort_label
    group_col = "split_group" if "split_group" in frame.columns else ("group" if "group" in frame.columns else None)
    if "compartment" in frame.columns:
        frame = frame.copy()
        frame["compartment"] = frame["compartment"].astype(str).str.strip().str.lower()
        for compartment in ("soma", "bouton"):
            subset = frame.loc[frame["compartment"] == compartment].copy()
            if subset.empty:
                continue
            rows_for_plot = _state_comparison_rows_for_plot(comparison_rows, compartment=compartment)
            if group_col is not None:
                generated.extend(
                    plot_grouped_boxplot_series(
                        subset.to_dict("records"),
                        output_dir,
                        state_col=state_col,
                        value_col=value_col,
                        state_order=list(state_order or []),
                        stem=f"{compartment.title()}_{stem_prefix}",
                        title=f"{compartment.title()} {title_prefix.lower()} by state",
                        ylabel=ylabel,
                        xlabel="State",
                        title_color=COMPARTMENT_ACCENTS[compartment],
                        edge_color=COMPARTMENT_ACCENTS[compartment],
                        group_col=group_col,
                        state_label_col=label_col,
                        state_color_col="state_color" if "state_color" in subset.columns else None,
                        group_label_col=f"{group_col}_display" if f"{group_col}_display" in subset.columns else None,
                        group_color_col=f"{group_col}_color" if f"{group_col}_color" in subset.columns else None,
                        group_rank_col=f"{group_col}_rank" if f"{group_col}_rank" in subset.columns else None,
                        comparison_rows=rows_for_plot,
                    )
                )
            else:
                generated.extend(
                    _plot_boxplot(
                        subset,
                        state_col=state_col,
                        label_col=label_col,
                        value_col=value_col,
                        output_dir=output_dir,
                        stem=f"{compartment.title()}_{stem_prefix}",
                        title=f"{compartment.title()} {title_prefix.lower()} by state",
                        ylabel=ylabel,
                        accent_color=COMPARTMENT_ACCENTS[compartment],
                        comparison_rows=rows_for_plot,
                    )
                )
        return generated
    if group_col is not None:
        return plot_grouped_boxplot_series(
            frame.to_dict("records"),
            output_dir,
            state_col=state_col,
            value_col=value_col,
            state_order=list(state_order or []),
            stem=stem_prefix,
            title=f"{title_prefix} by state",
            ylabel=ylabel,
            xlabel="State",
            title_color="#334155",
            edge_color="#334155",
            group_col=group_col,
            state_label_col=label_col,
            state_color_col="state_color" if "state_color" in frame.columns else None,
            group_label_col=f"{group_col}_display" if f"{group_col}_display" in frame.columns else None,
            group_color_col=f"{group_col}_color" if f"{group_col}_color" in frame.columns else None,
            group_rank_col=f"{group_col}_rank" if f"{group_col}_rank" in frame.columns else None,
            comparison_rows=_state_comparison_rows_for_plot(comparison_rows),
        )
    return _plot_boxplot(
        frame,
        state_col=state_col,
        label_col=label_col,
        value_col=value_col,
        output_dir=output_dir,
        stem=stem_prefix,
        title=f"{title_prefix} by state",
        ylabel=ylabel,
        accent_color="#334155",
        comparison_rows=_state_comparison_rows_for_plot(comparison_rows),
    )


def plot_state_activity(*args: Any, **kwargs: Any) -> list[Path]:
    rows = args[0] if args else kwargs.get("rows")
    output_root = args[1] if len(args) > 1 else kwargs.get("output_root") or kwargs.get("result_root")
    comparison_rows = kwargs.get("comparison_rows")
    cohort_label = kwargs.get("cohort_label", "all")
    state_order = kwargs.get("state_order")
    return _plot_state_metric(
        rows,
        output_root,
        value_col="mean",
        title_prefix="Activity",
        ylabel="Activity",
        stem_prefix="state_summary_boxplots_mean",
        comparison_rows=comparison_rows,
        cohort_label=cohort_label,
        state_order=state_order,
    )


def plot_state_event_frequency(*args: Any, **kwargs: Any) -> list[Path]:
    rows = args[0] if args else kwargs.get("rows")
    output_root = args[1] if len(args) > 1 else kwargs.get("output_root") or kwargs.get("result_root")
    comparison_rows = kwargs.get("comparison_rows")
    cohort_label = kwargs.get("cohort_label", "all")
    state_order = kwargs.get("state_order")
    return _plot_state_metric(
        rows,
        output_root,
        value_col="event_frequency_per_min",
        title_prefix="Event frequency",
        ylabel="Event frequency per min",
        stem_prefix="state_summary_boxplots_event_frequency",
        comparison_rows=comparison_rows,
        cohort_label=cohort_label,
        state_order=state_order,
    )


def plot_state_correlation(*args: Any, **kwargs: Any) -> list[Path]:
    rows = args[0] if args else kwargs.get("rows")
    output_root = args[1] if len(args) > 1 else kwargs.get("output_root") or kwargs.get("result_root")
    comparison_rows = kwargs.get("comparison_rows")
    cohort_label = kwargs.get("cohort_label", "all")
    state_order = kwargs.get("state_order")
    title = kwargs.get("title", "state_summary_boxplots_correlation")
    output_stem = kwargs.get("output_stem", "state_summary_boxplots_correlation")
    if output_root is None:
        raise ValueError("plot_state_correlation needs an output root")
    frame = _read_frame(rows)
    if frame.empty:
        return []
    state_col = "state" if "state" in frame.columns else "state_display"
    label_col = "state_display" if "state_display" in frame.columns else state_col
    value_col = "mean_corr" if "mean_corr" in frame.columns else "corr"
    if value_col not in frame.columns:
        raise ValueError("plot_state_correlation could not find a correlation column")
    group_col = "split_group" if "split_group" in frame.columns else ("group" if "group" in frame.columns else None)
    if group_col is not None:
        return plot_grouped_boxplot_series(
            frame.to_dict("records"),
            Path(output_root) / "figures" / "correlation" / cohort_label,
            state_col=state_col,
            value_col=value_col,
            state_order=list(state_order or []),
            stem=output_stem,
            title=title,
            ylabel="Correlation",
            xlabel="State",
            title_color="#334155",
            edge_color="#334155",
            group_col=group_col,
            state_label_col=label_col,
            state_color_col="state_color" if "state_color" in frame.columns else None,
            group_label_col=f"{group_col}_display" if f"{group_col}_display" in frame.columns else None,
            group_color_col=f"{group_col}_color" if f"{group_col}_color" in frame.columns else None,
            group_rank_col=f"{group_col}_rank" if f"{group_col}_rank" in frame.columns else None,
            comparison_rows=_state_comparison_rows_for_plot(comparison_rows),
        )
    return _plot_boxplot(
        frame,
        state_col=state_col,
        label_col=label_col,
        value_col=value_col,
        output_dir=Path(output_root) / "figures" / "correlation" / cohort_label,
        stem=output_stem,
        title=title,
        ylabel="Correlation",
        accent_color="#334155",
        comparison_rows=_state_comparison_rows_for_plot(comparison_rows),
    )


def plot_lag_heatmap(*args: Any, **kwargs: Any) -> list[Path]:
    rows = args[0] if args else kwargs.get("rows")
    output_root = args[1] if len(args) > 1 else kwargs.get("output_root") or kwargs.get("result_root")
    cohort_label = kwargs.get("cohort_label", "all")
    output_stem = kwargs.get("output_stem", "state_summary_lag_heatmap")
    if output_root is None:
        raise ValueError("plot_lag_heatmap needs an output root")
    frame = _read_frame(rows)
    if frame.empty:
        return []
    state_col = "state" if "state" in frame.columns else "state_display"
    label_col = "state_display" if "state_display" in frame.columns else state_col
    if "lag_s" not in frame.columns:
        raise ValueError("plot_lag_heatmap could not find a lag_s column")
    value_col = "corr" if "corr" in frame.columns else "mean_corr"
    if value_col not in frame.columns:
        raise ValueError("plot_lag_heatmap could not find a correlation column")

    def render_heatmap(frame: pd.DataFrame, *, output_stem_name: str, title: str) -> list[Path]:
        frame = frame.copy()
        frame[state_col] = frame[state_col].astype(str).str.strip().str.lower()
        frame[label_col] = frame[label_col].astype(str)
        frame["lag_s"] = pd.to_numeric(frame["lag_s"], errors="coerce")
        frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
        frame = frame.dropna(subset=["lag_s", value_col])
        grouped = frame.groupby([state_col, "lag_s"], as_index=False)[value_col].mean()
        if grouped.empty:
            return []

        states = _ordered_unique(grouped[state_col])
        label_map = _display_label_map(frame, state_col, label_col)
        labels = [label_map.get(state, state) for state in states]
        lags = sorted(float(lag) for lag in grouped["lag_s"].dropna().unique().tolist())
        matrix = grouped.pivot(index=state_col, columns="lag_s", values=value_col).reindex(index=states, columns=lags)
        data = matrix.to_numpy(dtype=float)
        finite = data[np.isfinite(data)]
        bound = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        if bound == 0.0:
            bound = 1.0

        fig_width = max(14.0, 0.22 * len(lags) + 4.0)
        fig_height = max(6.0, 0.45 * len(states) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound),
        )
        ax.set_title(title, fontsize=20, fontweight="bold", color="#334155", pad=12)
        ax.set_xlabel("Lag (s)", fontsize=18)
        ax.set_ylabel("State", fontsize=18)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=13)
        for tick, state in zip(ax.get_yticklabels(), states):
            tick.set_color(state_display_color(state))
            tick.set_fontweight("bold")

        if lags:
            tick_positions = list(range(0, len(lags), max(1, len(lags) // 8)))
            if (len(lags) - 1) not in tick_positions:
                tick_positions.append(len(lags) - 1)
            zero_index = None
            for idx, lag in enumerate(lags):
                if abs(lag) < 1e-12:
                    zero_index = idx
                    break
            if zero_index is not None and zero_index not in tick_positions:
                tick_positions.append(zero_index)
            tick_positions = sorted(set(tick_positions))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"{lags[index]:g}" for index in tick_positions], fontsize=12)
            if zero_index is not None:
                ax.axvline(zero_index, color="#111827", linestyle="--", linewidth=1.4, alpha=0.65)

        cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_label("Mean correlation", fontsize=16)
        cbar.ax.tick_params(labelsize=12)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        output_dir = Path(output_root) / "figures" / "lag" / cohort_label
        return _save_figure(fig, output_dir, output_stem_name)

    group_col = "split_group" if "split_group" in frame.columns else ("group" if "group" in frame.columns else None)
    if group_col is not None:
        group_series = frame[group_col].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False)
        group_values: list[str] = []
        group_display: dict[str, str] = {}
        for raw, slug in zip(frame[group_col].astype(str).tolist(), group_series.tolist()):
            if not slug:
                continue
            if slug not in group_values:
                group_values.append(slug)
                display = raw.strip().replace("_", " ").replace("-", " ").strip().title()
                group_display[slug] = display or slug
        if len(group_values) > 1:
            generated: list[Path] = []
            for group in group_values:
                subset = frame.loc[group_series == group].copy()
                if subset.empty:
                    continue
                title = f"Axon-soma correlation by state and lag - {group_display.get(group, group)}"
                generated.extend(render_heatmap(subset, output_stem_name=f"{output_stem}_{group}", title=title))
            return generated

    return render_heatmap(frame, output_stem_name=output_stem, title="Axon-soma correlation by state and lag")

