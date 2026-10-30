"""Inspect and clean the two original House 1 parts; preserve unsupported gaps."""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

COLS = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def fill_short_gaps(minutes):
    """Fill complete bounded missing runs of one or two minute bins only."""
    filled = minutes.copy()
    log = []
    for c in minutes.columns:
        proposed = minutes[c].interpolate(method="time", limit=2, limit_area="inside")
        for s, e in runs(minutes[c].isna().to_numpy()):
            bounded = s > 0 and e < len(minutes)
            approved = bounded and e - s <= 2
            if approved:
                filled.loc[minutes.index[s:e], c] = proposed.iloc[s:e]
            log.append(
                {
                    "channel": c,
                    "start": minutes.index[s],
                    "end_exclusive": minutes.index[e - 1] + pd.Timedelta(minutes=1),
                    "minutes": int(e - s),
                    "interpolated": bool(approved),
                    "long_outage": bool(e - s > 2),
                }
            )
    return filled, pd.DataFrame(log)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("reports/raw_house1"))
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    files = [a.data_dir / f"RAW_House1_Part{i}.csv" for i in (1, 2)]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    u = pd.to_numeric(d.Unix, errors="coerce")
    parsed = pd.to_datetime(d.Time, errors="coerce", utc=True)
    unix_time = pd.to_datetime(u, unit="s", errors="coerce", utc=True)
    valid_time = unix_time.notna()
    offsets = (parsed - unix_time).dt.total_seconds()
    powers = d[COLS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(powers)
    sub_violation = (
        powers[COLS[1:]].gt(powers.Aggregate, axis=0)
        & finite[COLS[1:]]
        & finite.Aggregate.to_numpy()[:, None]
    )
    complete = finite.all(axis=1)
    stats = {
        "files": [f.name for f in files],
        "source_rows": len(d),
        "invalid_time_text": int(parsed.isna().sum()),
        "invalid_unix": int((~valid_time).sum()),
        "duplicate_unix_rows": int(u[valid_time].duplicated().sum()),
        "backward_steps": int((u.diff() < 0).sum()),
        "time_minus_unix_seconds_counts": offsets.value_counts().to_dict(),
        "text_unix_discrepancy_rows": int((offsets.notna() & offsets.ne(0)).sum()),
        "negative_cells": int((powers < 0).sum().sum()),
        "nonfinite_cells": int((~finite).sum().sum()),
        "appliance_cells_above_3500w": int((powers[COLS[1:]] > 3500).sum().sum()),
        "aggregate_above_25000w": int((powers.Aggregate > 25000).sum()),
        "appliance_gt_aggregate_cells": int(sub_violation.sum().sum()),
        "any_appliance_gt_aggregate_rows": int(sub_violation.any(axis=1).sum()),
        "sum_appliances_gt_aggregate_rows": int(
            (complete & (powers[COLS[1:]].sum(axis=1) > powers.Aggregate)).sum()
        ),
        "sum_comparison_complete_rows": int(complete.sum()),
        "thresholds_w": {"appliance": 3500, "aggregate": 25000},
    }
    del finite, sub_violation, complete
    d = powers.loc[valid_time].copy()
    d["Unix"] = u[valid_time].astype("int64")
    duplicate = d[d.Unix.duplicated(keep=False)]
    first = duplicate.groupby("Unix", sort=False)[COLS].transform("first")
    # Missing values compare equal; compare each excess row against the first source row.
    first = (
        duplicate.drop_duplicates("Unix", keep="first")
        .set_index("Unix")[COLS]
        .reindex(duplicate.Unix)
        .set_axis(duplicate.index)
    )
    equal = (duplicate[COLS].eq(first) | (duplicate[COLS].isna() & first.isna())).all(
        axis=1
    )
    excess = duplicate.Unix.duplicated(keep="first")
    stats["duplicate_identical_rows"] = int((equal & excess).sum())
    stats["duplicate_conflicting_rows"] = int((~equal & excess).sum())
    stats["duplicate_identical_pct"] = (
        100 * stats["duplicate_identical_rows"] / stats["duplicate_unix_rows"]
    )
    stats["duplicate_conflicting_pct"] = (
        100 * stats["duplicate_conflicting_rows"] / stats["duplicate_unix_rows"]
    )
    averages = duplicate.groupby("Unix")[COLS].mean()
    kept = duplicate.drop_duplicates("Unix", keep="last").set_index("Unix")[COLS]
    difference = (kept - averages).abs()
    stats["duplicate_keep_last_vs_mean_MAE_W"] = float(difference.stack().mean())
    stats["duplicate_keep_last_vs_mean_by_channel_W"] = difference.mean().to_dict()
    d = d.sort_values("Unix", kind="stable").drop_duplicates("Unix", keep="last")
    delta = d.Unix.diff().dropna()
    stats.update(
        {
            "unique_rows": len(d),
            "median_interval_s": float(delta.median()),
            "mean_interval_s": float(delta.mean()),
            "p95_interval_s": float(delta.quantile(0.95)),
            "max_gap_seconds": int(delta.max()),
            "intervals_gt_60s": int((delta > 60).sum()),
            "intervals_gt_120s": int((delta > 120).sum()),
            "intervals_gt_300s": int((delta > 300).sum()),
        }
    )
    d.index = pd.to_datetime(d.Unix, unit="s", utc=True)
    stats.update(first_unix_time=str(d.index.min()), last_unix_time=str(d.index.max()))
    before = d[COLS].resample("1min").mean()
    clean = d[COLS].mask(~np.isfinite(d[COLS]) | (d[COLS] < 0))
    stats["removed_appliance_cells_after_dedup"] = int(
        (clean[COLS[1:]] > 3500).sum().sum()
    )
    stats["removed_aggregate_cells_after_dedup"] = int((clean.Aggregate > 25000).sum())
    stats["wm_raw_max_w"] = float(d.Appliance5.max())
    clean[COLS[1:]] = clean[COLS[1:]].mask(clean[COLS[1:]] > 3500)
    clean["Aggregate"] = clean.Aggregate.mask(clean.Aggregate > 25000)
    m = clean.resample("1min").mean()
    filled, log = fill_short_gaps(m)
    log.to_csv(a.output_dir / "channel_gaps.csv", index=False)
    agg_log = log[log.channel.eq("Aggregate")]
    agg_log.to_csv(a.output_dir / "aggregate_gaps.csv", index=False)
    stats.update(
        calendar_minutes=len(m),
        missing_minute_cells_before_fill=int(m.isna().sum().sum()),
        filled_minute_cells=int(log.loc[log.interpolated, "minutes"].sum()),
        filled_channel_gaps=int(log.interpolated.sum()),
        missing_minute_cells_after_fill=int(filled.isna().sum().sum()),
        aggregate_long_outage_count=int(agg_log.long_outage.sum()),
        aggregate_long_outage_minutes=int(
            agg_log.loc[agg_log.long_outage, "minutes"].sum()
        ),
        aggregate_filled_minutes=int(
            agg_log.loc[agg_log.interpolated, "minutes"].sum()
        ),
        short_unbounded_gaps_left_missing=int(
            ((log.minutes <= 2) & ~log.interpolated).sum()
        ),
        wm_clean_max_w=float(filled.Appliance5.max()),
    )
    long_mask = np.zeros(len(m), dtype=bool)
    for s, e in runs(m.Aggregate.isna().to_numpy()):
        if e - s > 2:
            long_mask[s:e] = True
    assert not filled.loc[long_mask, "Aggregate"].notna().any()
    assert filled[COLS[1:]].max().max() <= 3500
    assert filled.Aggregate.max() <= 25000
    assert (
        stats["filled_minute_cells"]
        == stats["missing_minute_cells_before_fill"]
        - stats["missing_minute_cells_after_fill"]
    )
    # Choose a real day containing both a ~16 kW minute glitch and heating activity.
    daily_max = before.Appliance5.resample("1D").max()
    daily_clean_max = filled.Appliance5.resample("1D").max()
    candidates = daily_max[(daily_max > 10000) & (daily_clean_max > 1500)]
    if candidates.empty:
        raise ValueError(
            "No day contains both a large WM glitch and valid heating activity"
        )
    day = (candidates - 16000).abs().idxmin()
    end = day + pd.Timedelta(days=1)
    stats["plot_day_utc"] = str(day)
    stats["plot_before_wm_max_w"] = float(
        before.loc[(before.index >= day) & (before.index < end), "Appliance5"].max()
    )
    stats["plot_after_wm_max_w"] = float(
        filled.loc[(filled.index >= day) & (filled.index < end), "Appliance5"].max()
    )
    fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex=True)
    for row, (c, label) in enumerate(
        [
            ("Aggregate", "Whole home"),
            ("Appliance1", "Fridge"),
            ("Appliance5", "Washing machine"),
        ]
    ):
        for col, (values, heading) in enumerate(
            [(before, "Before cleaning (raw minute means)"), (filled, "After cleaning")]
        ):
            ax = axes[row, col]
            v = values.loc[(values.index >= day) & (values.index < end), c]
            ax.plot(v.index, v, color="#23679a" if col == 0 else "#16806a", lw=1)
            if col == 1:
                changed = v.index[m.loc[v.index, c].isna() & v.notna()]
                ax.scatter(
                    changed,
                    v.loc[changed],
                    color="#e48d13",
                    s=12,
                    label="Interpolated",
                    zorder=3,
                )
            ax.set_ylabel(f"{label}\nPower (W)")
            ax.grid(alpha=0.2)
            ax.set_ylim(bottom=0)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=day.tz))
            if row == 0:
                ax.set_title(heading)
    fig.suptitle(
        f"House 1 — {day.date()} UTC | separate scales expose spike removal; gaps stay blank"
    )
    fig.tight_layout()
    fig.savefig(a.output_dir / "before_after.png", dpi=160)
    plt.close(fig)
    # Separate observed gap examples: do not pretend both occur on the glitch day.
    examples = []
    short = log[log.interpolated]
    if len(short):
        examples.append(("Short gap: estimated values", short.iloc[0]))
    long = agg_log[agg_log.long_outage]
    if len(long):
        examples.append(
            ("Long aggregate outage: left missing", long.sort_values("minutes").iloc[0])
        )
    fig, axes = plt.subplots(len(examples), 1, figsize=(13, 6), squeeze=False)
    for ax, (title, r) in zip(axes[:, 0], examples):
        left, right = r.start - pd.Timedelta(minutes=5), r.end_exclusive + pd.Timedelta(
            minutes=5
        )
        v = filled.loc[left:right, r.channel]
        ax.plot(v.index, v, color="#16806a", marker=".", label="After gap handling")
        ax.scatter(
            m.loc[left:right].index,
            m.loc[left:right, r.channel],
            s=22,
            color="#23679a",
            label="Observed minute mean",
            zorder=3,
        )
        ax.axvspan(
            r.start,
            r.end_exclusive,
            alpha=0.15,
            color="#e48d13",
            label="Missing minute bins",
        )
        ax.set_title(
            f"{title} | {r.channel} | {r.start.date()} UTC | {r.minutes} missing minute(s)"
        )
        ax.set_ylabel("Power (W)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=day.tz))
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(a.output_dir / "gap_examples.png", dpi=160)
    plt.close(fig)
    filled["is_long_outage"] = long_mask
    filled.index.name = "timestamp_utc"
    filled.to_csv(a.output_dir / "cleaned_minutes.csv.gz", compression="gzip")
    (a.output_dir / "audit.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2), flush=True)


"""Render DATA.md from the raw audit, so reruns cannot restore stale prose."""


def raw_data_report(raw):
    rows = [
        ("Raw rows (both parts, before deduplication)", "source_rows"),
        ("Coverage start (UTC)", "first_unix_time"),
        ("Coverage end (UTC)", "last_unix_time"),
        ("Invalid Unix timestamps", "invalid_unix"),
        ("Unparseable text timestamps", "invalid_time_text"),
        ("Text/Unix discrepancies", "text_unix_discrepancy_rows"),
        ("Backward time steps in source order", "backward_steps"),
        ("Duplicate timestamps removed", "duplicate_unix_rows"),
        ("Unique timestamp rows", "unique_rows"),
        ("Median sampling interval (seconds)", "median_interval_s"),
        ("Mean sampling interval (seconds)", "mean_interval_s"),
        ("95th percentile interval (seconds)", "p95_interval_s"),
        ("Maximum timestamp gap (seconds)", "max_gap_seconds"),
        ("Timestamp intervals > 60 seconds", "intervals_gt_60s"),
        ("Timestamp intervals > 120 seconds", "intervals_gt_120s"),
        ("Timestamp intervals > 300 seconds", "intervals_gt_300s"),
        ("Missing/non-finite power cells in raw input", "nonfinite_cells"),
        ("Negative power cells in raw input", "negative_cells"),
        ("Appliance cells > 3,500 W in raw input", "appliance_cells_above_3500w"),
        ("Aggregate cells > 25,000 W in raw input", "aggregate_above_25000w"),
        (
            "Appliance cells > 3,500 W removed after deduplication",
            "removed_appliance_cells_after_dedup",
        ),
        (
            "Aggregate cells > 25,000 W removed after deduplication",
            "removed_aggregate_cells_after_dedup",
        ),
        (
            "Individual appliance > Aggregate violations (cells)",
            "appliance_gt_aggregate_cells",
        ),
        ("Rows with any appliance > Aggregate", "any_appliance_gt_aggregate_rows"),
        (
            "Sum of all appliances > Aggregate (complete rows only)",
            "sum_appliances_gt_aggregate_rows",
        ),
        ("Complete rows eligible for sum comparison", "sum_comparison_complete_rows"),
        ("Calendar minute bins", "calendar_minutes"),
        (
            "Missing channel-minute cells before interpolation",
            "missing_minute_cells_before_fill",
        ),
        ("Short channel gaps interpolated", "filled_channel_gaps"),
        ("Channel-minute cells interpolated", "filled_minute_cells"),
        ("Aggregate minutes interpolated", "aggregate_filled_minutes"),
        (
            "Unbounded short gaps left missing",
            "short_unbounded_gaps_left_missing",
        ),
        ("Aggregate outage runs > 2 missing minutes", "aggregate_long_outage_count"),
        ("Aggregate long-outage minutes left missing", "aggregate_long_outage_minutes"),
        (
            "All missing channel-minute cells after cleaning",
            "missing_minute_cells_after_fill",
        ),
        ("Maximum raw washing-machine power (W)", "wm_raw_max_w"),
        ("Maximum cleaned washing-machine minute mean (W)", "wm_clean_max_w"),
    ]

    def fmt(x):
        return (
            f"{x:,}"
            if isinstance(x, int)
            else f"{x:,.3f}" if isinstance(x, float) else str(x)
        )

    table = "| Finding | Value |\n| --- | ---: |\n" + "\n".join(
        f"| {name} | {fmt(raw[key])} |" for name, key in rows
    )
    offsets = ", ".join(
        f"{float(k):g} seconds: {int(v):,} rows"
        for k, v in raw["time_minus_unix_seconds_counts"].items()
    )
    return f"""# Data and preprocessing

## Dataset overview

[REFIT Electrical Load Measurements](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/), University of Strathclyde, supplies the raw and official cleaned releases. Section 1 uses `RAW_House1_Part1.csv` and `RAW_House1_Part2.csv`. House 1 covers **{raw['first_unix_time']} to {raw['last_unix_time']}**. Source sampling is nominally about 8 seconds and is irregular; the outputs use 1-minute arithmetic means. All power values are in watts (W).

## Raw inspection: House 1

{table}

Raw-input cell counts include duplicated rows. Removal counts are measured after timestamp deduplication, so the two counts need not match. A channel-minute cell is one channel in one minute; it is not a whole-house minute. Text timestamp minus Unix offsets: {offsets}. Timestamp gaps describe absent source rows; channel gaps also include missing or rejected sensor values. Detailed machine-readable records are in [audit.json](reports/raw_house1/audit.json), [aggregate gaps](reports/raw_house1/aggregate_gaps.csv) and [all channel gaps](reports/raw_house1/channel_gaps.csv).

## Cleaning steps and reasons

1. **Use Unix time, sort and deduplicate.** Convert Unix timestamps to UTC, sort stably, and retain the last source row at each duplicate timestamp (`keep='last'`). Unix gives an unambiguous timeline; the text column is audited separately rather than silently mixing its one-second discrepancies or local-time assumptions into the index. This rule is reproducible but does not reconstruct the official REFIT corrections.
2. **Remove invalid and extreme readings before averaging.** Non-finite and negative values become `NaN`. Appliance1–Appliance9 readings above **3,500 W** and Aggregate readings above **25,000 W** also become `NaN`. A UK 13 A plug at 230 V corresponds to about 3,000 W; 3,500 W allows a screening margin. [Electrical Safety First](https://www.electricalsafetyfirst.org.uk/safety-advice/home-and-people/house-maintenance/plugs-and-fuses/) describes the approximately 3,000 W socket rating. The aggregate cutoff allows a margin above 230 V × an assumed 100 A service (23,000 W). These are engineering screening assumptions for this exercise, not verified fuse ratings for House 1 or universal physical limits for every appliance. No statistical cutoff is fitted to hide high but plausible cycles.
3. **Average valid readings within each minute.** Missing bins remain missing; a boundary reading contributes only to its own bin. Nothing is carried forward across an outage. A partially observed minute can still have a mean: resampling alone does not establish full-minute coverage.
4. **Fill only short, bounded gaps.** Interpolate whole runs of at most two missing minute bins, using the values on either side. Leave runs of three or more missing minutes unchanged; interpolation can smooth a real switching event.
5. **Keep outages explicit.** Longer runs remain `NaN`. The exported minute table includes `is_long_outage`, defined as an Aggregate run longer than two missing minute bins. Channel-specific missing runs and reasons for filling or leaving them are exported separately. Sub-meter/aggregate mismatches are counted but are not automatically removed by this raw pipeline: asynchronous measurements can disagree, and the comparison cannot identify the faulty sensor on its own.

## Duplicate sensitivity and clock conventions

Of the {raw['duplicate_unix_rows']:,} excess timestamp rows, {raw['duplicate_identical_pct']:.2f}% have identical powers and missingness to the first source row at that timestamp; {raw['duplicate_conflicting_pct']:.2f}% differ. Mean absolute difference between keep-last and per-timestamp means is {raw['duplicate_keep_last_vs_mean_MAE_W']:.3f} W across finite channel/timestamp pairs in duplicated groups. This comparison uses original power values before outlier masking; missing values compare equal for the identical/conflicting classification.

The raw House 1 start label (2013-10-09 14:06:17) and cleaned start label (13:06:17) differ by 3,600 seconds. The cause is unconfirmed: a distinctive common event has not been matched, so the file-start difference is not evidence of a BST correction. Raw inspection converts the supplied Unix values to UTC; EDA hour/weekday tables retain the cleaned source clock, documented as UTC. The House 3 model illustration explicitly converts that cleaned UTC clock to Europe/London local time.

## Before and after cleaning

![Before and After Cleaning](reports/raw_house1/before_after.png)

The plot channels are Aggregate, Appliance1 (Fridge, as named in the supplied REFIT readme), and Appliance5 (Washing Machine). The selected day is **{raw['plot_day_utc']}**, chosen to contain a roughly 16 kW washing-machine minute spike and observed heating activity. Separate panels show its peak falling from **{raw['plot_before_wm_max_w']:,.1f} W** before cleaning to **{raw['plot_after_wm_max_w']:,.1f} W** after invalid readings are removed; the after panel uses its own readable scale. Values below the cutoff are retained, but this does not recover the true power during a rejected sensor glitch. The additional examples below show actual short-gap interpolation and an unfilled long outage on their own dates, rather than claiming both occurred on the spike day.

![Observed short and long gap examples](reports/raw_house1/gap_examples.png)

Blue dots are available minute means; green points/lines show the result after gap handling. Orange shading marks missing bins. The short supported gap receives an estimate; the long outage stays blank.

## Official cleaned data: Sections 2 and 3

EDA uses Houses 1, 2, 3, 7, 8 and 10 at one-minute resolution without added filling. Channels are Appliance5, Appliance2, Appliance6, Appliance5, Appliance4 and Appliance5 respectively. Training uses 1, 2, 7 and 10; validation uses 8; test uses 3.

The official release already forward-filled some readings and replaced appliance spikes above 4,000 W with zeros. Missing intervals, asynchronous polling and uncertain cycle boundaries remain limitations; EDA observation coverage is distinct from the model evaluation mask described in RESULTS.md.
"""


if __name__ == "__main__":
    main()
    raw_output = (
        Path(__import__("sys").argv[__import__("sys").argv.index("--output-dir") + 1])
        if "--output-dir" in __import__("sys").argv
        else Path("reports/raw_house1")
    )
    Path("DATA.md").write_text(
        raw_data_report(json.loads((raw_output / "audit.json").read_text())),
        encoding="utf-8",
    )
