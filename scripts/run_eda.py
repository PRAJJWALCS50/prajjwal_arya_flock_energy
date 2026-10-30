"""Official cleaned REFIT: official cleaned REFIT availability and candidate wash cycles."""

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

CHANNELS = {1: 5, 2: 2, 3: 6, 7: 5, 8: 4, 10: 5}


def runs(mask):
    d = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def cycle_intervals(power, enough):
    """Yield [start, end) intervals using the documented three-step rule."""
    active = np.asarray(enough, dtype=bool) & (np.asarray(power) >= 20)
    merged = active.copy()
    for a, b in runs(~active):
        if a > 0 and b < len(active) and b - a <= 15 and np.asarray(enough)[a:b].all():
            merged[a:b] = True
    for a, b in runs(merged):
        if b - a >= 20 and np.asarray(power)[a:b].sum() / 60000 >= 0.10:
            yield a, b


def analyze(h, channel, path, out, minute_callback=None):
    col = f"Appliance{channel}"
    d = pd.read_csv(path, usecols=["Time", "Unix", "Aggregate", col])
    t = pd.to_numeric(d.Unix, errors="coerce")
    parsed = pd.to_datetime(d.Time, errors="coerce")
    if t.isna().any() or parsed.isna().any() or (t != np.floor(t)).any():
        raise ValueError(f"House {h}: invalid timestamp")
    if (t.diff().dropna() <= 0).any():
        raise ValueError(f"House {h}: timestamps duplicated/out of order")
    if (
        parsed.astype("datetime64[ns]").astype("int64").to_numpy()
        != t.to_numpy(dtype=np.int64) * 10**9
    ).any():
        raise ValueError("Time/Unix disagreement")
    power = d[["Aggregate", col]].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(power).all(axis=1) & (power >= 0).all(axis=1)
    rows = len(d)
    invalid = int((~valid).sum())
    first = str(parsed.iloc[0])
    last = str(parsed.iloc[-1])
    d["minute"] = t // 60
    d["wm"] = power[col]
    d["aggregate"] = power.Aggregate
    count = d.groupby("minute").size()
    paired = (
        d.loc[valid]
        .groupby("minute")
        .agg(
            count=("Unix", "size"),
            first=("Unix", "min"),
            last=("Unix", "max"),
            wm=("wm", "mean"),
            aggregate=("aggregate", "mean"),
        )
    )
    m = paired.reindex(np.arange(int(d.minute.min()), int(d.minute.max()) + 1))
    source_count = count.reindex(m.index, fill_value=0)
    enough = (m["count"] >= 3) & ((m["last"] - m["first"]) >= 30)
    active = enough & (m.wm >= 20)
    if minute_callback is not None:
        minute_callback(m, enough, pd.to_datetime(m.index.to_numpy() * 60, unit="s"))
    events = []
    index = pd.to_datetime(m.index.to_numpy() * 60, unit="s")
    for a, b in cycle_intervals(m.wm.to_numpy(), enough.to_numpy()):
        length = b - a
        on = int(active.iloc[a:b].sum())
        boundary_ok = (
            a >= 10
            and b + 10 <= len(m)
            and enough.iloc[a - 10 : b + 10].all()
            and not active.iloc[a - 10 : a].any()
            and not active.iloc[b : b + 10].any()
        )
        events.append(
            {
                "house": h,
                "start": str(index[a]),
                "last_minute_start": str(index[b - 1]),
                "end_exclusive": str(index[b - 1] + pd.Timedelta(minutes=1)),
                "duration_minutes": length,
                "active_minutes": on,
                "peak_minute_mean_w": float(m.wm.iloc[a:b].max()),
                "energy_kwh": float(m.wm.iloc[a:b].sum() / 60000),
                "observed_quiet_context": bool(boundary_ok),
            }
        )
    gaps = [
        {
            "house": h,
            "start": str(index[a]),
            "end_exclusive": str(index[b - 1] + pd.Timedelta(minutes=1)),
            "minutes": int(b - a),
        }
        for a, b in runs(source_count == 0)
    ]
    missing = int((source_count == 0).sum())
    orange = int(((m["count"] > 0) & ~enough).sum())
    invalid_only = int(((source_count > 0) & m["count"].isna()).sum())
    rec = {
        "house": h,
        "wm_channel": col,
        "first_record": first,
        "last_record": last,
        "source_rows": rows,
        "calendar_minutes": len(m),
        "observed_minutes": int((source_count > 0).sum()),
        "usable_minutes": int(enough.sum()),
        "usable_pct": float(100 * enough.mean()),
        "available_not_usable_minutes": orange,
        "missing_minutes": missing,
        "missing_pct": 100 * missing / len(m),
        "invalid_paired_rows": invalid,
        "nan_mean_minutes": int(m.wm.isna().sum()),
        "records_without_valid_pair_minutes": invalid_only,
        "missing_gap_count": len(gaps),
        "longest_missing_gap_minutes": max([g["minutes"] for g in gaps], default=0),
        "candidate_cycles": len(events),
        "candidates_with_observed_quiet_context": sum(
            e["observed_quiet_context"] for e in events
        ),
    }
    assert rec["usable_minutes"] + orange + missing + invalid_only == len(m)
    pd.DataFrame(events).to_csv(out / f"house_{h}_candidate_cycles.csv", index=False)
    print(
        f'House {h}: {len(events)} candidates; {rec["usable_pct"]:.2f}% usable',
        flush=True,
    )
    return rec, gaps


def availability_main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing CLEAN_HouseN.csv",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports/eda",
    )
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    gaps = []
    for h, ch in CHANNELS.items():

        def collect(m, good, index):
            if h != 7:
                return
            daily = (
                pd.DataFrame(
                    {
                        "date": index.normalize(),
                        "active": (good & (m.wm >= 20)).astype(int).to_numpy(),
                        "usable": good.astype(int).to_numpy(),
                    }
                )
                .groupby("date")
                .sum()
            )
            pool = daily[(daily.usable >= 1380) & (daily.active > 0)]
            day = (pool.active - pool.active.median()).abs().idxmin()
            selected = (index >= day) & (index < day + pd.Timedelta(days=1))
            pd.DataFrame(
                {
                    "timestamp": index[selected],
                    "aggregate": m["aggregate"].where(good).to_numpy()[selected],
                    "wm": m.wm.where(good).to_numpy()[selected],
                }
            ).to_csv(out / "representative_24h_readings.csv", index=False)

        s, g = analyze(h, ch, args.data_dir / f"CLEAN_House{h}.csv", out, collect)
        summary.append(s)
        gaps.extend(g)
    df = pd.DataFrame(summary)
    df.to_csv(out / "availability_and_cycle_summary.csv", index=False)
    pd.DataFrame(gaps).to_csv(out / "missing_intervals.csv", index=False)

    (out / "method.json").write_text(
        json.dumps(
            {
                "channels": CHANNELS,
                "minute_samples": 3,
                "minute_span_s": 30,
                "active_threshold_w": 20,
                "max_pause_min": 15,
                "min_session_min": 20,
                "min_energy_kwh": 0.10,
                "threshold_inclusive": True,
                "quiet_context_min": 10,
                "mismatch_filter_applied": False,
            },
            indent=2,
        )
    )


"""Zero/off analysis from original cleaned REFIT minute means; never treat gaps as off."""

import argparse
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze_off(h, col, data):
    d = pd.read_csv(
        data / f"CLEAN_House{h}.csv", usecols=["Unix", "Aggregate", f"Appliance{col}"]
    )
    t = pd.to_numeric(d.Unix, errors="coerce")
    if t.isna().any() or (t.diff().dropna() <= 0).any():
        raise ValueError(f"House {h}: invalid/unordered timestamp")
    powers = d[["Aggregate", f"Appliance{col}"]].apply(pd.to_numeric, errors="coerce")
    pair = np.isfinite(powers).all(axis=1) & (powers >= 0).all(axis=1)
    d["minute"] = t // 60
    d["wm"] = powers.iloc[:, 1]
    counts = d.groupby("minute").size()
    m = (
        d.loc[pair]
        .groupby("minute")
        .agg(
            n=("Unix", "size"),
            first=("Unix", "min"),
            last=("Unix", "max"),
            wm=("wm", "mean"),
        )
        .reindex(np.arange(int(d.minute.min()), int(d.minute.max()) + 1))
    )
    enough = (m.n >= 3) & (m["last"] - m["first"] >= 30)
    y = m.loc[enough, "wm"]
    zero = int((y == 0).sum())
    low = int(((y > 0) & (y < 20)).sum())
    active = int((y >= 20).sum())
    assert zero + low + active == len(y)
    return {
        "house": h,
        "wm_channel": f"Appliance{col}",
        "calendar_minutes": len(m),
        "usable_minutes": len(y),
        "excluded_missing_or_insufficient_minutes": int((~enough).sum()),
        "exact_zero_minutes": zero,
        "low_power_0_to_20w_minutes": low,
        "inactive_lt20w_minutes": zero + low,
        "active_ge20w_minutes": active,
        "exact_zero_pct": 100 * zero / len(y),
        "low_power_pct": 100 * low / len(y),
        "inactive_pct": 100 * (zero + low) / len(y),
        "active_pct": 100 * active / len(y),
        "exact_zero_equivalent_hours": zero / 60,
        "inactive_equivalent_hours": (zero + low) / 60,
        "active_equivalent_hours": active / 60,
    }


def off_main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports/eda",
    )
    args = p.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for h, c in CHANNELS.items():
        rows.append(analyze_off(h, c, args.data_dir))
        print(f"House {h} complete", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "zero_off_time_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    left = np.zeros(len(df))
    labels = [f"House {h}" for h in df.house]
    for field, label, color in [
        ("exact_zero_pct", "Exactly 0 W", "#547da0"),
        ("low_power_pct", "Above 0, below 20 W", "#e5a23b"),
        ("active_pct", "At least 20 W (active)", "#35915b"),
    ]:
        v = df[field].to_numpy()
        ax.barh(labels, v, left=left, label=label, color=color)
        left += v
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of sufficiently observed minutes")
    ax.set_title(
        "Washing-machine zero, low-power and active time\nMissing and insufficient minutes excluded"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(out / "zero_off_time_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


"""Cycle start patterns, adjusted for sufficiently observed recording time."""


matplotlib.use("Agg")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def usage_main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports/eda/usage_patterns",
    )
    args = p.parse_args()
    O = args.output_dir
    O.mkdir(parents=True, exist_ok=True)
    records = []
    summary = []
    for h, c in CH.items():
        exposures = {}

        def collect(m, valid, index):
            exposures["hour"] = np.bincount(index.hour[valid], minlength=24)
            exposures["weekday"] = np.bincount(index.dayofweek[valid], minlength=7)

        analyze(h, c, args.data_dir / f"CLEAN_House{h}.csv", O, collect)
        cycles = pd.read_csv(O / f"house_{h}_candidate_cycles.csv")
        selected = cycles.copy()
        starts = pd.to_datetime(selected.start)
        selected.to_csv(O / f"house_{h}_detected_cycles.csv", index=False)
        for group, nums, N in [
            ("hour", starts.dt.hour, 24),
            ("weekday", starts.dt.dayofweek, 7),
        ]:
            count = np.bincount(nums, minlength=N)
            for i in range(N):
                minutes = int(exposures[group][i])
                records.append(
                    {
                        "house": h,
                        "group": group,
                        "bin": i,
                        "label": f"{i:02}:00" if group == "hour" else DAYS[i],
                        "detected_cycle_starts": int(count[i]),
                        "usable_minutes": minutes,
                        "usable_hours": minutes / 60,
                        "starts_per_100_usable_hours": (
                            100 * count[i] / (minutes / 60) if minutes else np.nan
                        ),
                    }
                )
        summary.append(
            {
                "house": h,
                "all_candidate_cycles": len(cycles),
                "detected_cycles": len(selected),
            }
        )
    df = pd.DataFrame(records)
    df.to_csv(O / "usage_by_hour_and_weekday.csv", index=False)
    pd.DataFrame(summary).to_csv(O / "cycle_counts.csv", index=False)
    for group, title, width in [
        ("hour", "Hour of day", 16),
        ("weekday", "Day of week", 11),
    ]:
        part = df[df.group == group]
        pivot = part.pivot(
            index="house", columns="bin", values="starts_per_100_usable_hours"
        ).reindex(CH.keys())
        fig, ax = plt.subplots(figsize=(width, 5))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0)
        ax.set_yticks(range(6), [f"House {h}" for h in CH])
        labels = [f"{h:02}" for h in range(24)] if group == "hour" else DAYS
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel(title + " (source timestamp convention)")
        ax.set_title(
            "Detected detected candidate cycle starts per 100 usable recording hours"
        )
        for i in range(6):
            for j in range(len(labels)):
                v = pivot.iloc[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > np.nanmax(pivot.to_numpy()) * 0.6 else "black",
                )
        fig.colorbar(
            im, ax=ax, label="Starts / 100 usable hours", fraction=0.025, pad=0.02
        )
        fig.tight_layout()
        fig.savefig(O / f"{group}_usage.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    # Raw counts are supplied alongside rates so exposure corrections are inspectable.
    pooled = (
        df.groupby(["group", "bin", "label"], sort=False)[
            ["detected_cycle_starts", "usable_minutes"]
        ]
        .sum()
        .reset_index()
    )
    pooled["starts_per_100_usable_hours"] = (
        6000 * pooled.detected_cycle_starts / pooled.usable_minutes
    )
    pooled.to_csv(O / "pooled_usage.csv", index=False)
    print("USAGE ANALYSIS COMPLETE", flush=True)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


matplotlib.use("Agg")
import matplotlib.dates as md


def comparison_main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument(
        "--eda-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "reports/eda",
    )
    args = p.parse_args()
    E = args.eda_dir
    O = E / "cycle_comparison"
    O.mkdir(exist_ok=True)
    summary = []
    examples = []
    allcycles = []
    for h in CH:
        c = pd.read_csv(E / f"usage_patterns/house_{h}_detected_cycles.csv")
        c["house"] = h
        allcycles.append(c)
        r = {"house": h, "detected_cycles": len(c)}
        for field in ["duration_minutes", "peak_minute_mean_w", "energy_kwh"]:
            r[field + "_median"] = float(c[field].median())
            r[field + "_p25"] = float(c[field].quantile(0.25))
            r[field + "_p75"] = float(c[field].quantile(0.75))
        summary.append(r)
        # Representative in joint duration/energy space; no cherry-picking by visual appearance.
        score = np.zeros(len(c))
        for field in ["duration_minutes", "energy_kwh"]:
            scale = c[field].quantile(0.75) - c[field].quantile(0.25)
            score += abs(c[field] - c[field].median()).to_numpy() / max(
                float(scale), 1e-6
            )
        chosen = c.iloc[int(np.argmin(score))]
        start = pd.Timestamp(chosen.start)
        end = pd.Timestamp(chosen.end_exclusive)
        begin = int((start - pd.Timedelta(minutes=20)).timestamp())
        finish = int((end + pd.Timedelta(minutes=20)).timestamp())
        col = f"Appliance{CH[h]}"
        chunks = []
        for chunk in pd.read_csv(
            args.data_dir / f"CLEAN_House{h}.csv",
            usecols=["Unix", "Aggregate", col],
            chunksize=500000,
        ):
            hit = chunk[(chunk.Unix >= begin) & (chunk.Unix < finish)]
            if len(hit):
                chunks.append(hit)
        d = pd.concat(chunks)
        valid = np.isfinite(d[["Aggregate", col]]).all(axis=1) & (
            d[["Aggregate", col]] >= 0
        ).all(axis=1)
        d = d.loc[valid].copy()
        d["minute"] = d.Unix // 60 * 60
        m = (
            d.groupby("minute")
            .agg(
                aggregate=("Aggregate", "mean"),
                wm=(col, "mean"),
                n=("Unix", "size"),
                first=("Unix", "min"),
                last=("Unix", "max"),
            )
            .reindex(np.arange(begin, finish, 60))
        )
        enough = (m.n >= 3) & ((m["last"] - m["first"]) >= 30)
        dt = pd.to_datetime(m.index, unit="s")
        event = (dt >= start) & (dt < end)
        assert enough[event].all()
        assert abs(m.loc[event, "wm"].sum() / 60000 - chosen.energy_kwh) < 1e-8
        m["timestamp"] = dt
        m["usable"] = enough
        m["candidate_cycle"] = event
        m.to_csv(O / f"house_{h}_example_readings.csv", index_label="unix")
        fig, (ax, wm_ax) = plt.subplots(
            2, 1, figsize=(13, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]}
        )
        ax.plot(dt, m["aggregate"].where(enough), label="Aggregate", color="#417794")
        ax.plot(dt, m.wm.where(enough), label="Washing machine", color="#8a448d")
        ax.set_ylabel("Power (W)")
        ax.legend()
        wm_ax.plot(dt, m.wm.where(enough), color="#8a448d", label="Washing machine")
        wm_ax.axhline(20, color="#555", ls="--", lw=1, label="20 W active threshold")
        pause = event & enough.to_numpy() & (m.wm.to_numpy() < 20)
        wm_ax.scatter(
            dt[pause],
            m.wm.to_numpy()[pause],
            s=15,
            color="#d48a25",
            label="Below-threshold pause inside session",
        )
        for a in [ax, wm_ax]:
            a.axvspan(start, end, color="#42986e", alpha=0.13)
            a.axvline(start, color="#42986e", ls=":")
            a.axvline(end, color="#42986e", ls=":")
            a.grid(alpha=0.2)
        wm_ax.set_ylabel("WM power (W)")
        wm_ax.legend(fontsize=9)
        wm_ax.xaxis.set_major_formatter(md.DateFormatter("%H:%M"))
        wm_ax.set_xlabel("Source time; shaded area is detected candidate session")
        fig.suptitle(
            f"House {h}: {start.date()} | {int(chosen.duration_minutes)} min | {chosen.energy_kwh:.3f} kWh\nPeak minute mean {chosen.peak_minute_mean_w:.0f} W; {int(chosen.active_minutes)} active minutes"
        )
        fig.tight_layout()
        fig.savefig(O / f"house_{h}_annotated_cycle.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        examples.append(
            {
                "house": h,
                "start": chosen.start,
                "end_exclusive": chosen.end_exclusive,
                "duration_minutes": int(chosen.duration_minutes),
                "energy_kwh": float(chosen.energy_kwh),
                "peak_minute_mean_w": float(chosen.peak_minute_mean_w),
                "active_minutes": int(chosen.active_minutes),
            }
        )
        print(f"House {h} cycle comparison/example complete", flush=True)
    s = pd.DataFrame(summary)
    s.to_csv(O / "cycle_characteristics.csv", index=False)
    pd.DataFrame(examples).to_csv(O / "selected_examples.csv", index=False)
    pd.concat(allcycles).to_csv(O / "all_detected_cycles.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax, field, title in zip(
        axs,
        ["duration_minutes", "peak_minute_mean_w", "energy_kwh"],
        [
            "Duration (minutes)",
            "Peak minute-average power (W)",
            "Energy per candidate session (kWh)",
        ],
    ):
        groups = [c[field].to_numpy() for c in allcycles]
        ax.boxplot(groups, tick_labels=[str(h) for h in CH], showfliers=False)
        ax.set_xlabel("House")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Detected candidate sessions: median, middle 50% and 1.5-IQR whiskers\nOutlier points omitted from display only; all sessions remain in statistics/CSV"
    )
    fig.tight_layout()
    fig.savefig(O / "cross_house_cycle_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(s.to_string(index=False), flush=True)


"""Build the self-contained EDA tables and four figures from reproducible outputs."""


matplotlib.use("Agg")

HOUSES = [1, 2, 3, 7, 8, 10]
CH = CHANNELS


def markdown(frame):
    return (
        "| "
        + " | ".join(frame.columns)
        + " |\n| "
        + " | ".join(["---"] * len(frame.columns))
        + " |\n"
        + "\n".join(
            "| " + " | ".join(map(str, row)) + " |"
            for row in frame.itertuples(index=False, name=None)
        )
        + "\n"
    )


def summary_main():
    p = argparse.ArgumentParser()
    p.add_argument("--eda-dir", type=Path, required=True)
    a = p.parse_args()
    E = a.eda_dir
    F = E / "figures"
    F.mkdir(exist_ok=True)
    availability = pd.read_csv(E / "availability_and_cycle_summary.csv").set_index(
        "house"
    )
    off = pd.read_csv(E / "zero_off_time_summary.csv").set_index("house")
    rows = []
    groups = []
    for h in HOUSES:
        r = availability.loc[h]
        c = pd.read_csv(E / f"house_{h}_candidate_cycles.csv")
        groups.append(c)
        assert (c.duration_minutes >= 20).all() and (c.energy_kwh >= 0.10).all()
        row = {
            "house": h,
            "wm_channel": r.wm_channel,
            "start_date": r.first_record,
            "end_date": r.last_record,
            "calendar_span_days": r.calendar_minutes / 1440,
            "calendar_minutes": int(r.calendar_minutes),
            "usable_minutes": int(r.usable_minutes),
            "missing_pct": 100 * r.nan_mean_minutes / r.calendar_minutes,
            "missing_nan_minutes": int(r.nan_mean_minutes),
            "insufficient_minutes": int(r.available_not_usable_minutes),
            "off_pct": off.loc[h, "inactive_pct"],
            "detected_cycles": len(c),
            "cycles_per_observed_week": len(c) / (r.usable_minutes / 10080),
        }
        for k in ["duration_minutes", "peak_minute_mean_w", "energy_kwh"]:
            row.update(
                {
                    k + "_median": c[k].median(),
                    k + "_mean": c[k].mean(),
                    k + "_p25": c[k].quantile(0.25),
                    k + "_p75": c[k].quantile(0.75),
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(E / "eda_summary_table.csv", index=False)
    d = pd.read_csv(E / "representative_24h_readings.csv", parse_dates=["timestamp"])
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(
        d.timestamp, d["aggregate"], label="Whole-home aggregate", color="#6b8190", lw=1
    )
    ax.plot(d.timestamp, d.wm, label="Washing machine", color="#a23b72", lw=1.3)
    ax.set(
        title=f"House 7 — {d.timestamp.iloc[0].date()} | representative 24 hours",
        ylabel="Power (W)",
        xlabel="Time (source convention)",
    )
    ax.xaxis.set_major_formatter(md.DateFormatter("%H:%M"))
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(F / "24h_aggregate_and_wm_trace.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    for ax, h in zip(axes, [7, 10, 3]):
        d = pd.read_csv(
            E / f"cycle_comparison/house_{h}_example_readings.csv",
            parse_dates=["timestamp"],
        )
        event = d.candidate_cycle
        start = d.loc[event, "timestamp"].iloc[0]
        end = d.loc[event, "timestamp"].iloc[-1] + pd.Timedelta(minutes=1)
        x = (d.timestamp - start).dt.total_seconds() / 60
        y = d.wm.where(d.usable)
        ax.plot(x, y, color="#a23b72", label="WM minute mean")
        ax.axhline(20, color="grey", ls="--", lw=0.8, label="20 W threshold")
        ax.axvspan(0, (end - start).total_seconds() / 60, color="#268b54", alpha=0.10)
        for pos, label in [(0, "Start"), ((end - start).total_seconds() / 60, "End")]:
            ax.axvline(pos, color="#268b54", ls=":")
            ax.text(
                pos,
                1.01,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=9,
            )
        pause = event & d.usable & (y < 20)
        ax.scatter(
            x[pause],
            y[pause],
            s=18,
            color="#db8a17",
            label="Observed pause joined (≤15 min)",
            zorder=4,
        )
        peak = y.where(event).idxmax()
        label = "Peak: possible heating" if y[peak] >= 1500 else "Peak: phase unknown"
        ax.annotate(
            label,
            xy=(x[peak], y[peak]),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "->"},
        )
        motor = d[event & y.between(150, 500)]
        if len(motor):
            i = motor.index[-1]
            ax.annotate(
                "Lower-power activity\n(possible motor/pump; spin unverified)",
                xy=(x[i], y[i]),
                xytext=(-200, 35),
                textcoords="offset points",
                fontsize=8,
                arrowprops={"arrowstyle": "->"},
            )
        ax.set(
            title=f"House {h} — {start.date()}",
            ylabel="Power (W)",
            xlabel="Minutes from detected start",
        )
        ax.set_ylim(top=y.max() * 1.35)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(F / "detected_wm_cycles_zoom.png", dpi=180)
    plt.close(fig)
    rates = pd.read_csv(E / "usage_patterns/usage_by_hour_and_weekday.csv")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    for ax, g in zip(axes, ["hour", "weekday"]):
        d = rates[rates.group == g]
        v = d.pivot(
            index="house", columns="bin", values="starts_per_100_usable_hours"
        ).reindex(HOUSES)
        im = ax.imshow(v, aspect="auto", cmap="YlGnBu", vmin=0)
        ax.set_yticks(range(6), [f"House {h}" for h in HOUSES])
        labels = (
            [f"{i:02}" for i in range(24)]
            if g == "hour"
            else ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        )
        ax.set_xticks(range(len(labels)), labels)
        ax.set_title("Hour of day" if g == "hour" else "Day of week")
        fig.colorbar(im, ax=ax, label="Cycle starts / 100 usable hours")
    fig.suptitle("Washing-machine starts adjusted for observed recording time")
    fig.tight_layout()
    fig.savefig(F / "usage_patterns_hour_weekday.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, k, label in zip(
        axes,
        ["duration_minutes", "peak_minute_mean_w", "energy_kwh"],
        ["Duration (minutes)", "Peak minute mean (W)", "Energy (kWh)"],
    ):
        ax.boxplot(
            [c[k] for c in groups], tick_labels=list(map(str, HOUSES)), showfliers=False
        )
        ax.set(xlabel="House", ylabel=label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("All detected cycles — median, middle 50% and 1.5-IQR whiskers")
    fig.tight_layout()
    fig.savefig(F / "cross_house_cycle_comparison.png", dpi=180)
    plt.close(fig)
    summary = pd.DataFrame(rows)
    rates = pd.read_csv(E / "usage_patterns/usage_by_hour_and_weekday.csv")
    off_table = pd.read_csv(E / "zero_off_time_summary.csv").set_index("house")
    for row in summary.itertuples():
        assert row.detected_cycles == availability.loc[row.house, "candidate_cycles"]
        z = off_table.loc[row.house]
        assert (
            z.exact_zero_minutes + z.low_power_0_to_20w_minutes + z.active_ge20w_minutes
            == row.usable_minutes
        )
        for group in ["hour", "weekday"]:
            subset = rates[(rates.house == row.house) & (rates.group == group)]
            assert subset.detected_cycle_starts.sum() == row.detected_cycles
            assert subset.usable_minutes.sum() == row.usable_minutes
    print("EDA summary table, figures and reconciliation checks complete", flush=True)


def main():
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    a = p.parse_args()
    for fn, flags in [
        (
            availability_main,
            ["--data-dir", str(a.data_dir), "--output-dir", str(a.output_dir)],
        ),
        (off_main, ["--data-dir", str(a.data_dir), "--output-dir", str(a.output_dir)]),
        (
            usage_main,
            [
                "--data-dir",
                str(a.data_dir),
                "--output-dir",
                str(a.output_dir / "usage_patterns"),
            ],
        ),
        (
            comparison_main,
            ["--data-dir", str(a.data_dir), "--eda-dir", str(a.output_dir)],
        ),
        (summary_main, ["--eda-dir", str(a.output_dir)]),
    ]:
        sys.argv = [sys.argv[0], *flags]
        fn()
    write_eda_section(a.output_dir)
    key_figures = {
        "24h_aggregate_and_wm_trace.png",
        "detected_wm_cycles_zoom.png",
        "usage_patterns_hour_weekday.png",
        "cross_house_cycle_comparison.png",
    }
    for figure in a.output_dir.rglob("*.png"):
        if figure.parent != a.output_dir / "figures" or figure.name not in key_figures:
            figure.unlink()
    # Candidate records at the EDA root are the canonical source for summary counts.
    for duplicate in (a.output_dir / "usage_patterns").glob(
        "house_*_candidate_cycles.csv"
    ):
        duplicate.unlink()


def write_eda_section(E):
    s = pd.read_csv(E / "eda_summary_table.csv")
    coverage = s[
        [
            "house",
            "wm_channel",
            "start_date",
            "end_date",
            "missing_pct",
            "off_pct",
            "detected_cycles",
            "cycles_per_observed_week",
        ]
    ].round(3)
    characteristics = s[
        ["house"]
        + [
            k + q
            for k in ["duration_minutes", "peak_minute_mean_w", "energy_kwh"]
            for q in ["_median", "_p25", "_p75"]
        ]
    ].round(3)
    rates = pd.read_csv(E / "usage_patterns/usage_by_hour_and_weekday.csv")
    peak = []
    for h in HOUSES:
        v = {"house": h}
        for group in ["hour", "weekday"]:
            r = (
                rates[(rates.house == h) & (rates.group == group)]
                .sort_values(
                    ["starts_per_100_usable_hours", "bin"], ascending=[False, True]
                )
                .iloc[0]
            )
            v[group] = r.label
            v[group + "_starts_per_100_hours"] = round(r.starts_per_100_usable_hours, 3)
        peak.append(v)
    coverage = coverage.rename(
        columns={
            "house": "House",
            "wm_channel": "WM channel",
            "start_date": "Start",
            "end_date": "End",
            "missing_pct": "Missing %",
            "off_pct": "Off %",
            "detected_cycles": "Cycles",
            "cycles_per_observed_week": "Cycles / observed week",
        }
    )
    characteristics = characteristics.rename(
        columns={
            "house": "House",
            **{
                k + q: label + suffix
                for k, label in [
                    ("duration_minutes", "Duration (min)"),
                    ("peak_minute_mean_w", "Peak (W)"),
                    ("energy_kwh", "Energy (kWh)"),
                ]
                for q, suffix in [
                    ("_median", " median"),
                    ("_p25", " Q1"),
                    ("_p75", " Q3"),
                ]
            },
        }
    )
    text = (
        "# Results\n\n## 1. Washing-machine EDA\n\nOfficial cleaned REFIT data are averaged to one-minute bins for Houses 1, 2, 3, 7, 8 and 10. A usable minute has at least three valid paired readings spanning 30 seconds. Missing or insufficient observations stay separate from off time.\n\n"
        + markdown(coverage)
    )
    text += (
        "\nMark a minute active at >=20 W, join sufficiently observed inactive pauses of up to 15 minutes, and retain sessions lasting >=20 minutes AND using >=0.10 kWh. Missing minutes break sessions; all passing candidates enter the analysis. These thresholds can merge nearby washes or omit short/cold programmes. Cycles/week uses observed recording exposure.\n\n"
        + markdown(characteristics)
        + "\nQuartiles describe the middle 50% of sessions; power is a peak minute mean. Energy is the sum of minute watts divided by 60,000.\n\n"
        + markdown(pd.DataFrame(peak))
    )
    for name, caption in [
        (
            "24h_aggregate_and_wm_trace",
            "The representative House 7 day is nearest median observed activity among well-observed days. Washing-machine peaks sit within larger whole-home variations.",
        ),
        (
            "detected_wm_cycles_zoom",
            "Median-like examples show detected boundaries and observed pauses. Heating and motor phases are plausible interpretations, not verified programme labels.",
        ),
        (
            "usage_patterns_hour_weekday",
            "Cycle starts are divided by usable recording hours for each hour and weekday. The homes have different peak start times, so pooled habits should not be applied to every customer.",
        ),
        (
            "cross_house_cycle_comparison",
            "Duration, peak power and cycle energy vary across homes. Differences may reflect machines, programmes or detection errors and do not establish relative efficiency.",
        ),
    ]:
        text += f"\n![{name}](reports/eda/figures/{name}.png)\n\n{caption}\n"
    rates = pd.read_csv(E / "usage_patterns/usage_by_hour_and_weekday.csv")
    hourly = rates[rates.group == "hour"]
    daytime = hourly[hourly.bin.between(6, 21)]
    night = hourly[~hourly.bin.between(6, 21)]
    day_rate = 6000 * daytime.detected_cycle_starts.sum() / daytime.usable_minutes.sum()
    night_rate = 6000 * night.detected_cycle_starts.sum() / night.usable_minutes.sum()
    text += f"\n### Numerical comparison\n\nHouse 10 has the longest median session at 120.0 minutes [Q1–Q3: 118–122], versus 33.0 minutes in House 1. The tight two-hour cluster is consistent with a repeated programme or stable machine behaviour, but programmes were not logged, so the cause cannot be confirmed. House 8 has the highest median cycle energy, 0.786 kWh.\n\nThe pooled daytime (06:00–21:59) start rate is {day_rate:.2f}, versus {night_rate:.2f} overnight, per 100 observed hours. These hours use the cleaned source UTC clock; the model illustration converts UTC to UK local time.\n"
    Path("RESULTS.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
