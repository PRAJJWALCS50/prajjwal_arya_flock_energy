"""A simple two-stage washing-machine model, split comparisons and six-house LOHO."""

import os

os.environ["OMP_NUM_THREADS"] = "8"
from pathlib import Path
import argparse, json, itertools
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = argparse.ArgumentParser()
P.add_argument("--data-dir", type=Path, required=True)
P.add_argument("--output-dir", type=Path, default=Path("reports/model"))
P.add_argument("--reuse-model", action="store_true")
args = P.parse_args()
O = args.output_dir
O.mkdir(parents=True, exist_ok=True)
CHANNELS = {1: 5, 2: 2, 3: 6, 7: 5, 8: 4, 10: 5}
FOLDS = {1: 2, 2: 1, 3: 8, 7: 10, 8: 7, 10: 3}


def prepare(h):
    cache = O / "cache" / f"simple_v1_house{h}.joblib"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        return joblib.load(cache)
    col = f"Appliance{CHANNELS[h]}"
    d = pd.read_csv(
        args.data_dir / f"CLEAN_House{h}.csv", usecols=["Unix", "Aggregate", col]
    )
    assert (d.Unix.diff().dropna() > 0).all()
    d.index = pd.to_datetime(d.Unix, unit="s")
    m = d[["Aggregate", col]].replace([np.inf, -np.inf], np.nan).resample("1min").mean()
    a = m.Aggregate
    y = m[col]
    frames = []
    segments = a.isna().cumsum()
    for _, s in a[a.notna()].groupby(segments[a.notna()]):
        base = s.rolling(60, min_periods=1).min()
        net = (s - base).clip(lower=0)
        f = pd.DataFrame({"aggregate": s, "base_load": base, "net_aggregate": net})
        for w in [15, 30, 60]:
            r = net.rolling(w, min_periods=1)
            for stat in ["mean", "max", "std"]:
                f[f"{stat}_{w}"] = getattr(r, stat)()
        frames.append(f)
    f = pd.concat(frames).reindex(m.index).astype("float32")
    good = a.notna() & y.notna() & y.between(0, 3500)
    result = (f.loc[good], y.loc[good].astype("float32"), a.loc[good])
    joblib.dump(result, cache)
    print(f"Prepared House {h}: {good.sum():,} observed minutes", flush=True)
    return result


def predict_watts(p, watts, index, tau):
    on = p >= tau
    breaks = np.r_[True, np.diff(index.as_unit("ns").asi8) != 60 * 10**9]
    starts = np.flatnonzero(on & (breaks | ~np.r_[False, on[:-1]]))
    ends = np.flatnonzero(on & (np.r_[breaks[1:], True] | ~np.r_[on[1:], False]))
    for s, e in zip(starts, ends):
        if e - s + 1 < 10:
            on[s : e + 1] = False
    return np.where(on, np.clip(p * watts, 0, 3000), 0)


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


def metric(y, v):
    active = y >= 20
    pr, re, f, _ = precision_recall_fscore_support(
        active, v >= 20, average="binary", zero_division=0
    )
    true = float(np.asarray(y, dtype=float).sum() / 60000)
    pred = float(v.sum() / 60000)
    return dict(
        MAE_W=mean_absolute_error(y, v),
        RMSE_W=np.sqrt(mean_squared_error(y, v)),
        active_MAE_W=mean_absolute_error(y[active], v[active]),
        inactive_MAE_W=mean_absolute_error(y[~active], v[~active]),
        precision_pct=100 * pr,
        recall_pct=100 * re,
        F1_pct=100 * f,
        true_energy_kwh=true,
        predicted_energy_kwh=pred,
        SAE_pct=100 * abs(pred - true) / true,
    )


def cycle_metrics(y, pred, strict=False):
    frame = pd.DataFrame({"true": y, "pred": pred}).resample("1min").asfreq()
    good = frame.notna().all(axis=1).to_numpy()
    truth = list(cycle_intervals(frame.true.to_numpy(), good))
    guess = list(cycle_intervals(frame.pred.to_numpy(), good))
    pairs = []
    for i, (a, b) in enumerate(truth):
        for j, (c, d) in enumerate(guess):
            if (
                (abs(a - c) <= 10)
                if strict
                else (max(a, c) < min(b, d) or abs(a - c) <= 30)
            ):
                pairs.append((abs(a - c), i, j))
    used_t = set()
    used_p = set()
    errors = []
    for _, i, j in sorted(pairs):
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        a, b = truth[i]
        c, d = guess[j]
        energy = frame.true.iloc[a:b].sum() / 60000
        pe = frame.pred.iloc[c:d].sum() / 60000
        errors.append(
            [
                abs(a - c),
                abs((b - a) - (d - c)),
                abs(pe - energy),
                100 * abs(pe - energy) / energy,
            ]
        )
    tp = len(errors)
    pr = tp / len(guess) if guess else 0
    re = tp / len(truth) if truth else 0
    med = np.median(errors, axis=0) if errors else [np.nan] * 4
    return dict(
        true_cycles=len(truth),
        predicted_cycles=len(guess),
        matched_cycles=tp,
        cycle_precision_pct=100 * pr,
        cycle_recall_pct=100 * re,
        cycle_F1_pct=200 * pr * re / (pr + re) if pr + re else 0,
        median_start_error_min=med[0],
        median_duration_error_min=med[1],
        median_energy_error_kwh=med[2],
        median_energy_error_pct=med[3],
    )


def fit_and_evaluate():
    all_rows = []
    all_cycles = []
    settings = []
    counts = []
    for test_house, val_house in FOLDS.items():
        train = [h for h in CHANNELS if h not in [test_house, val_house]]
        print(
            f"FOLD test={test_house}, validation={val_house}, train={train}", flush=True
        )
        file = O / f"two_stage_hgb_house{test_house}.joblib"
        if args.reuse_model and file.exists():
            saved = joblib.load(file)
            clf = saved["classifier"]
            reg = saved["regressor"]
            profile = saved["profile"]
            median = saved["median_active_power"]
        else:
            parts = []
            for h in train:
                x, y, a = prepare(h)
                keep = y <= a + 100
                parts.append((x.loc[keep], y.loc[keep]))
            X = pd.concat([v[0] for v in parts])
            y = pd.concat([v[1] for v in parts])
            median = float(y[y >= 20].median())
            profile = np.bincount(y.index.hour, weights=y, minlength=24) / np.bincount(
                y.index.hour, minlength=24
            )
            params = dict(
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=40,
                learning_rate=0.08,
                l2_regularization=1,
                random_state=42,
                early_stopping=False,
            )
            clf = HistGradientBoostingClassifier(class_weight="balanced", **params).fit(
                X, y >= 20
            )
            reg = HistGradientBoostingRegressor(loss="absolute_error", **params).fit(
                X.loc[y >= 20], y[y >= 20]
            )
            del X, y, parts
            joblib.dump(
                dict(
                    classifier=clf,
                    regressor=reg,
                    profile=profile,
                    median_active_power=median,
                ),
                file,
            )
        vx, vy, va = prepare(val_house)
        p = clf.predict_proba(vx)[:, 1]
        power = reg.predict(vx)
        search = []
        for tau in np.arange(0.05, 1, 0.05):
            pred = predict_watts(p, power, vx.index, tau)
            search.append(dict(threshold=round(float(tau), 2), **metric(vy, pred)))
        search = pd.DataFrame(search)
        chosen = search.sort_values(["F1_pct", "MAE_W"], ascending=[False, True]).iloc[
            0
        ]
        tau = float(chosen.threshold)
        search.to_csv(O / f"fold_{test_house}_validation_search.csv", index=False)
        settings.append(
            dict(
                test_house=test_house,
                validation_house=val_house,
                training_houses=",".join(map(str, train)),
                threshold=tau,
                median_active_power_W=median,
            )
        )
        print(settings[-1], flush=True)
        for split, h, pack in [
            ("validation", val_house, (vx, vy, va)),
            ("test", test_house, prepare(test_house)),
        ]:
            x, y, a = pack
            p = clf.predict_proba(x)[:, 1]
            pred = predict_watts(p, reg.predict(x), x.index, tau)
            values = {
                "always_off": np.zeros(len(y)),
                "usage_profile": profile[y.index.hour],
                "plain_classifier": predict_watts(
                    p, np.full(len(y), median), x.index, tau
                ),
                "two_stage_hgb": pred,
            }
            counts.append(
                dict(
                    fold=test_house,
                    split=split,
                    house=h,
                    first=str(y.index[0]),
                    last=str(y.index[-1]),
                    minutes=len(y),
                    active_minutes=int((y >= 20).sum()),
                )
            )
            for name, v in values.items():
                all_rows.append(
                    dict(
                        fold=test_house,
                        split=split,
                        house=h,
                        validation_house=val_house,
                        training_houses=",".join(map(str, train)),
                        predictor=name,
                        **metric(y, v),
                    )
                )
                if name in ["plain_classifier", "two_stage_hgb"]:
                    for strict in [False, True]:
                        all_cycles.append(
                            dict(
                                fold=test_house,
                                split=split,
                                house=h,
                                predictor=name,
                                matching=(
                                    "strict_start_10min"
                                    if strict
                                    else "overlap_or_start_30min"
                                ),
                                **cycle_metrics(y, v, strict),
                            )
                        )
            if split == "test" or test_house in [2, 3]:
                pd.DataFrame(
                    dict(
                        actual_w=y,
                        predicted_w=pred,
                        classifier_w=values["plain_classifier"],
                        aggregate_w=a,
                    )
                ).to_csv(
                    O / f"fold_{test_house}_{split}_predictions.csv.gz",
                    index_label="timestamp",
                )
        pd.DataFrame(all_rows).to_csv(O / "all_split_metrics.csv", index=False)
        pd.DataFrame(all_cycles).to_csv(O / "all_cycle_metrics.csv", index=False)
        del vx, vy, va, x, y, a
    rows = pd.DataFrame(all_rows)
    cycles = pd.DataFrame(all_cycles)
    loho = rows[rows.split == "test"].copy()
    c = cycles[
        (cycles.split == "test")
        & (cycles.predictor == "two_stage_hgb")
        & (cycles.matching == "overlap_or_start_30min")
    ]
    loho = loho.merge(
        c[
            [
                "house",
                "cycle_precision_pct",
                "cycle_recall_pct",
                "cycle_F1_pct",
                "median_start_error_min",
                "median_duration_error_min",
                "median_energy_error_kwh",
            ]
        ],
        on="house",
        how="left",
    )
    # Event columns are meaningful for the two-stage model; baseline event metrics are not substituted.
    loho.loc[
        loho.predictor != "two_stage_hgb",
        c.columns.difference(
            ["fold", "split", "house", "predictor", "matching"]
        ).intersection(loho.columns),
    ] = np.nan
    loho.to_csv(O / "loho_evaluation_metrics.csv", index=False)
    summary = []
    for name, g in loho.groupby("predictor"):
        for field in [
            "MAE_W",
            "RMSE_W",
            "active_MAE_W",
            "F1_pct",
            "SAE_pct",
            "cycle_F1_pct",
            "median_start_error_min",
        ]:
            summary.append(
                dict(
                    predictor=name,
                    metric=field,
                    mean=g[field].mean(),
                    SD=g[field].std(ddof=1),
                    median=g[field].median(),
                    Q1=g[field].quantile(0.25),
                    Q3=g[field].quantile(0.75),
                )
            )
    pd.DataFrame(summary).to_csv(O / "loho_summary_metrics.csv", index=False)
    rows[rows.fold.isin([2, 3])].to_csv(O / "evaluation_metrics.csv", index=False)
    cycles[cycles.fold.isin([2, 3])].to_csv(
        O / "cycle_evaluation_metrics.csv", index=False
    )
    pd.DataFrame(counts).to_csv(O / "data_counts.csv", index=False)
    (O / "metadata.json").write_text(
        json.dumps(
            dict(
                folds=settings,
                features=["aggregate", "base_load", "net_aggregate"]
                + [
                    f"{stat}_{w}"
                    for w in [15, 30, 60]
                    for stat in ["mean", "max", "std"]
                ],
                run_length_minutes=10,
            ),
            indent=2,
        )
    )
    make_reports()


def select_test_example(test, output):
    """Performance-ranked illustration; never used for fitting or full-test scoring."""
    local = test.copy()
    local.index = local.index.tz_localize("UTC").tz_convert("Europe/London")
    calendar = local.resample("1min").asfreq()
    good = calendar.notna().all(axis=1).to_numpy()
    reference = []
    for a, b in cycle_intervals(calendar.actual_w.to_numpy(), good):
        reference.append(
            dict(
                start=calendar.index[a],
                end=calendar.index[b - 1] + pd.Timedelta(minutes=1),
                duration=b - a,
                energy=float(calendar.actual_w.iloc[a:b].sum() / 60000),
            )
        )
    ref = pd.DataFrame(reference)
    d25, d75 = ref.duration.quantile([0.25, 0.75])
    e25, e75 = ref.energy.quantile([0.25, 0.75])
    # Verify sufficiently observed source minutes, not merely finite model means.
    raw = pd.read_csv(
        args.data_dir / "CLEAN_House3.csv", usecols=["Unix", "Aggregate", "Appliance6"]
    )
    valid = np.isfinite(raw[["Aggregate", "Appliance6"]]).all(axis=1) & (
        raw[["Aggregate", "Appliance6"]] >= 0
    ).all(axis=1)
    stats = (
        raw.loc[valid]
        .groupby(raw.loc[valid, "Unix"] // 60)
        .Unix.agg(["size", "min", "max"])
    )
    source_good = stats[
        (stats["size"] >= 3) & (stats["max"] - stats["min"] >= 30)
    ].index
    source_good = pd.to_datetime(
        source_good.to_numpy() * 60, unit="s", utc=True
    ).tz_convert("Europe/London")
    eligible = []
    for day, frame in calendar.groupby(calendar.index.normalize()):
        end = day + pd.DateOffset(days=1)
        if (
            len(frame) != 1440
            or frame.isna().any().any()
            or not frame.index.isin(source_good).all()
        ):
            continue
        candidates = ref[
            (ref.start >= day + pd.Timedelta(minutes=10))
            & (ref.end <= end - pd.Timedelta(minutes=10))
        ].copy()
        accepted = []
        for r in candidates.itertuples():
            before = calendar.loc[
                r.start - pd.Timedelta(minutes=10) : r.start - pd.Timedelta(minutes=1),
                "actual_w",
            ]
            after = calendar.loc[r.end : r.end + pd.Timedelta(minutes=9), "actual_w"]
            if (
                len(before) == 10
                and len(after) == 10
                and (before < 20).all()
                and (after < 20).all()
            ):
                accepted.append(r.Index)
        if not accepted:
            continue
        candidates = candidates.loc[accepted]
        candidates["distance"] = abs(candidates.duration - ref.duration.median()) / max(
            d75 - d25, 1
        ) + abs(candidates.energy - ref.energy.median()) / max(e75 - e25, 1e-6)
        chosen = candidates.sort_values(["distance", "start"]).iloc[0]
        cm = cycle_metrics(frame.actual_w, frame.predicted_w.to_numpy())
        active = frame.actual_w >= 20
        eligible.append(
            dict(
                day=str(day),
                usable_minutes=1440,
                cycle_start=str(chosen.start),
                cycle_end=str(chosen.end),
                duration_minutes=int(chosen.duration),
                energy_kwh=float(chosen.energy),
                representativeness_distance=float(chosen.distance),
                cycle_F1_pct=cm["cycle_F1_pct"],
                active_MAE_W=float(
                    abs(
                        frame.loc[active, "actual_w"] - frame.loc[active, "predicted_w"]
                    ).mean()
                ),
            )
        )
    if not eligible:
        raise ValueError(
            "No complete day with an observed, IQR-representative reference cycle"
        )
    ranked = pd.DataFrame(eligible)
    ranked["distance_to_median_daily_F1"] = abs(
        ranked.cycle_F1_pct - ranked.cycle_F1_pct.median()
    )
    ranked = ranked.sort_values(
        ["distance_to_median_daily_F1", "representativeness_distance", "day"]
    )
    ranked.to_csv(output / "test_example_day_candidates.csv", index=False)
    chosen = ranked.iloc[0].to_dict()
    chosen["timezone"] = "Europe/London"
    chosen["selection"] = (
        "complete observed day; selected reference cycle nearest household medians; nearest median daily cycle F1; representative-cycle distance and date break ties"
    )
    (output / "test_example_selection.json").write_text(json.dumps(chosen, indent=2))
    return (
        local,
        pd.Timestamp(chosen["day"]),
        pd.Timestamp(chosen["cycle_start"]),
        pd.Timestamp(chosen["cycle_end"]),
    )


def table(df):
    def fmt(x):
        if isinstance(x, float):
            return "—" if pd.isna(x) else f"{x:.3f}"
        return str(x)

    return (
        "| "
        + " | ".join(df.columns)
        + " |\n| "
        + " | ".join(["---"] * len(df.columns))
        + " |\n"
        + "\n".join(
            "| " + " | ".join(fmt(x) for x in row) + " |"
            for row in df.itertuples(index=False, name=None)
        )
        + "\n\n"
    )


def identified_false_alarm(test):
    candidate = test[(test.actual_w < 20) & (test.predicted_w >= 20)]
    if candidate.empty:
        return None, False
    timestamps = candidate.index.as_unit("s").asi8
    chunks = []
    cols = ["Unix", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
    for part in pd.read_csv(
        args.data_dir / "CLEAN_House3.csv", usecols=cols, chunksize=500000
    ):
        minutes = part.Unix // 60 * 60
        selected = part[minutes.isin(timestamps)].copy()
        if len(selected):
            selected["minute"] = selected.Unix // 60 * 60
            chunks.append(selected)
    means = pd.concat(chunks).groupby("minute")[cols[1:]].mean()
    non_wm = [f"Appliance{i}" for i in range(1, 10) if i != 6]
    strong = means[non_wm].max(axis=1)
    eligible = means.index[(strong >= 1000) & (strong >= 0.5 * means.Aggregate)]
    subset = candidate[np.isin(candidate.index.as_unit("s").asi8, eligible)]
    if len(subset):
        return (subset.predicted_w - subset.actual_w).idxmax(), True
    return (candidate.predicted_w - candidate.actual_w).idxmax(), False


def make_reports():
    import nbformat
    from nbclient import NotebookClient

    root = Path(__file__).resolve().parents[1]
    metrics = pd.read_csv(O / "evaluation_metrics.csv")
    loho = pd.read_csv(O / "loho_evaluation_metrics.csv")
    cycles = pd.read_csv(O / "cycle_evaluation_metrics.csv")
    summ = pd.read_csv(O / "loho_summary_metrics.csv")
    primary = metrics[(metrics.fold == 3) & (metrics.split == "test")].set_index(
        "predictor"
    )
    model = primary.loc["two_stage_hgb"]
    plain = primary.loc["plain_classifier"]
    off = primary.loc["always_off"]
    profile = primary.loc["usage_profile"]
    c = cycles[
        (cycles.fold == 3)
        & (cycles.split == "test")
        & (cycles.predictor == "two_stage_hgb")
        & (cycles.matching == "overlap_or_start_30min")
    ].iloc[0]
    headline = []
    for name in ["always_off", "usage_profile", "two_stage_hgb"]:
        r = primary.loc[name]
        headline.append(
            {
                "Evaluation": "House 3 test",
                "Model": name,
                "MAE W": r.MAE_W,
                "Active MAE W": r.active_MAE_W,
                "F1 %": r.F1_pct,
                "Energy error %": r.SAE_pct,
            }
        )
        r = loho[loho.predictor == name]
        headline.append(
            {
                "Evaluation": "6-house mean",
                "Model": name,
                "MAE W": r.MAE_W.mean(),
                "Active MAE W": r.active_MAE_W.mean(),
                "F1 %": r.F1_pct.mean(),
                "Energy error %": r.SAE_pct.mean(),
            }
        )
    lead = (
        f"We estimate washing-machine power from one-minute whole-home readings using a simple activity classifier and an active-power regressor. On House 3, minute F1 is {model.F1_pct:.2f}% and cycle F1 is {c.cycle_F1_pct:.2f}%, with {c.median_start_error_min:.1f}-minute median matched start error. Stage 2 active-minute MAE is {model.active_MAE_W:.2f} W versus {plain.active_MAE_W:.2f} W for the plain classifier, while overall MAE is {model.MAE_W:.2f} versus {plain.MAE_W:.2f} W. Both named splits and six-house LOHO show where results transfer and where they fail.\n\n"
        + table(pd.DataFrame(headline))
    )
    coverage_counts = pd.read_csv(O / "data_counts.csv")
    coverage_row = coverage_counts[
        (coverage_counts.fold == 3) & (coverage_counts.split == "test")
    ].iloc[0]
    inactive_pct = 100 * (1 - coverage_row.active_minutes / coverage_row.minutes)
    limitation = f"Overlapping high-power appliances cause false positives; {inactive_pct:.2f}% of House 3 evaluated minutes are below the 20 W activity threshold. House 3 overall MAE is {model.MAE_W:.2f} W versus always-off {off.MAE_W:.2f} W; total-energy error is {model.SAE_pct:.2f}% versus hourly-profile {profile.SAE_pct:.2f}%. The suitable use is evaluated cycle timing and opt-in feedback, not appliance billing.\n\n"
    test = pd.read_csv(
        O / "fold_3_test_predictions.csv.gz", index_col="timestamp", parse_dates=True
    )
    local, day, start, end = select_test_example(test, O)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for ax, a, b in [
        (axes[0], day, day + pd.DateOffset(days=1) - pd.Timedelta(minutes=1)),
        (
            axes[1],
            max(day, start - pd.Timedelta(minutes=15)),
            min(
                day + pd.DateOffset(days=1) - pd.Timedelta(minutes=1),
                end + pd.Timedelta(minutes=15),
            ),
        ),
    ]:
        fields = (
            ["actual_w", "predicted_w", "aggregate_w"]
            if ax is axes[0]
            else ["actual_w", "predicted_w"]
        )
        local.loc[a:b, fields].resample("1min").asfreq().rename(
            columns={
                "actual_w": "Actual WM",
                "predicted_w": "Two-stage prediction",
                "aggregate_w": "Aggregate",
            }
        ).plot(ax=ax)
        ax.set(
            title=f"House 3 | {day.date()}",
            ylabel="Power (W)",
            xlabel="Time (UK Local Time)",
        )
    fig.tight_layout()
    fig.savefig(O / "test_actual_vs_predicted.png", dpi=170)
    plt.close(fig)
    # Diagnose the observed sub-meter contributions during a test false alarm.
    candidates = test[(test.actual_w < 20) & (test.predicted_w >= 20)]
    evidence = []
    at, identified = identified_false_alarm(test)
    if at is not None:
        parts = []
        for chunk in pd.read_csv(args.data_dir / "CLEAN_House3.csv", chunksize=500000):
            hit = chunk[
                chunk.Unix.between(
                    int((at - pd.Timedelta(hours=1)).timestamp()),
                    int((at + pd.Timedelta(hours=1)).timestamp()),
                )
            ]
            if len(hit):
                parts.append(hit)
        raw = pd.concat(parts)
        raw["minute"] = raw.Unix // 60 * 60
        means = raw.groupby("minute").mean(numeric_only=True)
        point = means.loc[int(at.timestamp())]
        names = {
            1: "Toaster",
            2: "Fridge-Freezer",
            3: "Freezer",
            4: "Tumble Dryer",
            5: "Dishwasher",
            6: "Washing Machine",
            7: "Television Site",
            8: "Microwave",
            9: "Kettle",
        }
        evidence = [
            dict(
                channel=f"Appliance{i}",
                appliance=names[i],
                mean_W_at_false_alarm=float(point[f"Appliance{i}"]),
            )
            for i in names
        ]
        evidence = sorted(
            evidence, key=lambda x: x["mean_W_at_false_alarm"], reverse=True
        )
        pd.DataFrame(evidence).to_csv(
            O / "test_false_alarm_appliance_evidence.csv", index=False
        )
        means.to_csv(O / "test_false_alarm_submeter_window.csv", index_label="unix")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    missed = test[(test.actual_w >= 20) & (test.predicted_w < test.actual_w)]
    miss = (missed.actual_w - missed.predicted_w).idxmax() if len(missed) else None
    for ax, center, title in [
        (axes[0], at, "House 3: false-positive example"),
        (axes[1], miss, "House 3: underestimated washing-machine activity"),
    ]:
        if center is None:
            ax.text(0.5, 0.5, "No qualifying case", transform=ax.transAxes)
            continue
        window = test.loc[
            center - pd.Timedelta(hours=1) : center + pd.Timedelta(hours=1)
        ].copy()
        window.index = window.index.tz_localize("UTC").tz_convert("Europe/London")
        window[["actual_w", "predicted_w", "aggregate_w"]].resample(
            "1min"
        ).asfreq().plot(ax=ax)
        ax.set(title=title, ylabel="Power (W)", xlabel="UK Local Time")
    fig.tight_layout()
    fig.savefig(O / "failure_cases_examples.png", dpi=170)
    plt.close(fig)
    eda = (root / "RESULTS.md").read_text(encoding="utf-8")
    eda = eda[eda.index("## 1.") :].split("## 2.")[0]
    text = (
        "# Results\n\n"
        + lead
        + limitation
        + eda
        + """\n## 2. Model and evaluation

For each household and uninterrupted aggregate segment, subtract a trailing 60-minute rolling minimum. Features are current aggregate/base/net power plus trailing 15-, 30- and 60-minute mean, maximum and standard deviation of net power: twelve inputs in total. No clock or appliance inputs enter the trees.

A balanced HistGradientBoostingClassifier estimates the probability of WM >=20 W. The plain-classifier baseline multiplies this probability by training median active power; the two-stage model replaces that constant with an absolute-error HistGradientBoostingRegressor trained on active minutes. Both apply the same probability threshold and retain runs of at least ten minutes, then output probability × watts, clipped to 0–3,000 W. This batch run rule confirms activity retrospectively; it is not immediate automatic control. Threshold is selected only on validation F1, with MAE breaking ties.

Compare the classifier baseline directly with the two-stage model to judge whether Stage 2 helps estimate heating versus lower-power activity. Always-off and a training-only hourly mean provide the other baselines. Evaluation excludes missing aggregate/WM means and WM outside 0–3,500 W; training additionally excludes WM > Aggregate +100 W. Missing aggregate intervals break feature history and missing evaluation intervals break predicted runs.

Split A trains on Houses 3, 7, 8 and 10, validates on House 1 and tests on House 2. Split B trains on Houses 1, 2, 7 and 10, validates on House 8 and tests on House 3. LOHO holds out each home once with the fixed validation rotation recorded below; training always uses the remaining four. This is a within-cohort robustness check, not proof of external generalisation or removal of every source of selection bias.

### Both named splits and baselines

"""
        + table(metrics)
        + "\n### Cycle-level results\n\n"
        + table(cycles)
        + """
Reference and predicted cycles use >=20 W, observed pauses <=15 minutes, duration >=20 minutes AND energy >=0.10 kWh. Standard matching accepts overlap or starts within 30 minutes; strict matching requires start error <=10 minutes. Greedy one-to-one matching uses smallest start error first; matched start/duration/energy errors must be read with precision and recall. These reference cycles are rule-defined operating sessions, not manually labelled programmes.

### Six-house LOHO

"""
        + table(
            loho[loho.predictor == "two_stage_hgb"][
                [
                    "house",
                    "validation_house",
                    "training_houses",
                    "MAE_W",
                    "active_MAE_W",
                    "F1_pct",
                    "cycle_F1_pct",
                    "median_start_error_min",
                    "SAE_pct",
                ]
            ]
        )
        + "\n"
        + table(summ)
        + """
The summary reports unweighted household means, sample SD, medians and Q1–Q3. Results describe these six houses; the full per-fold baseline rows are in loho_evaluation_metrics.csv.

![Test actual versus predicted](reports/model/test_actual_vs_predicted.png)

"""
        + f"The illustration uses {day.date()} with 1,440 observed minutes, nearest the median daily cycle F1 among eligible complete active days. The selected cycle has complete observed context; this figure complements the full-test tables rather than showing the best day.\n\n![Test failures](reports/model/failure_cases_examples.png)\n\n"
    )
    if evidence:
        dominant = next(x for x in evidence if x["channel"] != "Appliance6")
        text += (
            f"At {at} UTC, {dominant['channel']} ({dominant['appliance']}) is the largest measured non-WM load at {dominant['mean_W_at_false_alarm']:.1f} W. The diagnostic example is the largest false error with a non-WM channel >=1,000 W and >=50% of aggregate, if available; otherwise it is the largest unattributed false error. This identifies a measured overlapping load; it does not establish exactly why the classifier fired.\n\n"
            + table(pd.DataFrame(evidence))
        )
    text += (
        "\n### Coverage and selected thresholds\n\n"
        + table(pd.read_csv(O / "data_counts.csv"))
        + "\n"
        + table(pd.DataFrame(json.loads((O / "metadata.json").read_text())["folds"]))
    )
    (root / "RESULTS.md").write_text(text, encoding="utf-8")
    readme = (
        "# REFIT washing-machine disaggregation\n\n"
        + lead
        + limitation
        + """## Reproduce

Use 64-bit Python 3.12 and CPU training with several GB of RAM. Extract raw House 1 Part1/Part2 and official cleaned Houses 1,2,3,7,8,10 outside Git.

```sh
python -m pip install -r requirements.txt
python scripts/run_pipeline.py --raw-dir data/REFIT_RAW_081116 --clean-dir data/CLEAN_REFIT_081116
```

Or run raw_house1.py, run_eda.py and train_model.py in that order, with --data-dir pointing to the matching archive directory. Default outputs are reports/raw_house1, reports/eda and reports/model. Training regenerates the Markdown reports and executes walkthrough.ipynb; --reuse-model can reuse matching cached fits for report/debugging runs. No pretrained model is required for the default command. Source CSVs, caches, predictions and fitted binaries are excluded from Git.

## Deliverables and decisions

- [DATA.md](DATA.md): raw audit, duplicate sensitivity, timestamps and cleaning.
- [RESULTS.md](RESULTS.md): EDA, both household splits, LOHO and baseline/failure analysis.
- [Recommendation.md](Recommendation.md): utility pilot, limits, India transfer and meter resolution.
- [walkthrough.ipynb](walkthrough.ipynb): executed tables and all eight key figures.

Raw cleaning uses stable keep-last deduplication, invalid/negative masking and appliance >3,500 W / aggregate >25,000 W screening before one-minute means. Fill whole bounded runs of <=2 missing minutes; >=3 minutes remain missing. EDA uses >=3 readings spanning >=30 seconds and the simple cycle rule documented in RESULTS.md. Model evaluation uses its separate missing/impossible-target mask.

The model uses a 60-minute rolling minimum, 15/30/60-minute rolling mean/max/std, a classifier and active-only regressor. Validation F1 selects one probability threshold; runs shorter than ten minutes are removed. A constant-active-watt classifier baseline tests whether the second stage helps. All feature windows are built within household aggregate segments before concatenation.

Two household splits are evaluated along with a fixed six-fold leave-one-house-out rotation. No held-out fold chooses its own parameters. Further work should add independent households, verify reference cycle labels and compare programme-aware models. Off-time, overlapping loads, sensor polling and missing data limit interpretation.

## References and implementation

Design choices, assumptions and pilot evaluation are stated explicitly in the source. Coding tools assisted script implementation and plot formatting; their use does not constitute independent validation.

- Murray, Stankovic & Stankovic (2017), [REFIT data descriptor](https://doi.org/10.1038/sdata.2016.122). Dataset: [University of Strathclyde](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/), CC BY 4.0.
- Zhang et al. (2018), [Seq2Point](https://doi.org/10.1609/aaai.v32i1.11873), related context-window research; this implementation uses tabular trees, not that neural model.
"""
    )
    (root / "README.md").write_text(readme, encoding="utf-8")
    false_share = 100 - c.cycle_precision_pct
    miss_share = 100 - c.cycle_recall_pct
    brief = f"""# Utility brief: test cycle feedback with customers

The simple two-stage model estimates when washing-machine sessions occur from one-minute whole-home power. On House 3 its cycle precision/recall/F1 are **{c.cycle_precision_pct:.2f}% / {c.cycle_recall_pct:.2f}% / {c.cycle_F1_pct:.2f}%**, with **{c.median_start_error_min:.1f} minutes** median start error among matched cycles. However, **{false_share:.1f}%** of predicted cycles are unmatched and **{miss_share:.1f}%** of reference cycles are missed. The six-house LOHO results in [RESULTS.md](RESULTS.md) show the variation across homes.

Test overall MAE is **{model.MAE_W:.2f} W**, versus **{off.MAE_W:.2f} W** for always-off. Total-energy error is **{model.SAE_pct:.2f}%**, versus **{profile.SAE_pct:.2f}%** for a training-only hourly profile. {inactive_pct:.2f}% of evaluated minutes are inactive (<20 W); static energy totals alone do not justify the ML model. Its proposed value is evaluated event timing for opt-in feedback, not billing, automatic switching or fault diagnosis.

## Practical pilot

Median detected cycle energy ranges from 0.32–0.79 kWh and peak minute means from 1.93–2.34 kW across the six homes. Household start peaks differ: House 7 at 07:00 and House 10 at 19:00 on the cleaned source clock. Offer tariff-aware shifts from locally confirmed evening peaks and manufacturer-approved cold/eco programme choices; shifting time alone need not save total kWh. Heating's separate energy share and programme choices were not measured.

Run an 8–12-week opt-in randomised pilot, illustratively 400 homes split 200/200 after a shared baseline. Measure treatment-minus-control changes in peak-start share, matched-load kWh/cycle, peak-window maximum demand, peak/total whole-home kWh and actual bills; track rebound, repeat washes, weather and tariff changes. Calibrate cycle detection and false notifications in a sub-metered subset and survey usefulness/opt-outs. Size the final study from blinded pilot variance and household clustering; 400 homes alone does not establish 80% power.

## Transfer and measurement limits

Overlapping heaters, dryers, dishwashers and cooking loads can produce false alarms. Missing data, asynchronous polling and upstream cleaning affect reference labels and boundaries. Before UK-to-India transfer, recruit representative machines and homes, retrain locally and test independently: heater-free/inverter machines, external geysers, AC/fans/pumps, voltage changes, outages and battery backup alter signatures.

At 15/30-minute intervals, switching edges, pauses and overlapping states merge. A 2 kW heater on for 15 minutes contributes 0.5 kWh and averages 1 kW over a containing 30-minute bin. Retrain and validate for coarse energy estimates with wider timing uncertainty; do not claim minute states were recovered by upsampling.
"""
    (root / "Recommendation.md").write_text(brief, encoding="utf-8")
    cells = [
        nbformat.v4.new_markdown_cell("# REFIT walkthrough\n\n" + lead + limitation),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display,Image\nR=Path('reports')\npd.set_option('display.max_columns',None)"
        ),
    ]
    cells += [
        nbformat.v4.new_markdown_cell(
            "## Raw audit\nStable keep-last duplicates, invalid/outlier masking, one-minute means, and interpolation of complete gaps no longer than two minutes."
        ),
        nbformat.v4.new_code_cell(
            "display(pd.DataFrame(list(json.loads((R/'raw_house1/audit.json').read_text()).items()),columns=['Finding','Value']))"
        ),
    ]
    for image in ["raw_house1/before_after", "raw_house1/gap_examples"]:
        cells.append(
            nbformat.v4.new_code_cell(f"display(Image(filename=str(R/'{image}.png')))")
        )
    cells += [
        nbformat.v4.new_markdown_cell(
            "## Washing-machine EDA\nActive >=20 W, observed pauses <=15 minutes, sessions >=20 minutes AND >=0.10 kWh. Missing minutes are not off time."
        ),
        nbformat.v4.new_code_cell(
            "display(pd.read_csv(R/'eda/eda_summary_table.csv'))"
        ),
    ]
    for image in [
        "24h_aggregate_and_wm_trace",
        "detected_wm_cycles_zoom",
        "usage_patterns_hour_weekday",
        "cross_house_cycle_comparison",
    ]:
        cells.append(
            nbformat.v4.new_code_cell(
                f"display(Image(filename=str(R/'eda/figures/{image}.png')))"
            )
        )
    cells += [
        nbformat.v4.new_markdown_cell(
            "## Model evaluation\nOne model with twelve features, one validation-tuned probability threshold and a ten-minute run rule. Compare both named splits, classifier and non-ML baselines, and six-house LOHO; matched-cycle errors must be read with recall."
        ),
        nbformat.v4.new_code_cell(
            "display(pd.read_csv(R/'model/evaluation_metrics.csv'))\ndisplay(pd.read_csv(R/'model/cycle_evaluation_metrics.csv'))\ndisplay(pd.read_csv(R/'model/loho_evaluation_metrics.csv'))\ndisplay(pd.read_csv(R/'model/loho_summary_metrics.csv'))"
        ),
    ]
    for image in ["test_actual_vs_predicted", "failure_cases_examples"]:
        cells.append(
            nbformat.v4.new_code_cell(
                f"display(Image(filename=str(R/'model/{image}.png')))"
            )
        )
    cells.append(
        nbformat.v4.new_markdown_cell(
            f"Illustrative day: {day.date()}, UK local time. It has 1,440 observed minutes and is selected nearest median daily cycle F1. See Recommendation.md for a measured utility pilot and transfer limits."
        )
    )
    if evidence:
        cells.append(
            nbformat.v4.new_code_cell(
                "display(pd.read_csv(R/'model/test_false_alarm_appliance_evidence.csv'))"
            )
        )
    nb = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            }
        },
    )
    NotebookClient(
        nb, timeout=120, resources={"metadata": {"path": str(root)}}
    ).execute()
    nbformat.write(nb, root / "walkthrough.ipynb")


if __name__ == "__main__":
    fit_and_evaluate()
