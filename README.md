# REFIT washing-machine disaggregation

We estimate washing-machine power from one-minute whole-home readings using a simple activity classifier and an active-power regressor. On House 3, minute F1 is 42.84% and cycle F1 is 52.09%, with 9.0-minute median matched start error. Stage 2 lowers active-minute MAE (six-house mean 357 W vs 480 W) but raises overall MAE, RMSE and total-energy error. Both models share one classifier, so minute F1 is identical.Both named splits and six-house LOHO show where results transfer and where they fail.

| Evaluation | Model | MAE W | Active MAE W | F1 % | Energy error % |
| --- | --- | --- | --- | --- | --- |
| House 3 test | always_off | 28.649 | 505.480 | 0.000 | 100.000 |
| 6-house mean | always_off | 26.226 | 527.546 | 0.000 | 100.000 |
| House 3 test | usage_profile | 48.610 | 470.021 | 16.705 | 15.863 |
| 6-house mean | usage_profile | 48.093 | 500.645 | 11.481 | 61.792 |
| House 3 test | two_stage_hgb | 48.487 | 347.226 | 42.836 | 53.030 |
| 6-house mean | two_stage_hgb | 46.667 | 356.673 | 36.995 | 78.569 |

House 3 overall MAE is 48.49 W vs 28.65 W for always-off (six-house mean 46.67 W vs 26.23 W). Total-energy error is 53.03% (six-house mean 78.57%) vs 15.86% for the hourly profile on House 3 (six-house mean 61.79%). Neither gives reliable energy totals. The suitable use is opt-in cycle-timing feedback, not billing.

## Model choice

An exploratory variant with future-window features reached 60.06% House 3 minute F1, compared with 42.84% for the final model reported here. It was deliberately dropped because it used future observations and was harder to justify in an interview. Houses 3 and 8 were seen during that exploratory work, so House 3 is not a pristine holdout. We therefore report six-house LOHO as a within-cohort robustness check and still require independent households for external validation.

## Reproduce

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

## Model choice
An earlier, more complex variant (centered future-context windows, soft probability gating, constrained validation search) scored higher on House 3 (minute F1 60.1%, cycle F1 60.7%). I dropped it because it used future data and was harder to justify. Houses 3 and 8 were examined during that work, so House 3 is not a pristine holdout; this is why leave-one-house-out results are reported.

## Not implemented
A non-ML heating-plateau baseline, post-hoc energy calibration, Seq2Point/HMM models and an empirical 15/30-minute resampling test were not run. See RESULTS.md for the effect on conclusions.

## References and implementation

The author set the requirements and selected or approved the cleaning thresholds, cycle rule, final model, validation and pilot design. AI coding tools assisted design refinement, script implementation, debugging, drafting and plot formatting; results are checked against the saved outputs.

## AI tools used
I designed the cleaning thresholds, cycle rule, model, validation splits and pilot. An AI coding assistant (OpenAI Codex) helped implement scripts, debug and draft report text. I reviewed the outputs.

- Murray, Stankovic & Stankovic (2017), [REFIT data descriptor](https://doi.org/10.1038/sdata.2016.122). Dataset: [University of Strathclyde](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/), CC BY 4.0.
- Zhang et al. (2018), [Seq2Point](https://doi.org/10.1609/aaai.v32i1.11873), related context-window research; this implementation uses tabular trees, not that neural model.
