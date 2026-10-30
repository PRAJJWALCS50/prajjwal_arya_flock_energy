# Utility brief: test cycle feedback with customers

The simple two-stage model estimates when washing-machine sessions occur from one-minute whole-home power. On House 3 its cycle precision/recall/F1 are **43.41% / 65.09% / 52.09%**, with **9.0 minutes** median start error among matched cycles. However, **56.6%** of predicted cycles are unmatched and **34.9%** of reference cycles are missed. The six-house LOHO results in [RESULTS.md](RESULTS.md) show the variation across homes.

Test overall MAE is **48.49 W**, versus **28.65 W** for always-off. Total-energy error is **53.03%**, versus **15.86%** for a training-only hourly profile. 94.34% of evaluated minutes are inactive (<20 W); static energy totals alone do not justify the ML model. Across six held-out houses, always-off mean MAE is **26.23 W** and usage_profile mean energy error is **61.79%**. Its proposed value is evaluated event timing for opt-in feedback, not billing, automatic switching or fault diagnosis.

## Practical pilot

Median detected cycle energy ranges from 0.32–0.79 kWh and peak minute means from 1.93–2.34 kW across the six homes. Household start peaks differ: House 7 at 07:00 and House 10 at 19:00 on the cleaned source clock. Offer tariff-aware shifts from locally confirmed evening peaks and manufacturer-approved cold/eco programme choices; shifting time alone need not save total kWh. Heating's separate energy share and programme choices were not measured.

Run an 8–12-week opt-in randomised pilot, illustratively 400 homes split 200/200 after a shared baseline. Measure treatment-minus-control changes in peak-start share, matched-load kWh/cycle, peak-window maximum demand, peak/total whole-home kWh and actual bills; track rebound, repeat washes, weather and tariff changes. Calibrate cycle detection and false notifications in a sub-metered subset and survey usefulness/opt-outs. Size the final study from blinded pilot variance and household clustering; 400 homes alone does not establish 80% power.

## Transfer and measurement limits

Overlapping heaters, dryers, dishwashers and cooking loads can produce false alarms. Missing data, asynchronous polling and upstream cleaning affect reference labels and boundaries. Before UK-to-India transfer, recruit representative machines and homes, retrain locally and test independently: heater-free/inverter machines, external geysers, AC/fans/pumps, voltage changes, outages and battery backup alter signatures.

At 15/30-minute intervals, switching edges, pauses and overlapping states merge. A 2 kW heater on for 15 minutes contributes 0.5 kWh and averages 1 kW over a containing 30-minute bin. Retrain and validate for coarse energy estimates with wider timing uncertainty; do not claim minute states were recovered by upsampling.
