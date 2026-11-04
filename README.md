# Windstorms on the German transmission grid: line failures, lost load and the value of information for reinforcement (v0.1)

[![tests](https://github.com/michaelserikov/de-storm-grid-risk/actions/workflows/tests.yml/badge.svg)](https://github.com/michaelserikov/de-storm-grid-risk/actions/workflows/tests.yml)

Which storms hurt the transmission grid, which lines matter, and what would better knowledge of line fragility be worth for the decision to reinforce them? This project takes 861 storm footprints from 35 winters of DWD gust records (from [de-windstorm-risk](https://github.com/michaelserikov/de-windstorm-risk)), an open OpenStreetMap model of the German grid at 220 kV and above, a fragility curve, Monte Carlo line failures and a DC power flow that minimises load shedding.

This is an illustrative single-country study. **No public failure data exist for German transmission lines, so the fragility is an assumption, set by the number of line failures assumed in the largest storm (2, 5 or 15), and every lost-load number depends on it.** Compare the numbers with each other, not with real outages. See [Limitations](#limitations).

**Main findings**
- **The storm that costs insurers most is not the storm that costs the grid most.** Kyrill has the largest loss index, but in the central case Lothar loses about five times as much load (258 against 52 MW expected) with the largest number of expected line failures (8.2 against 5.0). Across all events the rank correlation between loss index and lost load is 0.45 (Kendall's tau; 0.37 to 0.60 across the fragility cases).
- **The meshed grid absorbs most failures.** Even when 5 lines are expected to fail in Kyrill, 63% of the samples lose no load. Lost load is a rare and heavy-tailed event: 100 MW about once in 21 winters, 1,000 MW about once in 370.
- **Correlated failures change the tail more than the assumed fragility of any one line suggests.** With unchanged failure probabilities per line, a correlation length of 50 km raises expected lost load from 20 to 33 MW per winter and events above 1,000 MW from about 1 in 440 to 1 in 120 winters; 200 km gives 50 MW and 1 in 56. Parallel circuits and neighbouring corridors failing together is what breaks the N-1 logic.
- **Reinforcement decisions concentrate on few branches.** The five most critical branches carry 20% of the lost load attributed to failed branches and the top 30 carry 54%, out of 1,019 lines.
- **Fragility is worth knowing only in a window of reinforcement costs.** Below about 2.5 thousand EUR per km and year a large program (over 1,000 km) is chosen, and above about 40 (independent failures) or 20 (correlated) nothing is reinforced whatever the fragility is. In between, the expected value of perfect information peaks at about 1.1 MEUR per year (19% of the expected cost, independent failures) and 1.3 MEUR per year (12%, correlated). Resampling winters puts a wide 90% range around the peak: 0.03 to 1.4 and 0.05 to 1.5 MEUR.
- **Imperfect information loses value quickly.** At the peak with independent failures, a signal that names the right fragility case 80% of the time is worth about half of perfect information (54%), and one that is right 60% of the time about a fifth (19%).

## Method

- **Grid.** PyPSA-Eur prebuilt network v0.7, German buses and branches only, largest connected component: 780 buses, 1,019 lines and 153 transformers, 25,030 km of overhead route. Germany is an island. Line susceptance from voltage and reactance, limits from the dataset's thermal rating; transformers with an assumed 0.1 per unit reactance.
- **Generation and load, one winter storm hour.** Plants from powerplantmatching, in operation in 2026, at the nearest bus, with assumed availability (fossil 0.9, hydro 0.5, biomass 0.8, onshore wind 0.55, offshore wind 0.7, solar 0): 142 GW available. Load 70 GW, spread by the population weights of the DWD stations over their three nearest buses. The intact grid already sheds 122 MW through congestion in this simplified setting, which is subtracted from all results.
- **Hazard.** Each kilometre of overhead line takes the event gust of its nearest station as a ratio to that station's 98th percentile, since lines, like buildings, are built for their local wind climate. A missing station takes its nearest reporting neighbour.
- **Fragility.** Lognormal in this ratio, per 10 km of line, beta 0.2. The median follows from the assumed number of expected line failures in the largest storm (Kyrill): 3.01, 2.82 and 2.59 for 2, 5 and 15 failures, giving 0.8, 1.9 and 6.0 failed lines per winter on average.
- **Failures.** Monte Carlo per event, 200 samples (100 in the value of information runs), a failed line losing all its circuits. Independent given the footprint, or with a Gaussian copula with exponential correlation between line midpoints.
- **Lost load.** For each failure set, a linear program (SciPy, HiGHS) over bus angles, generation and shed, with DC power flow and thermal limits.
- **Value of information.** Reinforcing makes a branch immune. Plans are the top 1 to 30 branches ranked under each fragility case on one Monte Carlo sample, evaluated on another, so a line is not credited with the luck of the sample that selected it. Cost per year = reinforcement cost per km x km + lost load x 24 h x 10,000 EUR/MWh (both illustrative). Prior 25%, 50%, 25% on the three fragility cases. EVPI = min over plans of expected cost, minus the expected minimum per case. An imperfect signal is valued by Bayesian updating (EVSI).

## Results

| Expected failures in Kyrill | Failed lines per winter | Lost load per winter (MW) | Kyrill, chance of any loss | Largest sampled loss (MW) | Kendall's tau, loss index vs lost load |
|---|---|---|---|---|---|
| 2 | 0.8 | 6.5 | 24% | 1,861 | 0.37 |
| 5 (central) | 1.9 | 19.4 | 38% | 1,972 | 0.45 |
| 15 | 6.0 | 70.5 | 51% | 2,769 | 0.60 |

Named storms in the central case, with GDV insured losses where the calibration in de-windstorm-risk uses them:

| Storm | Loss index | Expected failed lines | Expected lost load (MW) | Chance of any loss | GDV (MEUR, prices of 2020) |
|---|---|---:|---:|---:|---:|
| Lothar, Dec 1999 | 0.120 | 8.2 | 258 | 68% | not in the source |
| Kyrill, Jan 2007 | 0.155 | 5.0 | 52 | 38% | 3,350 |
| Emma, Mar 2008 | 0.056 | 1.2 | 31 | 18% | 610 |
| Friederike, Jan 2018 | 0.054 | 1.5 | 20 | 16% | 1,160 |
| Niklas, Mar 2015 | 0.050 | 0.7 | 19 | 13% | 730 |
| Sabine, Feb 2020 | 0.055 | 0.8 | 7 | 11% | 600 |
| Ylenia/Zeynep, Feb 2022 | 0.067 | 1.4 | 3 | 14% | 1,250 |

An unnamed event of January 1994 ranks second by lost load (107 MW). Storm names are attached by date only.

Sensitivities of the central case, top 40 events, lost load per winter: load 55 GW 12.5 MW, 70 GW 19.6 MW, 80 GW 25.4 MW; flow limits at 70% of thermal rating 21.6 MW (`results/sensitivity.csv`). Tables by plan and cost are in `results/voi_actions_*.csv` and `results/voi_curve_*.csv`, the critical branches in `results/critical_branches.csv`.

## Data

| Source | Content | Licence |
|---|---|---|
| [Prebuilt network for PyPSA-Eur, v0.7](https://zenodo.org/records/18619025), Xiong et al. (2025), Scientific Data 12, 277 | Buses, lines, transformers from OpenStreetMap | ODbL 1.0, © OpenStreetMap contributors |
| [powerplantmatching](https://github.com/PyPSA/powerplantmatching) | Plants with location and capacity | See the project |
| DWD Climate Data Center via de-windstorm-risk (`data/hazard/`) | Event gust footprints, stations, events | CC BY 4.0, source: Deutscher Wetterdienst |

## Run

```bash
pip install -e ".[test]"
python scripts/download_data.py   # network and plants into data/raw/, not committed
python -m stormgrid.run           # about 15 minutes: results/summary_by_fragility.csv, top_events.csv, exceedance.csv
python -m stormgrid.value         # about 30 minutes: correlated failures and value of information
pytest                            # offline tests on small grids with answers known by hand
```

## Limitations

- **The fragility is assumed.** Nothing here is calibrated to observed line failures. The three cases span a guess, the shape is one lognormal, and the prior on the cases is a judgement. Checking against open outage records is the next step.
- **Transmission only.** Most storm outages happen in distribution grids, which are not modelled, so the lost load is far from total customer outages.
- **One static hour.** Fixed load, no redispatch costs, no protection, no cascading, no restoration process and wind output at a fixed level, even though turbines may cut out in a storm. Germany as an island removes support from neighbours. Station gusts stand in for a wind field.
- **Value of information.** Plans are limited to the ranked lists. The numbers scale with the assumed 24 hours and 10,000 EUR/MWh. Reinforcement is treated as making a branch immune, and a sample without loss is assumed to stay without loss after reinforcement, ignoring rare cases where an added line worsens flows. Noise in the Monte Carlo estimate biases the minimum over plans, so EVPI near zero should be read as zero.
- **Today's climate.** The footprints are 1990/91 to 2024/25. A change in storms under future climate is not included.

## Reference

Xiong, B., Fioriti, D., Neumann, F., Riepin, I., Brown, T. (2025). Modelling the high-voltage grid using open data for Europe and beyond. Scientific Data, 12, 277. https://doi.org/10.1038/s41597-025-04550-7
