# Data Mining in Social Science: NYC 311 Public Service Response

This repository contains the paper, analysis code, summary results, and figures for a course paper on public service response efficiency and risk warning in digital government platforms. The empirical case is New York City's 311 service request system.

## Repository Structure

- `paper/main.tex`: LaTeX paper.
- `paper/`: figures used by the LaTeX paper.
- `scripts/analyze_nyc311_response.py`: main data cleaning, descriptive statistics, prediction, mechanism analysis, time-series summaries, and figure generation script.
- `scripts/run_cp_sat_optimization.py`: supplementary 0-1 integer-programming optimization script corresponding to the policy allocation section.
- `scripts/svg_to_png_simple.py`: helper for converting simple SVG charts to PNG.
- `results/`: lightweight result tables used in the paper.

Large raw data files are not committed to the repository. They can be downloaded from the public sources below or fetched/cached by the scripts.

## Data Sources

1. **NYC 311 Service Requests from 2020 to Present**
   - Portal: <https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9>
   - API endpoint used by the script: <https://data.cityofnewyork.us/resource/erm2-nwe9.csv>
   - Paper sample: closed requests created in 2023 for six complaint types: `Noise - Residential`, `HEAT/HOT WATER`, `UNSANITARY CONDITION`, `Street Condition`, `Street Light Condition`, and `Traffic Signal Condition`.

2. **American Community Survey ZCTA-level community variables**
   - Census Reporter API: <https://api.censusreporter.org/>
   - Tables used: `B19013` median household income, `B17001` poverty, and `B25003` tenure/renter share.
   - The paper uses ACS 2024 five-year ZCTA-level community background variables where available.

3. **NYC HPD Housing Maintenance Code Violations**
   - Portal: <https://data.cityofnewyork.us/Housing-Development/Housing-Maintenance-Code-Violations/wvxf-dwi5>
   - API endpoint used by the script: <https://data.cityofnewyork.us/resource/wvxf-dwi5.csv>
   - Paper sample: violations approved in 2023, aggregated by ZIP code.

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main analysis:

```bash
python scripts/analyze_nyc311_response.py
```

The script downloads public data where possible, caches raw CSV files under `data/`, and writes outputs under `outputs/nyc311/`.

Run the supplementary CP-SAT optimization after the main analysis has produced `outputs/nyc311/nyc311_mechanism_features.csv`:

```bash
python scripts/run_cp_sat_optimization.py
```

## Notes

- The uploaded `results/` directory contains lightweight summary tables rather than full raw records.
- The paper treats income, poverty, housing violations, and workload primarily as mechanism and association variables. Except where explicitly stated, the empirical design should not be read as proving strict causal effects.
- ZIP/ZCTA matching is an approximate spatial join at the postal-code level.
