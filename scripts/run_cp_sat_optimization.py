from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ortools.sat.python import cp_model
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install OR-Tools first: pip install ortools") from exc


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "nyc311"
RESULTS = ROOT / "results"


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1 / (1 + np.exp(-z))


def fit_logistic_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.15, epochs: int = 500, l2: float = 0.002) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = sigmoid(X @ beta)
        grad = X.T @ (p - y) / len(y) + l2 * np.r_[0, beta[1:]]
        beta -= lr * grad
    return beta


def target_encode(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], y_col: str, alpha: float = 30.0) -> tuple[pd.Series, pd.Series]:
    global_mean = float(train[y_col].mean())
    key_train = train[cols].astype(str).agg("|".join, axis=1)
    key_test = test[cols].astype(str).agg("|".join, axis=1)
    stat = train.assign(_key=key_train).groupby("_key")[y_col].agg(["sum", "count"])
    enc = (stat["sum"] + alpha * global_mean) / (stat["count"] + alpha)
    return key_train.map(enc).fillna(global_mean), key_test.map(enc).fillna(global_mean)


def make_model_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["created_date"] = pd.to_datetime(d["created_date"])
    d["month"] = d["created_date"].dt.month
    d["week"] = d["created_date"].dt.isocalendar().week.astype(int)
    d["dow"] = d["created_date"].dt.dayofweek
    d["hour"] = d["created_date"].dt.hour
    d["response_days"] = d["response_hours"] / 24
    threshold = d["response_days"].quantile(0.75)
    d["slow_response"] = (d["response_days"] >= threshold).astype(int)

    # Month-stratified split approximates the policy setting in which historical multi-year data
    # have already exposed the model to each calendar month.
    rng = np.random.default_rng(42)
    test_mask = np.zeros(len(d), dtype=bool)
    for _, idx in d.groupby("month").groups.items():
        idx = np.array(list(idx))
        test_n = max(1, int(round(len(idx) * 0.25)))
        test_mask[rng.choice(idx, size=test_n, replace=False)] = True
    train = d.loc[~test_mask].copy()
    test = d.loc[test_mask].copy()

    base_cols = ["complaint_type", "descriptor", "borough", "incident_zip", "community_board", "agency", "month", "dow", "hour"]
    Xtr = pd.get_dummies(train[base_cols].astype(str), drop_first=False, dtype=float)
    Xte = pd.get_dummies(test[base_cols].astype(str), drop_first=False, dtype=float).reindex(columns=Xtr.columns, fill_value=0)

    te_groups = [
        ["complaint_type"],
        ["descriptor"],
        ["incident_zip"],
        ["community_board"],
        ["agency", "complaint_type"],
        ["incident_zip", "complaint_type"],
        ["community_board", "complaint_type"],
        ["descriptor", "borough"],
    ]
    for group in te_groups:
        name = "te_" + "_".join(group)
        tr_enc, te_enc = target_encode(train, test, group, "slow_response")
        Xtr[name] = tr_enc.to_numpy()
        Xte[name] = te_enc.to_numpy()

    means = Xtr.mean()
    stds = Xtr.std(ddof=0).replace(0, 1)
    X_train = np.column_stack([np.ones(len(train)), ((Xtr - means) / stds).to_numpy()])
    X_test = np.column_stack([np.ones(len(test)), ((Xte - means) / stds).to_numpy()])
    beta = fit_logistic_gd(X_train, train["slow_response"].to_numpy())
    test["risk"] = sigmoid(X_test @ beta)
    return test


def add_time_series_delay(df: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    train = df.copy()
    train["created_date"] = pd.to_datetime(train["created_date"])
    train["week"] = train["created_date"].dt.isocalendar().week.astype(int)
    train["response_days"] = train["response_hours"] / 24
    weekly = train.groupby(["complaint_type", "week"])["response_days"].median().reset_index()
    global_median = train["response_days"].median()

    lookup: dict[tuple[str, int], float] = {}
    for complaint_type, part in weekly.groupby("complaint_type"):
        s = part.set_index("week")["response_days"].sort_index()
        for week in range(1, 54):
            previous = s.loc[s.index < week].tail(4)
            lookup[(complaint_type, week)] = float(previous.mean()) if len(previous) else float(s.median())

    scored = scored.copy()
    scored["ts_delay"] = [
        lookup.get((row.complaint_type, int(row.week)), global_median)
        for row in scored.itertuples(index=False)
    ]
    return scored


def prepare_optimization_sample(scored: pd.DataFrame, n: int = 1200) -> pd.DataFrame:
    d = scored.dropna(subset=["risk", "ts_delay", "response_hours"]).copy()
    d["median_income"] = pd.to_numeric(d.get("median_income"), errors="coerce")
    d["poverty_rate"] = pd.to_numeric(d.get("poverty_rate"), errors="coerce")
    d["hpd_violations"] = pd.to_numeric(d.get("hpd_violations"), errors="coerce")
    d["agency_borough_week_volume"] = pd.to_numeric(d.get("agency_borough_week_volume"), errors="coerce")

    for c in ["median_income", "poverty_rate", "hpd_violations", "agency_borough_week_volume"]:
        d[c] = d[c].fillna(d[c].median())

    income_rank = d["median_income"].rank(pct=True)
    poverty_rank = d["poverty_rate"].rank(pct=True)
    hpd_rank = d["hpd_violations"].rank(pct=True)
    d["vulnerability"] = ((1 - income_rank) + poverty_rank + hpd_rank) / 3
    d["vulnerable"] = d["vulnerability"] >= d["vulnerability"].quantile(0.75)
    d["workload_rank"] = d["agency_borough_week_volume"].rank(pct=True)
    d["delay_rank"] = d["ts_delay"].rank(pct=True)
    d["cost"] = 1 + 0.70 * d["delay_rank"] + 0.35 * d["workload_rank"] + 0.20 * d["vulnerable"].astype(float)
    d["expected_saved"] = 0.2 * d["risk"] * d["ts_delay"]
    d["fair_value"] = d["expected_saved"] * (1 + 1.35 * d["vulnerability"]) * (1 + 0.25 * d["workload_rank"])

    if len(d) > n:
        d = d.sample(n=n, random_state=42)
    return d.reset_index(drop=True)


def solve_ip(d: pd.DataFrame, objective_col: str, require_fair: bool = False) -> dict:
    scale = 1000
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x_{i}") for i in range(len(d))]
    cost = np.round(d["cost"].to_numpy() * scale).astype(int)
    budget = int(round(0.25 * cost.sum()))

    model.Add(sum(int(cost[i]) * x[i] for i in range(len(d))) <= budget)

    for agency, idx in d.groupby("agency").groups.items():
        cap = int(round(0.45 * cost[list(idx)].sum()))
        model.Add(sum(int(cost[i]) * x[i] for i in idx) <= cap)
    for complaint_type, idx in d.groupby("complaint_type").groups.items():
        cap = int(round(0.38 * cost[list(idx)].sum()))
        model.Add(sum(int(cost[i]) * x[i] for i in idx) <= cap)
    for borough, idx in d.groupby("borough").groups.items():
        minimum = int(round(0.08 * budget * len(idx) / len(d)))
        model.Add(sum(int(cost[i]) * x[i] for i in idx) >= minimum)
    if require_fair:
        vulnerable_idx = np.where(d["vulnerable"].to_numpy())[0]
        model.Add(sum(int(cost[i]) * x[i] for i in vulnerable_idx) >= int(round(0.38 * budget)))

    obj = np.round(d[objective_col].to_numpy() * scale).astype(int)
    model.Maximize(sum(int(obj[i]) * x[i] for i in range(len(d))))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    chosen = np.array([solver.Value(v) for v in x], dtype=bool)

    actual_saved = float((d.loc[chosen, "response_hours"] * 0.2 / 24).sum())
    slow_total = max(int(d["slow_response"].sum()), 1)
    vuln_slow_total = max(int((d["vulnerable"] & (d["slow_response"] == 1)).sum()), 1)
    return {
        "status": solver.StatusName(status),
        "selected_n": int(chosen.sum()),
        "actual_days_saved": actual_saved,
        "expected_saved_ts": float(d.loc[chosen, "expected_saved"].sum()),
        "slow_captured_pct": float(100 * ((chosen) & (d["slow_response"].to_numpy() == 1)).sum() / slow_total),
        "selected_slow_rate": float(d.loc[chosen, "slow_response"].mean()) if chosen.any() else 0.0,
        "avg_vulnerability": float(d.loc[chosen, "vulnerability"].mean()) if chosen.any() else 0.0,
        "vulnerable_slow_captured_pct": float(100 * ((chosen) & d["vulnerable"].to_numpy() & (d["slow_response"].to_numpy() == 1)).sum() / vuln_slow_total),
    }


def random_baseline(d: pd.DataFrame) -> dict:
    rng = np.random.default_rng(42)
    order = rng.permutation(len(d))
    budget = 0.25 * d["cost"].sum()
    selected = []
    running = 0.0
    for i in order:
        if running + d.loc[i, "cost"] <= budget:
            selected.append(i)
            running += float(d.loc[i, "cost"])
    chosen = np.zeros(len(d), dtype=bool)
    chosen[selected] = True
    slow_total = max(int(d["slow_response"].sum()), 1)
    vuln_slow_total = max(int((d["vulnerable"] & (d["slow_response"] == 1)).sum()), 1)
    return {
        "status": "FEASIBLE_RANDOM",
        "selected_n": int(chosen.sum()),
        "actual_days_saved": float((d.loc[chosen, "response_hours"] * 0.2 / 24).sum()),
        "expected_saved_ts": float(d.loc[chosen, "expected_saved"].sum()),
        "slow_captured_pct": float(100 * (chosen & (d["slow_response"].to_numpy() == 1)).sum() / slow_total),
        "selected_slow_rate": float(d.loc[chosen, "slow_response"].mean()) if chosen.any() else 0.0,
        "avg_vulnerability": float(d.loc[chosen, "vulnerability"].mean()) if chosen.any() else 0.0,
        "vulnerable_slow_captured_pct": float(100 * (chosen & d["vulnerable"].to_numpy() & (d["slow_response"].to_numpy() == 1)).sum() / vuln_slow_total),
    }


def main() -> None:
    source = OUT / "nyc311_mechanism_features.csv"
    if not source.exists():
        raise SystemExit("Run scripts/analyze_nyc311_response.py first to create outputs/nyc311/nyc311_mechanism_features.csv")
    df = pd.read_csv(source, low_memory=False)
    scored = make_model_scores(df)
    scored = add_time_series_delay(df, scored)
    sample = prepare_optimization_sample(scored)

    rows = []
    rows.append({"strategy": "random_feasible_baseline", **random_baseline(sample)})
    rows.append({"strategy": "historical_delay_ip", **solve_ip(sample, "ts_delay")})
    rows.append({"strategy": "risk_efficiency_ip", **solve_ip(sample, "expected_saved")})
    rows.append({"strategy": "fairness_weighted_ip", **solve_ip(sample, "fair_value", require_fair=True)})

    res = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "cp_sat_optimization_results.csv", index=False, encoding="utf-8-sig")
    res.to_csv(OUT / "cp_sat_optimization_results.csv", index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
