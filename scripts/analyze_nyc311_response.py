from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "nyc311"
FIG = OUT / "figures"

BASE = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
COMPLAINT_TYPES = [
    "Noise - Residential",
    "HEAT/HOT WATER",
    "UNSANITARY CONDITION",
    "Street Condition",
    "Street Light Condition",
    "Traffic Signal Condition",
]

COLS = [
    "unique_key", "created_date", "closed_date", "agency", "complaint_type",
    "descriptor", "borough", "incident_zip", "community_board", "status",
    "latitude", "longitude",
]

BLUE = "#5477C4"
BLUE_LIGHT = "#CEDFFE"
ORANGE = "#F0986E"
GOLD = "#FFE15B"
OLIVE = "#71B436"
PINK = "#F390CA"
INK = "#1F2430"
MUTED = "#6F768A"
GRID = "#E6E8F0"
PANEL = "#FFFFFF"
SURFACE = "#FCFCFD"


def fetch_type(complaint_type: str, limit: int = 60_000) -> pd.DataFrame:
    safe = complaint_type.lower().replace("/", "_").replace(" ", "_").replace("-", "").replace("__", "_")
    path = DATA / f"nyc311_2023_{safe}.csv"
    if path.exists() and path.stat().st_size > 1000:
        print(f"cached: {path.name}")
        return pd.read_csv(path)
    where = (
        "created_date between '2023-01-01T00:00:00' and '2024-01-01T00:00:00' "
        f"and complaint_type='{complaint_type}' and closed_date is not null "
        "and borough not in('Unspecified')"
    )
    params = {
        "$select": ",".join(COLS),
        "$where": where,
        "$limit": str(limit),
        "$order": "created_date",
    }
    url = BASE + "?" + urlencode(params)
    print(f"download: {complaint_type}")
    df = pd.read_csv(url, low_memory=False)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def load_data() -> pd.DataFrame:
    DATA.mkdir(exist_ok=True)
    parts = [fetch_type(t) for t in COMPLAINT_TYPES]
    df = pd.concat(parts, ignore_index=True)
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["closed_date"] = pd.to_datetime(df["closed_date"], errors="coerce")
    df = df.dropna(subset=["created_date", "closed_date", "complaint_type", "borough", "community_board"])
    df = df[df["closed_date"] >= df["created_date"]].copy()
    df["response_hours"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600
    # Remove extreme records likely reflecting administrative closure artifacts.
    df = df[(df["response_hours"] >= 0) & (df["response_hours"] <= 24 * 60)].copy()
    df["response_days"] = df["response_hours"] / 24
    df["month"] = df["created_date"].dt.to_period("M").astype(str)
    df["incident_zip"] = pd.to_numeric(df["incident_zip"], errors="coerce").astype("Int64").astype(str).str.zfill(5)
    df["community_board"] = df["community_board"].astype(str).str.replace(" Unspecified", "", regex=False)
    df = df[~df["community_board"].str.contains("Unspecified", case=False, na=False)].copy()
    borough_token = {
        "BRONX": "BRONX",
        "BROOKLYN": "BROOKLYN",
        "MANHATTAN": "MANHATTAN",
        "QUEENS": "QUEENS",
        "STATEN ISLAND": "STATEN ISLAND",
    }
    df = df[df.apply(lambda r: borough_token.get(r["borough"], "") in r["community_board"], axis=1)].copy()
    df.to_csv(OUT / "nyc311_clean_2023_selected.csv", index=False, encoding="utf-8-sig")
    return df


def load_acs_zip() -> pd.DataFrame:
    path = DATA / "acs2024_zcta_nyc.csv"
    if path.exists() and path.stat().st_size > 1000:
        return pd.read_csv(path, dtype={"zip": str})
    nyc_zips = set(pd.read_csv(OUT / "nyc311_clean_2023_selected.csv", dtype={"incident_zip": str})["incident_zip"].dropna().unique())
    url = (
        "https://api.censusreporter.org/1.0/data/show/latest?"
        + urlencode({"table_ids": "B19013,B17001,B25003", "geo_ids": "860|04000US36"})
    )
    request = Request(url, headers={"User-Agent": "social-science-course-paper/1.0"})
    with urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = []
    for geo_id, tables in payload.get("data", {}).items():
        zip_code = geo_id.replace("86000US", "")
        if zip_code not in nyc_zips:
            continue
        est_income = tables.get("B19013", {}).get("estimate", {})
        est_poverty = tables.get("B17001", {}).get("estimate", {})
        est_tenure = tables.get("B25003", {}).get("estimate", {})
        rows.append({
            "zip": zip_code,
            "median_income": est_income.get("B19013001"),
            "poverty_total": est_poverty.get("B17001001"),
            "poverty_count": est_poverty.get("B17001002"),
            "tenure_total": est_tenure.get("B25003001"),
            "renter_count": est_tenure.get("B25003003"),
        })
    df = pd.DataFrame(rows)
    for c in ["median_income", "poverty_total", "poverty_count", "tenure_total", "renter_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["poverty_rate"] = df["poverty_count"] / df["poverty_total"].replace(0, np.nan)
    df["renter_rate"] = df["renter_count"] / df["tenure_total"].replace(0, np.nan)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df


def load_hpd_zip() -> pd.DataFrame:
    path = DATA / "hpd_violations_zip_2023.csv"
    if path.exists() and path.stat().st_size > 1000:
        return pd.read_csv(path, dtype={"zip": str})
    cols = ["violationid", "boro", "zip", "class", "approveddate", "currentstatus"]
    where = "approveddate between '2023-01-01T00:00:00' and '2024-01-01T00:00:00'"
    url = (
        "https://data.cityofnewyork.us/resource/wvxf-dwi5.csv?"
        + urlencode({"$select": ",".join(cols), "$where": where, "$limit": "500000"})
    )
    try:
        df = pd.read_csv(url, low_memory=False)
    except Exception as exc:
        print(f"HPD violations download failed: {exc}")
        return pd.DataFrame(columns=["zip", "hpd_violations", "hpd_class_c_share"])
    if "zip" not in df:
        return pd.DataFrame(columns=["zip", "hpd_violations", "hpd_class_c_share"])
    df["zip"] = pd.to_numeric(df["zip"], errors="coerce").astype("Int64").astype(str).str.zfill(5)
    agg = df.groupby("zip").agg(
        hpd_violations=("violationid", "size"),
        hpd_class_c_share=("class", lambda s: (s.astype(str).str.upper() == "C").mean()),
    ).reset_index()
    agg.to_csv(path, index=False, encoding="utf-8-sig")
    return agg


def add_mechanism_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["week"] = d["created_date"].dt.to_period("W-MON").dt.start_time
    workload = d.groupby(["agency", "borough", "week"]).size().rename("agency_borough_week_volume").reset_index()
    d = d.merge(workload, on=["agency", "borough", "week"], how="left")
    acs = load_acs_zip()
    hpd = load_hpd_zip()
    d = d.merge(acs[["zip", "median_income", "poverty_rate", "renter_rate"]], left_on="incident_zip", right_on="zip", how="left")
    if not hpd.empty:
        d = d.merge(hpd[["zip", "hpd_violations", "hpd_class_c_share"]], left_on="incident_zip", right_on="zip", how="left", suffixes=("", "_hpd"))
    else:
        d["hpd_violations"] = np.nan
        d["hpd_class_c_share"] = np.nan
    d["hpd_violations"] = d["hpd_violations"].fillna(0)
    d.to_csv(OUT / "nyc311_mechanism_features.csv", index=False, encoding="utf-8-sig")
    return d


def adjusted_community_index(df: pd.DataFrame) -> pd.DataFrame:
    type_median = df.groupby("complaint_type")["response_hours"].median().rename("type_median_hours")
    d = df.join(type_median, on="complaint_type")
    d["relative_response"] = d["response_hours"] / d["type_median_hours"].replace(0, np.nan)
    board = (
        d.groupby(["borough", "community_board"])
        .agg(
            n=("unique_key", "size"),
            median_days=("response_days", "median"),
            p75_days=("response_days", lambda s: s.quantile(0.75)),
            slow_index=("relative_response", "median"),
        )
        .reset_index()
    )
    board = board[board["n"] >= 200].sort_values("slow_index", ascending=False)
    board.to_csv(OUT / "community_board_slow_index.csv", index=False, encoding="utf-8-sig")
    return board


def regression_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    # OLS on log response hours with complaint type and month controls, then compare borough residuals.
    d = df.copy()
    d["log_hours"] = np.log1p(d["response_hours"])
    X = pd.get_dummies(d[["complaint_type", "month"]], drop_first=True, dtype=float)
    X.insert(0, "intercept", 1.0)
    beta, *_ = np.linalg.lstsq(X.to_numpy(), d["log_hours"].to_numpy(), rcond=None)
    yhat = X.to_numpy() @ beta
    resid = d["log_hours"].to_numpy() - yhat
    d["residual_log_hours"] = resid
    borough = d.groupby("borough").agg(
        n=("unique_key", "size"),
        raw_median_days=("response_days", "median"),
        adjusted_residual=("residual_log_hours", "mean"),
    ).reset_index()
    borough["adjusted_percent_vs_average"] = (np.exp(borough["adjusted_residual"]) - 1) * 100
    borough.to_csv(OUT / "borough_adjusted_response.csv", index=False, encoding="utf-8-sig")
    return borough


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1 / (1 + np.exp(-z))


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_logistic_gd(X: np.ndarray, y: np.ndarray, lr: float = 0.15, epochs: int = 700, l2: float = 0.002) -> np.ndarray:
    beta = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = sigmoid(X @ beta)
        grad = X.T @ (p - y) / len(y)
        grad[1:] += l2 * beta[1:]
        beta -= lr * grad
    return beta


def fit_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden: int = 24,
    lr: float = 0.04,
    epochs: int = 260,
    l2: float = 0.0005,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.08, size=(X.shape[1], hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.08, size=hidden)
    b2 = 0.0
    y = y.astype(float)
    for _ in range(epochs):
        H_pre = X @ W1 + b1
        H = np.maximum(H_pre, 0)
        p = sigmoid(H @ W2 + b2)
        dz = p - y
        gW2 = H.T @ dz / len(y) + l2 * W2
        gb2 = float(dz.mean())
        dH = dz[:, None] * W2[None, :]
        dH[H_pre <= 0] = 0
        gW1 = X.T @ dH / len(y) + l2 * W1
        gb1 = dH.mean(axis=0)
        W2 -= lr * gW2
        b2 -= lr * gb2
        W1 -= lr * gW1
        b1 -= lr * gb1
    return W1, b1, W2, b2


def mlp_predict(X: np.ndarray, model: tuple[np.ndarray, np.ndarray, np.ndarray, float]) -> np.ndarray:
    W1, b1, W2, b2 = model
    H = np.maximum(X @ W1 + b1, 0)
    return sigmoid(H @ W2 + b2)


def categorical_nb_scores(train: pd.DataFrame, test: pd.DataFrame, features: list[str], y_col: str) -> np.ndarray:
    y = train[y_col].astype(int)
    prior = (y.sum() + 1) / (len(y) + 2)
    scores = np.full(len(test), math.log(prior / (1 - prior)))
    for f in features:
        values = sorted(set(train[f].astype(str)).union(set(test[f].astype(str))))
        pos_counts = train[y == 1][f].astype(str).value_counts()
        neg_counts = train[y == 0][f].astype(str).value_counts()
        pos_total = int((y == 1).sum())
        neg_total = int((y == 0).sum())
        k = len(values)
        pos_prob = {v: (pos_counts.get(v, 0) + 1) / (pos_total + k) for v in values}
        neg_prob = {v: (neg_counts.get(v, 0) + 1) / (neg_total + k) for v in values}
        scores += test[f].astype(str).map(lambda v: math.log(pos_prob[v] / neg_prob[v])).to_numpy()
    return sigmoid(scores)


def ml_slow_response_experiment(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    threshold = float(d["response_days"].quantile(0.75))
    d["slow_response"] = (d["response_days"] >= threshold).astype(int)
    d["created_month_num"] = d["created_date"].dt.month
    d["created_dow"] = d["created_date"].dt.dayofweek
    d["created_hour"] = d["created_date"].dt.hour

    train = d[d["created_date"] < "2023-10-01"].copy()
    test = d[d["created_date"] >= "2023-10-01"].copy()

    feature_cols = ["complaint_type", "borough", "created_month_num", "created_dow", "created_hour"]
    train_X_df = pd.get_dummies(train[feature_cols].astype(str), drop_first=False, dtype=float)
    test_X_df = pd.get_dummies(test[feature_cols].astype(str), drop_first=False, dtype=float)
    test_X_df = test_X_df.reindex(columns=train_X_df.columns, fill_value=0.0)

    means = train_X_df.mean()
    stds = train_X_df.std(ddof=0).replace(0, 1)
    X_train = ((train_X_df - means) / stds).to_numpy()
    X_test = ((test_X_df - means) / stds).to_numpy()
    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])
    y_train = train["slow_response"].to_numpy()
    y_test = test["slow_response"].to_numpy()

    beta = fit_logistic_gd(X_train, y_train)
    p_test = sigmoid(X_test @ beta)
    nb_test = categorical_nb_scores(train, test, feature_cols, "slow_response")
    mlp_model = fit_mlp(X_train, y_train)
    mlp_test = mlp_predict(X_test, mlp_model)

    # Baseline 1: use training prevalence only.
    base_score = np.repeat(y_train.mean(), len(y_test))
    # Baseline 2: complaint-type historical slow-rate lookup.
    type_rate = train.groupby("complaint_type")["slow_response"].mean()
    type_score = test["complaint_type"].map(type_rate).fillna(y_train.mean()).to_numpy()

    def metrics(name: str, scores: np.ndarray) -> dict:
        # Policy-relevant screening: flag the same share of requests as the observed slow-response rate.
        cutoff = float(np.quantile(scores, 1 - y_test.mean()))
        pred = (scores >= cutoff).astype(int)
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        tn = int(((pred == 0) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        return {
            "model": name,
            "train_n": int(len(y_train)),
            "test_n": int(len(y_test)),
            "slow_threshold_days": threshold,
            "test_slow_rate": float(y_test.mean()),
            "screening_cutoff": cutoff,
            "auc": auc_score(y_test, scores),
            "accuracy": float((pred == y_test).mean()),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        }

    model_results = pd.DataFrame([
        metrics("majority_baseline", base_score),
        metrics("complaint_type_lookup", type_score),
        metrics("naive_bayes", nb_test),
        metrics("logistic_regression", p_test),
        metrics("simple_neural_network", mlp_test),
    ])
    model_results.to_csv(OUT / "slow_response_model_results.csv", index=False, encoding="utf-8-sig")

    feature_names = ["intercept"] + list(train_X_df.columns)
    coef = pd.DataFrame({"feature": feature_names, "coef": beta})
    coef["abs_coef"] = coef["coef"].abs()
    coef = coef[coef["feature"] != "intercept"].sort_values("abs_coef", ascending=False)
    coef.head(25).to_csv(OUT / "slow_response_logistic_top_features.csv", index=False, encoding="utf-8-sig")

    # Risk deciles show whether predicted slow-risk is calibrated enough for screening.
    dec = pd.DataFrame({"score": p_test, "actual": y_test})
    dec["risk_decile"] = pd.qcut(dec["score"], 10, labels=False, duplicates="drop") + 1
    decile = dec.groupby("risk_decile").agg(n=("actual", "size"), predicted_mean=("score", "mean"), actual_slow_rate=("actual", "mean")).reset_index()
    decile.to_csv(OUT / "slow_response_risk_deciles.csv", index=False, encoding="utf-8-sig")

    # Operational simulation: suppose flagged requests get extra coordination and their response time falls by 20%.
    # This is not an estimated treatment effect; it is a transparent policy scenario.
    sim_rows = []
    for share in [0.10, 0.20, 0.25]:
        cutoff = float(np.quantile(p_test, 1 - share))
        flagged = p_test >= cutoff
        actual_hours = test["response_hours"].to_numpy()
        simulated_hours = actual_hours.copy()
        simulated_hours[flagged] *= 0.8
        sim_rows.append({
            "flagged_share": share,
            "flagged_n": int(flagged.sum()),
            "slow_cases_captured": int(((flagged) & (y_test == 1)).sum()),
            "slow_case_capture_rate": float(((flagged) & (y_test == 1)).sum() / max(y_test.sum(), 1)),
            "median_days_before": float(np.median(actual_hours) / 24),
            "median_days_after": float(np.median(simulated_hours) / 24),
            "p75_days_before": float(np.quantile(actual_hours, 0.75) / 24),
            "p75_days_after": float(np.quantile(simulated_hours, 0.75) / 24),
            "total_days_saved": float((actual_hours - simulated_hours).sum() / 24),
        })
    pd.DataFrame(sim_rows).to_csv(OUT / "triage_simulation.csv", index=False, encoding="utf-8-sig")
    return model_results, coef.head(25)


def mechanism_experiment(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = add_mechanism_features(df)
    d["slow_response"] = (d["response_days"] >= float(df["response_days"].quantile(0.75))).astype(int)
    d["created_month_num"] = d["created_date"].dt.month
    d["created_dow"] = d["created_date"].dt.dayofweek
    d["created_hour"] = d["created_date"].dt.hour
    train = d[d["created_date"] < "2023-10-01"].copy()
    test = d[d["created_date"] >= "2023-10-01"].copy()

    cat_cols = ["complaint_type", "borough", "created_month_num", "created_dow", "created_hour"]
    num_cols = ["agency_borough_week_volume", "median_income", "poverty_rate", "renter_rate", "hpd_violations", "hpd_class_c_share"]

    # Median impute numerical mechanism fields using training data.
    for c in num_cols:
        med = float(train[c].median()) if train[c].notna().any() else 0.0
        train[c] = train[c].fillna(med)
        test[c] = test[c].fillna(med)

    base_train = pd.get_dummies(train[cat_cols].astype(str), drop_first=False, dtype=float)
    base_test = pd.get_dummies(test[cat_cols].astype(str), drop_first=False, dtype=float).reindex(columns=base_train.columns, fill_value=0.0)
    mech_train = pd.concat([base_train.reset_index(drop=True), train[num_cols].reset_index(drop=True)], axis=1)
    mech_test = pd.concat([base_test.reset_index(drop=True), test[num_cols].reset_index(drop=True)], axis=1)

    def prep_fit(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
        means = train_df.mean()
        stds = train_df.std(ddof=0).replace(0, 1)
        Xtr = ((train_df - means) / stds).to_numpy()
        Xte = ((test_df - means) / stds).to_numpy()
        return np.column_stack([np.ones(len(Xtr)), Xtr]), np.column_stack([np.ones(len(Xte)), Xte]), ["intercept"] + list(train_df.columns)

    y_train = train["slow_response"].to_numpy()
    y_test = test["slow_response"].to_numpy()
    rows = []
    coefs = []
    for name, tr_df, te_df in [
        ("baseline_logistic", base_train, base_test),
        ("mechanism_logistic", mech_train, mech_test),
    ]:
        Xtr, Xte, names = prep_fit(tr_df, te_df)
        beta = fit_logistic_gd(Xtr, y_train, lr=0.13, epochs=650, l2=0.002)
        score = sigmoid(Xte @ beta)
        cutoff = float(np.quantile(score, 1 - y_test.mean()))
        pred = (score >= cutoff).astype(int)
        tp = int(((pred == 1) & (y_test == 1)).sum())
        fp = int(((pred == 1) & (y_test == 0)).sum())
        fn = int(((pred == 0) & (y_test == 1)).sum())
        rows.append({
            "model": name,
            "test_n": int(len(y_test)),
            "auc": auc_score(y_test, score),
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        })
        coef_df = pd.DataFrame({"model": name, "feature": names, "coef": beta})
        coefs.append(coef_df)

    res = pd.DataFrame(rows)
    coef_all = pd.concat(coefs, ignore_index=True)
    mech_coef = coef_all[(coef_all["model"] == "mechanism_logistic") & (coef_all["feature"].isin(num_cols))].copy()
    res.to_csv(OUT / "mechanism_model_results.csv", index=False, encoding="utf-8-sig")
    mech_coef.to_csv(OUT / "mechanism_logistic_coefficients.csv", index=False, encoding="utf-8-sig")

    # ZIP-level associations provide a more transparent view of social/housing mechanisms.
    zip_summary = d.groupby("incident_zip").agg(
        n=("unique_key", "size"),
        slow_rate=("slow_response", "mean"),
        median_days=("response_days", "median"),
        median_income=("median_income", "median"),
        poverty_rate=("poverty_rate", "median"),
        renter_rate=("renter_rate", "median"),
        hpd_violations=("hpd_violations", "median"),
        workload=("agency_borough_week_volume", "median"),
    ).reset_index()
    zip_summary = zip_summary[zip_summary["n"] >= 200].copy()
    corr_cols = ["median_income", "poverty_rate", "renter_rate", "hpd_violations", "workload"]
    corrs = []
    for c in corr_cols:
        sub = zip_summary[["slow_rate", c]].dropna()
        corrs.append({"variable": c, "corr_with_zip_slow_rate": float(sub["slow_rate"].corr(sub[c]))})
    pd.DataFrame(corrs).to_csv(OUT / "mechanism_zip_correlations.csv", index=False, encoding="utf-8-sig")
    return res, mech_coef


def optimization_experiment(df: pd.DataFrame) -> pd.DataFrame:
    d = add_mechanism_features(df)
    threshold = float(d["response_days"].quantile(0.75))
    d["slow_response"] = (d["response_days"] >= threshold).astype(int)
    d["created_month_num"] = d["created_date"].dt.month
    d["created_dow"] = d["created_date"].dt.dayofweek
    d["created_hour"] = d["created_date"].dt.hour
    train = d[d["created_date"] < "2023-10-01"].copy()
    score_df = d.copy()

    cat_cols = ["complaint_type", "borough", "created_month_num", "created_dow", "created_hour"]
    Xtr_df = pd.get_dummies(train[cat_cols].astype(str), drop_first=False, dtype=float)
    Xte_df = pd.get_dummies(score_df[cat_cols].astype(str), drop_first=False, dtype=float).reindex(columns=Xtr_df.columns, fill_value=0.0)
    means = Xtr_df.mean()
    stds = Xtr_df.std(ddof=0).replace(0, 1)
    Xtr = np.column_stack([np.ones(len(train)), ((Xtr_df - means) / stds).to_numpy()])
    Xte = np.column_stack([np.ones(len(score_df)), ((Xte_df - means) / stds).to_numpy()])
    beta = fit_logistic_gd(Xtr, train["slow_response"].to_numpy(), lr=0.15, epochs=700, l2=0.002)
    risk = sigmoid(Xte @ beta)

    type_median = train.groupby("complaint_type")["response_days"].median()
    expected_delay = score_df["complaint_type"].map(type_median).fillna(train["response_days"].median()).to_numpy()

    for c in ["median_income", "poverty_rate", "hpd_violations", "agency_borough_week_volume"]:
        med = float(train[c].median()) if train[c].notna().any() else 0.0
        score_df[c] = score_df[c].fillna(med)
    income_rank = score_df["median_income"].rank(pct=True, na_option="keep").fillna(0.5)
    poverty_rank = score_df["poverty_rate"].rank(pct=True, na_option="keep").fillna(0.5)
    hpd_rank = score_df["hpd_violations"].rank(pct=True, na_option="keep").fillna(0.5)
    workload_rank = score_df["agency_borough_week_volume"].rank(pct=True, na_option="keep").fillna(0.5)
    vulnerability = ((1 - income_rank) + poverty_rank + hpd_rank) / 3

    opt = pd.DataFrame({
        "unique_key": score_df["unique_key"].to_numpy(),
        "complaint_type": score_df["complaint_type"].to_numpy(),
        "borough": score_df["borough"].to_numpy(),
        "response_hours": score_df["response_hours"].to_numpy(),
        "slow_response": score_df["slow_response"].to_numpy(),
        "risk": risk,
        "expected_delay_days": expected_delay,
        "vulnerability": vulnerability.to_numpy(),
        "workload_rank": workload_rank.to_numpy(),
    })
    opt["vulnerable_area"] = opt["vulnerability"] >= opt["vulnerability"].quantile(0.75)
    rng = np.random.default_rng(42)
    opt["random_score"] = rng.random(len(opt))
    opt["type_delay_score"] = opt["expected_delay_days"]
    opt["risk_score"] = opt["risk"]
    opt["efficiency_benefit"] = opt["risk"] * opt["expected_delay_days"]
    opt["fair_weighted_benefit"] = opt["efficiency_benefit"] * (1 + opt["vulnerability"]) * (1 + 0.5 * opt["workload_rank"])

    def choose_top(score_col: str, n_flag: int) -> np.ndarray:
        chosen = opt[score_col].sort_values(ascending=False).index[:n_flag]
        return opt.index.isin(chosen)

    def solve_fair_binary_program(score_col: str, n_flag: int, min_vulnerable_share: float = 0.35) -> np.ndarray:
        # Exact solution to:
        # max sum_i score_i*x_i
        # s.t. sum_i x_i = n_flag, sum_i vulnerable_i*x_i >= min_vulnerable_share*n_flag, x_i in {0,1}.
        # Because each request has the same unit resource cost, the constrained optimum can be found
        # by enumerating how many vulnerable requests to select and taking top scores within each group.
        scores = opt[score_col].to_numpy()
        vulnerable_arr = opt["vulnerable_area"].to_numpy(dtype=bool)
        vuln_idx = np.where(vulnerable_arr)[0]
        other_idx = np.where(~vulnerable_arr)[0]
        vuln_sorted = vuln_idx[np.argsort(scores[vuln_idx])[::-1]]
        other_sorted = other_idx[np.argsort(scores[other_idx])[::-1]]
        vuln_prefix = np.concatenate([[0.0], np.cumsum(scores[vuln_sorted])])
        other_prefix = np.concatenate([[0.0], np.cumsum(scores[other_sorted])])
        min_v = int(math.ceil(min_vulnerable_share * n_flag))
        lo = max(0, n_flag - len(other_sorted), min_v)
        hi = min(n_flag, len(vuln_sorted))
        best_v = lo
        best_value = -np.inf
        for v_count in range(lo, hi + 1):
            o_count = n_flag - v_count
            value = vuln_prefix[v_count] + other_prefix[o_count]
            if value > best_value:
                best_value = value
                best_v = v_count
        chosen_pos = np.concatenate([vuln_sorted[:best_v], other_sorted[:n_flag - best_v]])
        flagged = np.zeros(len(opt), dtype=bool)
        flagged[chosen_pos] = True
        return flagged

    strategy_labels = [
        "random_priority",
        "slow_type_priority",
        "risk_priority",
        "optimal_efficiency",
        "constrained_fair_optimal",
    ]
    rows = []
    type_rows = []
    actual_hours = opt["response_hours"].to_numpy()
    y = opt["slow_response"].to_numpy()
    vulnerable = opt["vulnerable_area"].to_numpy()
    for share in [0.10, 0.20, 0.25]:
        n_flag = max(1, int(round(len(opt) * share)))
        strategy_flags = {
            "random_priority": choose_top("random_score", n_flag),
            "slow_type_priority": choose_top("type_delay_score", n_flag),
            "risk_priority": choose_top("risk_score", n_flag),
            "optimal_efficiency": choose_top("efficiency_benefit", n_flag),
            "constrained_fair_optimal": solve_fair_binary_program("fair_weighted_benefit", n_flag, min_vulnerable_share=0.35),
        }
        for strategy in strategy_labels:
            flagged = strategy_flags[strategy]
            simulated_hours = actual_hours.copy()
            simulated_hours[flagged] *= 0.8
            rows.append({
                "strategy": strategy,
                "flagged_share": share,
                "flagged_n": int(flagged.sum()),
                "slow_cases_captured": int((flagged & (y == 1)).sum()),
                "slow_case_capture_rate": float((flagged & (y == 1)).sum() / max(y.sum(), 1)),
                "vulnerable_flagged_share": float((flagged & vulnerable).sum() / max(flagged.sum(), 1)),
                "vulnerable_slow_capture_rate": float((flagged & vulnerable & (y == 1)).sum() / max((vulnerable & (y == 1)).sum(), 1)),
                "median_days_after": float(np.median(simulated_hours) / 24),
                "p75_days_after": float(np.quantile(simulated_hours, 0.75) / 24),
                "total_days_saved": float((actual_hours - simulated_hours).sum() / 24),
            })
            type_mix = opt.loc[flagged, "complaint_type"].value_counts(normalize=True).rename_axis("complaint_type").reset_index(name="selected_share")
            type_mix["strategy"] = strategy
            type_mix["flagged_share"] = share
            type_rows.append(type_mix)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "optimization_simulation.csv", index=False, encoding="utf-8-sig")
    pd.concat(type_rows, ignore_index=True).to_csv(OUT / "optimization_policy_by_type.csv", index=False, encoding="utf-8-sig")
    return res


def causal_workload_experiment(df: pd.DataFrame) -> pd.DataFrame:
    d = add_mechanism_features(df)
    threshold = float(d["response_days"].quantile(0.75))
    d["slow_response"] = (d["response_days"] >= threshold).astype(int)
    d["created_month_num"] = d["created_date"].dt.month
    d["created_dow"] = d["created_date"].dt.dayofweek
    d["created_hour"] = d["created_date"].dt.hour
    workload_cutoff = float(d["agency_borough_week_volume"].quantile(0.75))
    d["high_workload"] = (d["agency_borough_week_volume"] >= workload_cutoff).astype(int)

    cat_cols = ["complaint_type", "borough", "created_month_num", "created_dow", "created_hour"]
    num_cols = ["median_income", "poverty_rate", "renter_rate", "hpd_violations", "hpd_class_c_share"]
    for c in num_cols:
        med = float(d[c].median()) if d[c].notna().any() else 0.0
        d[c] = d[c].fillna(med)

    X_cat = pd.get_dummies(d[cat_cols].astype(str), drop_first=False, dtype=float)
    X_df = pd.concat([X_cat.reset_index(drop=True), d[num_cols].reset_index(drop=True)], axis=1)
    means = X_df.mean()
    stds = X_df.std(ddof=0).replace(0, 1)
    X = ((X_df - means) / stds).to_numpy()
    X = np.column_stack([np.ones(len(X)), X])
    T = d["high_workload"].to_numpy()
    Y = d["slow_response"].to_numpy()

    raw = float(Y[T == 1].mean() - Y[T == 0].mean())

    # Outcome regression standardization: compare predicted outcomes if every request
    # were assigned to high- vs low-workload conditions, conditional on observed controls.
    X_out = np.column_stack([X, T])
    outcome_beta = fit_logistic_gd(X_out, Y, lr=0.12, epochs=600, l2=0.002)
    m1 = sigmoid(np.column_stack([X, np.ones(len(X))]) @ outcome_beta)
    m0 = sigmoid(np.column_stack([X, np.zeros(len(X))]) @ outcome_beta)
    regression_adjusted = float(np.mean(m1 - m0))

    # Propensity and doubly robust estimates are included as sensitivity checks,
    # not as proof of randomized causal identification.
    prop_beta = fit_logistic_gd(X, T, lr=0.12, epochs=600, l2=0.002)
    e = np.clip(sigmoid(X @ prop_beta), 0.03, 0.97)
    ipw = float(np.mean(T * Y / e - (1 - T) * Y / (1 - e)))
    aipw = float(np.mean(m1 - m0 + T / e * (Y - m1) - (1 - T) / (1 - e) * (Y - m0)))

    rows = [
        {"estimand": "raw_difference", "treated": "agency_borough_week_volume_top_quartile", "outcome": "slow_response_top_quartile", "estimate_pp": raw * 100, "note": "Unadjusted difference; descriptive only."},
        {"estimand": "regression_standardization", "treated": "agency_borough_week_volume_top_quartile", "outcome": "slow_response_top_quartile", "estimate_pp": regression_adjusted * 100, "note": "Adjusted for complaint type, borough, time and ZIP-level ACS/HPD proxies."},
        {"estimand": "ipw", "treated": "agency_borough_week_volume_top_quartile", "outcome": "slow_response_top_quartile", "estimate_pp": ipw * 100, "note": "Observational propensity-weighted sensitivity estimate."},
        {"estimand": "aipw", "treated": "agency_borough_week_volume_top_quartile", "outcome": "slow_response_top_quartile", "estimate_pp": aipw * 100, "note": "Doubly robust observational sensitivity estimate; not randomized causal proof."},
    ]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "causal_workload_effect.csv", index=False, encoding="utf-8-sig")
    return res


def heterogeneity_panel_experiment(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = add_mechanism_features(df)
    threshold = float(d["response_days"].quantile(0.75))
    d["slow_response"] = (d["response_days"] >= threshold).astype(int)
    d["month_period"] = d["created_date"].dt.to_period("M").astype(str)
    for c in ["median_income", "poverty_rate", "hpd_violations", "agency_borough_week_volume"]:
        med = float(d[c].median()) if d[c].notna().any() else 0.0
        d[c] = d[c].fillna(med)

    income_q25 = float(d["median_income"].quantile(0.25))
    hpd_q75 = float(d["hpd_violations"].quantile(0.75))
    d["low_income_zip"] = d["median_income"] <= income_q25
    d["high_hpd_zip"] = d["hpd_violations"] >= hpd_q75

    rows = []
    for group_col, label in [("low_income_zip", "low_income"), ("high_hpd_zip", "high_hpd")]:
        tab = d.groupby(["complaint_type", group_col]).agg(
            n=("unique_key", "size"),
            slow_rate=("slow_response", "mean"),
            median_days=("response_days", "median"),
        ).reset_index()
        for comp in sorted(tab["complaint_type"].unique()):
            sub = tab[tab["complaint_type"] == comp].set_index(group_col)
            if True in sub.index and False in sub.index:
                rows.append({
                    "heterogeneity": label,
                    "complaint_type": comp,
                    "n_group_true": int(sub.loc[True, "n"]),
                    "n_group_false": int(sub.loc[False, "n"]),
                    "slow_rate_true": float(sub.loc[True, "slow_rate"]),
                    "slow_rate_false": float(sub.loc[False, "slow_rate"]),
                    "slow_rate_gap_pp": float((sub.loc[True, "slow_rate"] - sub.loc[False, "slow_rate"]) * 100),
                    "median_days_true": float(sub.loc[True, "median_days"]),
                    "median_days_false": float(sub.loc[False, "median_days"]),
                })
    hetero = pd.DataFrame(rows)
    hetero.to_csv(OUT / "heterogeneity_by_type.csv", index=False, encoding="utf-8-sig")

    panel = d.groupby(["incident_zip", "month_period", "complaint_type"]).agg(
        n=("unique_key", "size"),
        slow_rate=("slow_response", "mean"),
        median_days=("response_days", "median"),
        workload=("agency_borough_week_volume", "median"),
    ).reset_index()
    panel = panel[(panel["n"] >= 20) & panel["incident_zip"].notna()].copy()
    panel["log_workload"] = np.log1p(panel["workload"])
    y = panel["slow_rate"].to_numpy()
    x_raw = panel["log_workload"].to_numpy()
    x_std = (x_raw - x_raw.mean()) / (x_raw.std() or 1)
    X_df = pd.concat([
        pd.Series(x_std, name="log_workload_std").reset_index(drop=True),
        pd.get_dummies(panel["incident_zip"].astype(str), prefix="zip", drop_first=True, dtype=float).reset_index(drop=True),
        pd.get_dummies(panel["month_period"].astype(str), prefix="month", drop_first=True, dtype=float).reset_index(drop=True),
        pd.get_dummies(panel["complaint_type"].astype(str), prefix="type", drop_first=True, dtype=float).reset_index(drop=True),
    ], axis=1)
    X = np.column_stack([np.ones(len(X_df)), X_df.to_numpy()])
    w = np.sqrt(panel["n"].to_numpy())
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    panel_res = pd.DataFrame([{
        "model": "zip_month_type_fixed_effects",
        "outcome": "zip_month_type_slow_rate",
        "coef_log_workload_std_pp": float(beta[1] * 100),
        "n_cells": int(len(panel)),
        "weighted_rmse_pp": float(np.sqrt(np.average(resid ** 2, weights=panel["n"])) * 100),
        "note": "Weighted panel FE association; controls ZIP, month and complaint-type fixed effects, not a causal design.",
    }])
    panel_res.to_csv(OUT / "panel_fixed_effects.csv", index=False, encoding="utf-8-sig")
    return hetero, panel_res


def time_series_experiment(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["week"] = d["created_date"].dt.to_period("W-MON").dt.start_time
    weekly = d.groupby("week")["response_days"].median().reset_index(name="median_days").sort_values("week")
    train = weekly.iloc[:-13].copy()
    test = weekly.iloc[-13:].copy()
    y_train = train["median_days"].to_numpy()
    y_test = test["median_days"].to_numpy()
    naive = np.repeat(y_train[-1], len(y_test))
    ma4 = []
    history = list(y_train)
    for _ in range(len(y_test)):
        pred = float(np.mean(history[-4:]))
        ma4.append(pred)
        history.append(pred)
    # AR(1): y_t = a + b*y_{t-1}
    X = np.column_stack([np.ones(len(y_train) - 1), y_train[:-1]])
    beta, *_ = np.linalg.lstsq(X, y_train[1:], rcond=None)
    ar = []
    prev = y_train[-1]
    for _ in range(len(y_test)):
        pred = float(beta[0] + beta[1] * prev)
        ar.append(pred)
        prev = pred
    rows = []
    for name, pred in [("last_observation", naive), ("moving_average_4w", np.array(ma4)), ("ar1", np.array(ar))]:
        err = pred - y_test
        rows.append({
            "model": name,
            "test_weeks": int(len(y_test)),
            "mae_days": float(np.mean(np.abs(err))),
            "rmse_days": float(np.sqrt(np.mean(err ** 2))),
        })
    res = pd.DataFrame(rows)
    weekly.to_csv(OUT / "weekly_median_response.csv", index=False, encoding="utf-8-sig")
    res.to_csv(OUT / "time_series_forecast_results.csv", index=False, encoding="utf-8-sig")
    return res


def aipw_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    threshold = float(d["response_days"].quantile(0.75))
    d["slow_response"] = (d["response_days"] >= threshold).astype(int)
    d["treated_qb"] = d["borough"].isin(["QUEENS", "BROOKLYN"]).astype(int)
    d["created_month_num"] = d["created_date"].dt.month.astype(str)
    d["created_dow"] = d["created_date"].dt.dayofweek.astype(str)
    d["created_hour"] = d["created_date"].dt.hour.astype(str)
    controls = ["complaint_type", "created_month_num", "created_dow", "created_hour"]
    Xdf = pd.get_dummies(d[controls].astype(str), drop_first=False, dtype=float)
    means = Xdf.mean()
    stds = Xdf.std(ddof=0).replace(0, 1)
    X = ((Xdf - means) / stds).to_numpy()
    X = np.column_stack([np.ones(len(X)), X])

    T = d["treated_qb"].to_numpy()
    Y = d["slow_response"].to_numpy()
    e_beta = fit_logistic_gd(X, T, lr=0.12, epochs=500, l2=0.002)
    e = np.clip(sigmoid(X @ e_beta), 0.03, 0.97)

    X_out = np.column_stack([X, T])
    m_beta = fit_logistic_gd(X_out, Y, lr=0.12, epochs=500, l2=0.002)
    m1 = sigmoid(np.column_stack([X, np.ones(len(X))]) @ m_beta)
    m0 = sigmoid(np.column_stack([X, np.zeros(len(X))]) @ m_beta)
    aipw = np.mean(m1 - m0 + T / e * (Y - m1) - (1 - T) / (1 - e) * (Y - m0))
    raw = Y[T == 1].mean() - Y[T == 0].mean()
    out = pd.DataFrame([{
        "contrast": "Queens_or_Brooklyn_vs_other_boroughs",
        "outcome": "slow_response_top_quartile",
        "raw_difference_pp": float(raw * 100),
        "aipw_adjusted_difference_pp": float(aipw * 100),
        "treated_n": int(T.sum()),
        "control_n": int((1 - T).sum()),
        "note": "Observational adjustment; not a randomized causal estimate.",
    }])
    out.to_csv(OUT / "aipw_borough_adjustment.csv", index=False, encoding="utf-8-sig")
    return out


def svg_wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(str(text), width=width, break_long_words=False) or [""]


def save_bar_svg(path: Path, data: pd.DataFrame, label_col: str, value_col: str, title: str, subtitle: str, unit: str = "", color: str = BLUE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = data.copy()
    w, h = 920, 520
    left, right, top, bottom = 230, 70, 105, 55
    plot_w, plot_h = w - left - right, h - top - bottom
    maxv = float(data[value_col].max() or 1)
    row_h = plot_h / len(data)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="{SURFACE}"/><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="{PANEL}"/>')
    parts.append(f'<text x="{left}" y="32" font-size="20" font-weight="700" fill="{INK}">{title}</text>')
    parts.append(f'<text x="{left}" y="58" font-size="13" fill="{MUTED}">{subtitle}</text>')
    for gx in np.linspace(0, maxv, 5):
        x = left + gx / maxv * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+24}" font-size="11" text-anchor="middle" fill="{MUTED}">{gx:.0f}</text>')
    for i, row in data.reset_index(drop=True).iterrows():
        y = top + i * row_h + row_h * 0.18
        bar_h = row_h * 0.58
        val = float(row[value_col])
        bw = val / maxv * plot_w
        label = str(row[label_col])
        for j, line in enumerate(svg_wrap(label, 26)[:2]):
            parts.append(f'<text x="{left-12}" y="{y+bar_h/2-4+j*13:.1f}" font-size="12" text-anchor="end" fill="{INK}">{line}</text>')
        parts.append(f'<rect x="{left}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" fill="{color}" stroke="#2E4780" stroke-width="1"/>')
        parts.append(f'<text x="{left+bw+7:.1f}" y="{y+bar_h/2+4:.1f}" font-size="12" fill="{INK}">{val:.1f}{unit}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_grouped_svg(path: Path, data: pd.DataFrame, title: str, subtitle: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complaints = list(data["complaint_type"].unique())
    boroughs = list(data["borough"].unique())
    colors = [BLUE, ORANGE, OLIVE, PINK, GOLD]
    w, h = 1040, 560
    left, right, top, bottom = 150, 70, 110, 90
    plot_w, plot_h = w - left - right, h - top - bottom
    maxv = float(data["median_days"].max() or 1)
    group_w = plot_w / len(complaints)
    bar_w = group_w / (len(boroughs) + 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="{SURFACE}"/><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="{PANEL}"/>')
    parts.append(f'<text x="{left}" y="32" font-size="20" font-weight="700" fill="{INK}">{title}</text>')
    parts.append(f'<text x="{left}" y="58" font-size="13" fill="{MUTED}">{subtitle}</text>')
    for gy in np.linspace(0, maxv, 5):
        y = top + plot_h - gy / maxv * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{gy:.0f}</text>')
    for i, b in enumerate(boroughs):
        lx = left + i * 145
        parts.append(f'<rect x="{lx}" y="78" width="12" height="12" fill="{colors[i % len(colors)]}"/>')
        parts.append(f'<text x="{lx+18}" y="89" font-size="12" fill="{INK}">{b}</text>')
    for gi, comp in enumerate(complaints):
        subset = data[data["complaint_type"] == comp].set_index("borough")
        gx = left + gi * group_w
        for bi, b in enumerate(boroughs):
            val = float(subset.loc[b, "median_days"]) if b in subset.index else 0
            bh = val / maxv * plot_h
            x = gx + bi * bar_w + bar_w * 0.25
            y = top + plot_h - bh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*0.72:.1f}" height="{bh:.1f}" fill="{colors[bi % len(colors)]}" stroke="{INK}" stroke-width="0.5"/>')
        for j, line in enumerate(svg_wrap(comp, 14)[:2]):
            parts.append(f'<text x="{gx+group_w/2:.1f}" y="{top+plot_h+24+j*13}" font-size="11" text-anchor="middle" fill="{INK}">{line}</text>')
    parts.append(f'<text x="45" y="{top+plot_h/2}" transform="rotate(-90 45 {top+plot_h/2})" font-size="12" text-anchor="middle" fill="{MUTED}">Median response time, days</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_scatter_svg(path: Path, data: pd.DataFrame, title: str, subtitle: str) -> None:
    w, h = 920, 560
    left, right, top, bottom = 95, 80, 105, 70
    plot_w, plot_h = w - left - right, h - top - bottom
    xcol, ycol = "n", "slow_index"
    xmin, xmax = 0, float(data[xcol].max() * 1.08)
    ymin, ymax = max(0, float(data[ycol].min() * 0.9)), float(data[ycol].max() * 1.12)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="{SURFACE}"/><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="{PANEL}"/>')
    parts.append(f'<text x="{left}" y="32" font-size="20" font-weight="700" fill="{INK}">{title}</text>')
    parts.append(f'<text x="{left}" y="58" font-size="13" fill="{MUTED}">{subtitle}</text>')
    for gx in np.linspace(xmin, xmax, 5):
        x = left + (gx - xmin) / (xmax - xmin) * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+25}" font-size="11" text-anchor="middle" fill="{MUTED}">{gx:.0f}</text>')
    for gy in np.linspace(ymin, ymax, 5):
        y = top + plot_h - (gy - ymin) / (ymax - ymin) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" font-size="11" text-anchor="end" fill="{MUTED}">{gy:.1f}</text>')
    top_labels = set(data.sort_values(ycol, ascending=False).head(5)["community_board"])
    for _, r in data.iterrows():
        x = left + (r[xcol] - xmin) / (xmax - xmin) * plot_w
        y = top + plot_h - (r[ycol] - ymin) / (ymax - ymin) * plot_h
        color = ORANGE if r["community_board"] in top_labels else BLUE_LIGHT
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="{INK}" stroke-width="0.7" opacity="0.85"/>')
        if r["community_board"] in top_labels:
            label = str(r["community_board"]).replace(" Community Board", "")
            parts.append(f'<text x="{x+8:.1f}" y="{y-8:.1f}" font-size="11" fill="{INK}">{label}</text>')
    parts.append(f'<text x="{left+plot_w/2}" y="{h-20}" font-size="12" text-anchor="middle" fill="{MUTED}">Number of 311 requests in selected categories</text>')
    parts.append(f'<text x="30" y="{top+plot_h/2}" transform="rotate(-90 30 {top+plot_h/2})" font-size="12" text-anchor="middle" fill="{MUTED}">Slow response index</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_figures(df: pd.DataFrame, board: pd.DataFrame, borough_adj: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    mix = df["complaint_type"].value_counts().rename_axis("complaint_type").reset_index(name="n")
    save_bar_svg(FIG / "fig1_complaint_mix.svg", mix.sort_values("n"), "complaint_type", "n", "Selected 311 requests are dominated by daily-life problems", "NYC 311 closed service requests, selected complaint types, 2023", "")

    med_type = df.groupby("complaint_type")["response_days"].median().sort_values().rename_axis("complaint_type").reset_index(name="median_days")
    save_bar_svg(FIG / "fig2_median_by_type.svg", med_type, "complaint_type", "median_days", "Response time differs sharply by complaint type", "Median closure time in days; selected NYC 311 complaints, 2023", "d", ORANGE)

    med_borough = df.groupby("borough")["response_days"].median().sort_values().rename_axis("borough").reset_index(name="median_days")
    save_bar_svg(FIG / "fig3_median_by_borough.svg", med_borough, "borough", "median_days", "Borough-level differences remain visible", "Median closure time in days across selected complaint types, 2023", "d", OLIVE)

    group = df.groupby(["complaint_type", "borough"])["response_days"].median().reset_index(name="median_days")
    # Keep stable ordering by citywide median.
    order = med_type["complaint_type"].tolist()
    group["complaint_type"] = pd.Categorical(group["complaint_type"], categories=order, ordered=True)
    group = group.sort_values(["complaint_type", "borough"])
    save_grouped_svg(FIG / "fig4_type_borough_grouped.svg", group, "The same borough gap is not the same for every problem", "Median response time by complaint type and borough, days")

    save_scatter_svg(FIG / "fig5_community_slow_index.svg", board, "Some community boards wait longer than expected", "Slow index adjusts each request by the citywide median for its complaint type; boards with n>=200")

    monthly = df.groupby("month")["response_days"].median().reset_index()
    save_bar_svg(FIG / "fig6_monthly_median.svg", monthly, "month", "response_days", "Response time varies across the year", "Monthly median closure time in days; selected NYC 311 complaints, 2023", "d", PINK)


def make_model_figures(model_results: pd.DataFrame, top_features: pd.DataFrame, ts_results: pd.DataFrame) -> None:
    main_models = ["majority_baseline", "complaint_type_lookup", "naive_bayes", "logistic_regression"]
    model_plot = model_results[model_results["model"].isin(main_models)][["model", "auc"]].copy().sort_values("auc")
    save_bar_svg(FIG / "fig7_model_auc.svg", model_plot, "model", "auc", "Logistic regression best separates slow-response requests", "AUC on Oct-Dec 2023 holdout set; slow response = top quartile closure time", "", BLUE)

    feat = top_features.head(10).copy()
    feat["importance"] = feat["coef"].abs()
    save_bar_svg(FIG / "fig8_top_features.svg", feat.sort_values("importance"), "feature", "importance", "Complaint type dominates slow-response prediction", "Absolute logistic coefficient; larger values indicate stronger association", "", ORANGE)

    ts_plot = ts_results[["model", "mae_days"]].copy().sort_values("mae_days", ascending=False)
    save_bar_svg(FIG / "fig9_timeseries_mae.svg", ts_plot, "model", "mae_days", "Simple time-series baselines forecast weekly response better than AR(1)", "Mean absolute error in days for the last 13 weeks of 2023", "d", OLIVE)


def save_signed_bar_svg(path: Path, data: pd.DataFrame, label_col: str, value_col: str, title: str, subtitle: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = data.copy()
    w, h = 920, 500
    left, right, top, bottom = 240, 80, 105, 55
    plot_w, plot_h = w - left - right, h - top - bottom
    max_abs = float(data[value_col].abs().max() or 1)
    zero_x = left + plot_w / 2
    row_h = plot_h / len(data)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    parts.append(f'<rect width="{w}" height="{h}" fill="{SURFACE}"/><rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="{PANEL}"/>')
    parts.append(f'<text x="{left}" y="32" font-size="20" font-weight="700" fill="{INK}">{title}</text>')
    parts.append(f'<text x="{left}" y="58" font-size="13" fill="{MUTED}">{subtitle}</text>')
    for tick in np.linspace(-max_abs, max_abs, 5):
        x = zero_x + tick / max_abs * plot_w / 2
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+24}" font-size="11" text-anchor="middle" fill="{MUTED}">{tick:.2f}</text>')
    parts.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top+plot_h}" stroke="{INK}" stroke-width="1.3"/>')
    for i, row in data.reset_index(drop=True).iterrows():
        y = top + i * row_h + row_h * 0.2
        bar_h = row_h * 0.55
        val = float(row[value_col])
        bw = abs(val) / max_abs * plot_w / 2
        x = zero_x if val >= 0 else zero_x - bw
        color = ORANGE if val >= 0 else BLUE
        for j, line in enumerate(svg_wrap(str(row[label_col]), 28)[:2]):
            parts.append(f'<text x="{left-12}" y="{y+bar_h/2-4+j*13:.1f}" font-size="12" text-anchor="end" fill="{INK}">{line}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" fill="{color}" stroke="#555" stroke-width="0.8"/>')
        tx = x + bw + 7 if val >= 0 else x - 7
        anchor = "start" if val >= 0 else "end"
        parts.append(f'<text x="{tx:.1f}" y="{y+bar_h/2+4:.1f}" font-size="12" text-anchor="{anchor}" fill="{INK}">{val:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def make_mechanism_figures(mech_results: pd.DataFrame, mech_corr: pd.DataFrame) -> None:
    auc = mech_results[["model", "auc"]].copy()
    auc["auc_percent"] = auc["auc"] * 100
    save_bar_svg(FIG / "fig10_mechanism_auc.svg", auc, "model", "auc_percent", "Mechanism variables barely improve individual prediction", "AUC comparison: baseline categorical model vs model adding workload, ACS and HPD proxies", "%", GOLD)
    label_map = {
        "median_income": "Median household income",
        "poverty_rate": "Poverty rate",
        "renter_rate": "Renter-occupied share",
        "hpd_violations": "HPD housing violations",
        "workload": "Agency-borough-week workload",
    }
    corr = mech_corr.copy()
    corr["label"] = corr["variable"].map(label_map).fillna(corr["variable"])
    corr = corr.sort_values("corr_with_zip_slow_rate")
    save_signed_bar_svg(FIG / "fig11_mechanism_correlations.svg", corr, "label", "corr_with_zip_slow_rate", "Community conditions align with ZIP-level slow-response risk", "Correlation with ZIP slow-response rate; positive means slower areas tend to have more of that attribute")


def make_optimization_figures(opt_results: pd.DataFrame) -> None:
    plot = opt_results[opt_results["flagged_share"] == 0.25][["strategy", "total_days_saved"]].copy()
    plot = plot.sort_values("total_days_saved")
    save_bar_svg(FIG / "fig12_optimization_days_saved.svg", plot, "strategy", "total_days_saved", "Optimization turns prediction into resource allocation", "Simulated total waiting days saved when 25% of requests receive 20% faster handling", "d", PINK)


def write_report(df: pd.DataFrame, board: pd.DataFrame, borough_adj: pd.DataFrame) -> None:
    type_summary = df.groupby("complaint_type").agg(n=("unique_key", "size"), median_days=("response_days", "median"), p75_days=("response_days", lambda s: s.quantile(0.75))).reset_index().sort_values("median_days")
    borough_summary = df.groupby("borough").agg(n=("unique_key", "size"), median_days=("response_days", "median"), p75_days=("response_days", lambda s: s.quantile(0.75))).reset_index().sort_values("median_days")
    top_boards = board.head(8)
    summary = {
        "rows": int(len(df)),
        "complaint_types": int(df["complaint_type"].nunique()),
        "boroughs": int(df["borough"].nunique()),
        "median_days": float(df["response_days"].median()),
        "p75_days": float(df["response_days"].quantile(0.75)),
        "slowest_type": str(type_summary.iloc[-1]["complaint_type"]),
        "slowest_type_median_days": float(type_summary.iloc[-1]["median_days"]),
        "fastest_type": str(type_summary.iloc[0]["complaint_type"]),
        "fastest_type_median_days": float(type_summary.iloc[0]["median_days"]),
        "slowest_borough": str(borough_summary.iloc[-1]["borough"]),
        "slowest_borough_median_days": float(borough_summary.iloc[-1]["median_days"]),
        "fastest_borough": str(borough_summary.iloc[0]["borough"]),
        "fastest_borough_median_days": float(borough_summary.iloc[0]["median_days"]),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    type_summary.to_csv(OUT / "complaint_type_summary.csv", index=False, encoding="utf-8-sig")
    borough_summary.to_csv(OUT / "borough_summary.csv", index=False, encoding="utf-8-sig")

    model_results, top_features = ml_slow_response_experiment(df)
    ts_results = time_series_experiment(df)
    aipw_results = aipw_adjustment(df)
    mech_results, mech_coef = mechanism_experiment(df)
    opt_results = optimization_experiment(df)
    causal_results = causal_workload_experiment(df)
    hetero_results, panel_results = heterogeneity_panel_experiment(df)
    make_model_figures(model_results, top_features, ts_results)
    triage = pd.read_csv(OUT / "triage_simulation.csv")
    mech_corr = pd.read_csv(OUT / "mechanism_zip_correlations.csv")
    make_mechanism_figures(mech_results, mech_corr)
    make_optimization_figures(opt_results)
    md = f"""# NYC 311 Response Time Analysis

## Key Metrics

- Observations: {summary['rows']:,} closed 311 requests.
- Complaint types: {summary['complaint_types']}; boroughs: {summary['boroughs']}.
- Overall median closure time: {summary['median_days']:.2f} days; 75th percentile: {summary['p75_days']:.2f} days.
- Fastest complaint type: {summary['fastest_type']} ({summary['fastest_type_median_days']:.2f} days median).
- Slowest complaint type: {summary['slowest_type']} ({summary['slowest_type_median_days']:.2f} days median).
- Fastest borough: {summary['fastest_borough']} ({summary['fastest_borough_median_days']:.2f} days median).
- Slowest borough: {summary['slowest_borough']} ({summary['slowest_borough_median_days']:.2f} days median).

## Complaint Type Summary

{type_summary.to_string(index=False)}

## Borough Summary

{borough_summary.to_string(index=False)}

## Slowest Community Boards, adjusted by complaint type

{top_boards[['borough','community_board','n','median_days','slow_index']].to_string(index=False)}

## Borough regression adjustment

{borough_adj.sort_values('adjusted_percent_vs_average', ascending=False).to_string(index=False)}

## Slow-response prediction experiment

{model_results.to_string(index=False)}

## Top logistic features

{top_features[['feature','coef']].to_string(index=False)}

## Time-series forecast experiment

{ts_results.to_string(index=False)}

## AIPW-style adjustment

{aipw_results.to_string(index=False)}

## Triage simulation

{triage.to_string(index=False)}

## Constrained optimization simulation

{opt_results.to_string(index=False)}

## Observational causal sensitivity: high workload

{causal_results.to_string(index=False)}

## Heterogeneity by complaint type

{hetero_results.to_string(index=False)}

## Panel fixed-effects robustness

{panel_results.to_string(index=False)}

## Mechanism model comparison

{mech_results.to_string(index=False)}

## Mechanism coefficients

{mech_coef.to_string(index=False)}

## ZIP-level mechanism correlations

{mech_corr.to_string(index=False)}
"""
    (OUT / "analysis_results.md").write_text(md, encoding="utf-8")
    print(md)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    board = adjusted_community_index(df)
    borough_adj = regression_adjustment(df)
    make_figures(df, board, borough_adj)
    write_report(df, board, borough_adj)


if __name__ == "__main__":
    main()
