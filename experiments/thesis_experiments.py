#!/usr/bin/env python3
"""
Thesis experiments (Chapter 8) on the real dataset.

Reuses the engine's own feature code and network so the numbers describe the
system that actually runs. Writes everything to experiments/results/.

    python experiments/thesis_experiments.py            # full run (needs torch)
    python experiments/thesis_experiments.py --no-torch # sklearn parts only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import youtube_virality_engine as E  # noqa: E402

OUT = ROOT / "experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
FOLDS, REPEATS = 5, 3

V1 = list(E.FEATURE_COLUMNS_V1)
V2 = list(E.FEATURE_COLUMNS)
H0 = ["log_views_per_minute_0h", "likes_per_view_0h", "comments_per_view_0h",
      "title_length", "tags_count", "log_views_per_sub_0h", "log_subscribers", "log_duration"]


# ----------------------------------------------------------------------------- data
def num(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def load():
    raw = E.ExcelStore(E.CFG.excel_path).load()
    raw = E.relabel_existing(raw)
    labels = pd.to_numeric(raw[E.TARGET_COLUMN], errors="coerce")
    eligible = labels.isin([0, 1]) & (raw["stage_8h_status"].astype(str) == E.STATUS_OK) \
        & num(raw, "views_per_minute").notna()
    df = E.add_derived_features(raw.loc[eligible])
    for c in V1:
        df[c] = num(df, c)
    subs = num(df, "subscribers").where(lambda x: x > 0)
    v0, a0 = num(df, "views_0h"), num(df, "age_minutes_0h").where(lambda x: x > 0)
    df["log_views_per_minute_0h"] = np.log1p((v0 / a0).clip(lower=0))
    df["likes_per_view_0h"] = num(df, "likes_0h") / v0.where(v0 > 0)
    df["comments_per_view_0h"] = num(df, "comments_0h") / v0.where(v0 > 0)
    df["log_views_per_sub_0h"] = np.log1p((v0 / subs).clip(lower=0))
    # Acceleration over NON-overlapping windows (ablation, Ch. 8)
    v8, a8 = num(df, "views_8h"), num(df, "age_minutes_8h")
    interval = ((v8 - v0).clip(lower=0) / (a8 - a0).where(lambda x: x > 0))
    df["log_acceleration_interval"] = np.log((interval + 1e-3) / ((v0 / a0).clip(lower=0) + 1e-3))
    df["y"] = labels.loc[eligible].astype(int)
    return raw, df


def matrix(df, cols):
    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    return X.to_numpy()


def impute_scale(Xtr, Xte):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xte = np.where(np.isnan(Xte), med, Xte)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


# --------------------------------------------------------------------------- models
def fit_logreg(Xtr, ytr, Xte):
    Xtr, Xte = impute_scale(Xtr, Xte)
    m = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m


def fit_mlp(Xtr, ytr, Xte, seed=SEED):
    """Same network, loss, optimiser and early stopping as the engine (Ch. 6)."""
    import torch
    import torch.nn as nn
    E.set_seeds(seed)
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=E.CFG.validation_fraction,
                                      random_state=seed, stratify=ytr)
    med = np.nanmedian(Xa, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    fill = lambda X: np.where(np.isnan(X), med, X)
    sc = StandardScaler().fit(fill(Xa))
    T = lambda X: torch.tensor(sc.transform(fill(X)), dtype=torch.float32)
    Xa_t, Xv_t, Xte_t = T(Xa), T(Xv), T(Xte)
    ya_t, yv_t = torch.tensor(ya, dtype=torch.float32), torch.tensor(yv, dtype=torch.float32)
    model = E.build_network(Xa.shape[1], E.CFG.hidden_1, E.CFG.hidden_2, E.CFG.dropout)
    pw = torch.tensor([max(len(ya) - ya.sum(), 1) / max(ya.sum(), 1)], dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(model.parameters(), lr=E.CFG.learning_rate, weight_decay=E.CFG.weight_decay)
    best, best_state, patience, epochs_run = float("inf"), None, E.CFG.early_stopping_patience, 0
    bs = min(E.CFG.batch_size, max(2, len(Xa_t) // 2))
    for epoch in range(1, E.CFG.epochs + 1):
        epochs_run = epoch
        model.train()
        perm = torch.randperm(len(Xa_t))
        for s in range(0, len(Xa_t), bs):
            idx = perm[s:s + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = crit(model(Xa_t[idx]), ya_t[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv_t), yv_t).item()
        if vl < best - 1e-5:
            best, patience = vl, E.CFG.early_stopping_patience
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte_t)).numpy()
    return p, epochs_run


def metrics(y, p, thr=0.5):
    pr = (p >= thr).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "f1": round(float(f1_score(y, pr, zero_division=0)), 4),
        "precision": round(float(precision_score(y, pr, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pr, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, pr)), 4),
    }


def cross_validate(df, cols, kind):
    """Repeated stratified 5-fold CV. Returns pooled metrics (mean over repeats) + per-fold PR-AUC."""
    X, y = matrix(df, cols), df["y"].to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
    oof = np.zeros((REPEATS, len(y)))
    fold_pr, epochs = [], []
    for k, (tr, te) in enumerate(rskf.split(X, y)):
        r = k // FOLDS
        if kind == "logreg":
            p, _ = fit_logreg(X[tr], y[tr], X[te])
        elif kind == "mlp":
            p, ep = fit_mlp(X[tr], y[tr], X[te], seed=SEED + k)
            epochs.append(ep)
        elif kind == "majority":
            p = np.full(len(te), y[tr].mean())
        oof[r, te] = p
        fold_pr.append(average_precision_score(y[te], p))
    per_repeat = [metrics(y, oof[r]) for r in range(REPEATS)]
    out = {k: round(float(np.mean([m[k] for m in per_repeat])), 4) for k in per_repeat[0]}
    out["pr_auc_fold_std"] = round(float(np.std(fold_pr)), 4)
    if epochs:
        out["mean_epochs"] = round(float(np.mean(epochs)), 1)
    return out, oof.mean(axis=0)


# ---------------------------------------------------------------------- experiments
def dataset_stats(raw, df):
    st = {}
    st["rows_total"] = int(len(raw))
    st["row_status"] = raw["row_status"].astype(str).value_counts().to_dict()
    for c in ("stage_8h_status", "stage_24h_status", "stage_72h_status"):
        st[c] = raw[c].astype(str).value_counts().to_dict()
    lab = raw[raw["row_status"].astype(str) == "LABELLED"]
    y = pd.to_numeric(lab["is_viral"], errors="coerce")
    st["labelled"] = int(len(lab))
    st["viral"] = int(y.sum())
    st["viral_rate"] = round(float(y.mean()), 4)
    st["training_rows"] = int(len(df))
    st["training_viral"] = int(df["y"].sum())
    disc = pd.to_datetime(raw["discovered_at"], errors="coerce", utc=True)
    st["first_discovered"] = str(disc.min())
    st["last_discovered"] = str(disc.max())
    daily = disc.dt.strftime("%Y-%m-%d").value_counts().sort_index()
    daily.to_csv(OUT / "ingested_per_day.csv", header=["videos"])
    # MISSED rate before / after the cron-job.org fix (2026-10-29)
    cut = pd.Timestamp("2026-10-29T12:00:00Z")
    for name, part in (("before_2026_10_29", raw[disc < cut]), ("after_2026_10_29", raw[disc >= cut])):
        s8 = part["stage_8h_status"].astype(str)
        s72 = part["stage_72h_status"].astype(str)
        done8 = s8.isin(["OK", "MISSED"])
        done72 = s72.isin(["OK", "MISSED"])
        st[f"missed_8h_rate_{name}"] = round(float((s8 == "MISSED").sum() / max(done8.sum(), 1)), 4)
        st[f"missed_72h_rate_{name}"] = round(float((s72 == "MISSED").sum() / max(done72.sum(), 1)), 4)
        st[f"videos_{name}"] = int(len(part))
    d = df
    st["subscribers_median"] = float(num(d, "subscribers").median())
    st["views_8h_median"] = float(num(d, "views_8h").median())
    st["final_views_median"] = float(num(d, "final_views").median())
    st["growth_72h_over_8h_median"] = round(float((num(d, "final_views") / num(d, "views_8h")).median()), 3)
    st["category_top"] = raw["category_id"].astype(str).value_counts().head(8).to_dict()
    viral_by_cat = d.groupby(d["category_id"].astype(str))["y"].agg(["count", "mean"])
    viral_by_cat[viral_by_cat["count"] >= 30].round(3).to_csv(OUT / "viral_rate_by_category.csv")
    return st


def label_sensitivity(raw):
    lab = raw[raw["row_status"].astype(str) == "LABELLED"]
    subs, views, eng = num(lab, "subscribers"), num(lab, "final_views"), num(lab, "engagement_ratio")
    rows = []
    for mult in (2, 3, 5):
        for thr in (0.0, 0.003, 0.01, 0.02):
            v = ((subs > 0) & (views >= mult * subs) & (eng >= thr)).sum()
            rows.append({"multiplier": mult, "engagement_threshold": thr,
                         "viral": int(v), "rate": round(float(v / len(lab)), 4)})
    pd.DataFrame(rows).to_csv(OUT / "label_sensitivity.csv", index=False)
    return rows


def live_evaluation(raw):
    lab = raw[raw["row_status"].astype(str) == "LABELLED"].copy()
    lab["p"] = num(lab, "pred_prob")
    lab = lab[lab["p"].notna()]
    y, p = pd.to_numeric(lab["is_viral"]).astype(int).to_numpy(), lab["p"].to_numpy()
    res = {"n": int(len(y)), "viral": int(y.sum()), "base_rate": round(float(y.mean()), 4), **metrics(y, p)}
    pr = (p >= 0.5).astype(int)
    res["confusion"] = {"tp": int(((pr == 1) & (y == 1)).sum()), "fp": int(((pr == 1) & (y == 0)).sum()),
                        "fn": int(((pr == 0) & (y == 1)).sum()), "tn": int(((pr == 0) & (y == 0)).sum())}
    res["thresholds"] = [{"threshold": t, **{k: v for k, v in metrics(y, p, t).items()
                                             if k in ("precision", "recall", "f1")}}
                         for t in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)]
    bins = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001])
    idx = np.digitize(p, bins) - 1
    res["calibration"] = [{"bin": f"{bins[i]:.1f}-{min(bins[i + 1], 1):.1f}", "n": int((idx == i).sum()),
                           "mean_pred": round(float(p[idx == i].mean()), 3) if (idx == i).any() else None,
                           "observed_rate": round(float(y[idx == i].mean()), 3) if (idx == i).any() else None}
                          for i in range(len(bins) - 1)]
    pd.DataFrame({"pred_prob": p, "is_viral": y}).to_csv(OUT / "live_predictions.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-torch", action="store_true")
    args = ap.parse_args()
    use_torch = not args.no_torch

    raw, df = load()
    results = {"dataset": dataset_stats(raw, df)}
    try:
        results["git_commit"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                                        cwd=ROOT, text=True).strip()
    except Exception:
        results["git_commit"] = None
    print("dataset:", results["dataset"]["training_rows"], "rows,", results["dataset"]["training_viral"], "viral")

    results["label_sensitivity"] = label_sensitivity(raw)
    results["live"] = live_evaluation(raw)
    print("live:", results["live"]["n"], results["live"]["pr_auc"])

    # ---- 8.3 model comparison (same folds for every model)
    comp = {}
    oofs = {}
    plan = [("Majority (base rate)", ["log_views_per_sub_8h"], "majority"),
            ("LogReg: views/subs only", ["log_views_per_sub_8h"], "logreg"),
            ("LogReg: 5 features (v1)", V1, "logreg"),
            ("LogReg: 9 features (v2)", V2, "logreg")]
    if use_torch:
        plan += [("MLP: 5 features (v1)", V1, "mlp"), ("MLP: 9 features (v2)", V2, "mlp")]
    for name, cols, kind in plan:
        comp[name], oofs[name] = cross_validate(df, cols, kind)
        print(name, comp[name])
    results["comparison"] = comp
    pd.DataFrame(oofs).assign(y=df["y"].to_numpy()).to_csv(OUT / "oof_predictions.csv", index=False)

    # ---- 8.5 the 8-hour shift: same models on 0h features
    shift = {"LogReg 0h": cross_validate(df, H0, "logreg")[0], "LogReg 8h (v2)": comp["LogReg: 9 features (v2)"]}
    if use_torch:
        shift["MLP 0h"] = cross_validate(df, H0, "mlp")[0]
        shift["MLP 8h (v2)"] = comp["MLP: 9 features (v2)"]
    results["shift_0h_vs_8h"] = shift
    print("shift", shift)

    # ---- 8.4 feature importance: leave-one-feature-out
    kinds = ["logreg"] + (["mlp"] if use_torch else [])
    abl = {}
    for kind in kinds:
        full = comp["LogReg: 9 features (v2)" if kind == "logreg" else "MLP: 9 features (v2)"]["pr_auc"]
        abl[kind] = {"full_pr_auc": full, "drop": {}}
        for f in V2:
            m, _ = cross_validate(df, [c for c in V2 if c != f], kind)
            abl[kind]["drop"][f] = {"pr_auc_without": m["pr_auc"], "delta": round(m["pr_auc"] - full, 4)}
        print(kind, "ablation done")
    results["ablation_leave_one_out"] = abl

    # standardised logistic-regression coefficients on all data
    X = matrix(df, V2)
    Xs, _ = impute_scale(X, X)
    lr = LogisticRegression(class_weight="balanced", max_iter=5000).fit(Xs, df["y"])
    results["logreg_std_coefficients"] = {f: round(float(c), 4) for f, c in zip(V2, lr.coef_[0])}

    # ---- acceleration variant (non-overlapping windows)
    alt = [c if c != "log_acceleration" else "log_acceleration_interval" for c in V2]
    acc = {"LogReg cumulative": comp["LogReg: 9 features (v2)"]["pr_auc"],
           "LogReg interval": cross_validate(df, alt, "logreg")[0]["pr_auc"]}
    if use_torch:
        acc["MLP cumulative"] = comp["MLP: 9 features (v2)"]["pr_auc"]
        acc["MLP interval"] = cross_validate(df, alt, "mlp")[0]["pr_auc"]
    results["acceleration_variant_pr_auc"] = acc

    # ---- learning curve from the engine's own training history
    hist = [json.loads(l) for l in open(E.CFG.model_dir / "metrics_history.jsonl", encoding="utf-8")]
    lc = pd.DataFrame([{k: h.get(k) for k in ("trained_at", "n_total_labelled", "positives", "val_roc_auc",
                                              "val_pr_auc", "cv_logreg_v2_roc_auc", "cv_logreg_v2_pr_auc")}
                       for h in hist if "cv_logreg_v2_pr_auc" in h])
    lc.drop_duplicates("n_total_labelled", keep="last").to_csv(OUT / "learning_curve.csv", index=False)

    with open(OUT / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=str)
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
