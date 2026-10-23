"""GhostSpace AI - end-to-end training pipeline.
Run:  python train.py        -> writes models/model.joblib and models/results.json
Uses data/Occupancy_Estimation.csv (UCI, real) if present; otherwise a synthetic stand-in
with the identical schema so the whole project still runs.
"""
import json, os, time, warnings
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, adjusted_rand_score, confusion_matrix, davies_bouldin_score,
                             f1_score, mean_absolute_error, mean_squared_error,
                             normalized_mutual_info_score, precision_score, r2_score, recall_score,
                             silhouette_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from features import AGG, ENG, FEATS, L, RAW, S, T, aggregates, engineer

warnings.filterwarnings("ignore")
SEED, K_SELECT = 42, 10
rnd = lambda x, n=4: round(float(x), n)


# ---------------------------------------------------------------- 1. DATA
def synth(n=10129, seed=7):
    """Synthetic stand-in with the UCI schema: workday rhythm, CO2 build-up, and lights left on after people leave."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2017-12-22 10:49:41", periods=n, freq="30s")
    hr = ts.hour + ts.minute / 60
    wk = ts.dayofweek >= 5
    tgt = np.select([wk, (hr >= 9) & (hr < 12.5), (hr >= 12.5) & (hr < 14), (hr >= 14) & (hr < 18), (hr >= 18) & (hr < 20)],
                    [0.25, 2.4, 1.0, 2.2, 0.6], 0.0)
    cnt, c = np.zeros(n, int), 0
    for i in range(n):
        if rng.random() < 0.012:
            c = int(np.clip(round(rng.normal(tgt[i], 0.7)), 0, 3))
        cnt[i] = c
    on, timer, delay, lon = False, 0, 0, np.zeros(n, bool)
    for i in range(n):                       # lights stay on a random while after the room empties
        if cnt[i] > 0:
            on, timer, delay = True, 0, rng.geometric(1 / 90)
        elif on:
            timer += 1
            if timer > delay:
                on = False
        lon[i] = on
    sm = pd.Series(cnt).ewm(alpha=0.05).mean().values
    d = {}
    for j, off in enumerate([0, .2, .5, -.3], 1):
        d[f"S{j}_Temp"] = 24.9 + off + 0.35 * sm + 0.7 * np.sin((hr - 9) / 24 * 2 * np.pi) + rng.normal(0, .04, n)
    for j, b in enumerate([150, 120, 90, 60], 1):
        d[f"S{j}_Light"] = np.where(lon, b * rng.uniform(.9, 1.1, n), rng.choice([0, 0, 0, 1], n))
    for j in range(1, 5):
        d[f"S{j}_Sound"] = 0.06 + (cnt > 0) * rng.gamma(2, 0.03 * cnt + 0.01) + rng.normal(0, .008, n)
    co2 = np.zeros(n); co2[0] = 390
    for i in range(1, n):
        co2[i] = co2[i - 1] + cnt[i] * 2.6 - (co2[i - 1] - 390) * 0.012
    d["S5_CO2"] = co2 + rng.normal(0, 4, n)
    d["S5_CO2_Slope"] = (pd.Series(co2).diff(10) / 5).fillna(0).values + rng.normal(0, .1, n)
    p = np.array([.01, .15, .3, .5])[cnt]
    d["S6_PIR"] = (rng.random(n) < p).astype(int)
    d["S7_PIR"] = (rng.random(n) < p * .8).astype(int)
    df = pd.DataFrame(d)
    for col in T + L + S:                    # sprinkle ~0.2% missing so imputation has something to do
        df.loc[rng.random(n) < 0.002, col] = np.nan
    df.insert(0, "Date", ts.strftime("%Y/%m/%d")); df.insert(1, "Time", ts.strftime("%H:%M:%S"))
    df["Room_Occupancy_Count"] = cnt
    return df


def load():
    path = "data/Occupancy_Estimation.csv"
    if os.path.exists(path):
        return pd.read_csv(path), "UCI Room Occupancy Estimation (real data)"
    print("!! real CSV not found in data/ - using synthetic stand-in with the same schema")
    return synth(), "Synthetic stand-in (same schema as UCI dataset)"


def preprocess(raw):
    steps, d = [], raw.copy()
    n0 = len(d)
    d["ts"] = pd.to_datetime(d["Date"] + " " + d["Time"])
    d = d.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    steps.append(f"Parsed Date+Time into a timestamp, sorted, removed {n0 - len(d)} duplicate timestamps")
    miss = int(d[RAW].isna().sum().sum())
    steps.append(f"Found {miss} missing sensor values -> median imputation (fitted on training split only)")
    d = engineer(d)
    steps.append(f"Engineered {len(ENG)} features: sensor means/max/spread, PIR sum, log(CO2)")
    d["hour"], d["day"] = d.ts.dt.hour, d.ts.dt.strftime("%a %d %b")
    d["occ"] = d["Room_Occupancy_Count"].astype(int)
    steps.append("Target: occupant count 0-3 (multi-class classification and regression)")
    steps.append("Standardised all features (z-score) - required for KNN distances and logistic regression")
    steps.append(f"Feature selection: SelectKBest keeps the top {K_SELECT} of {len(FEATS)} features inside every pipeline")
    return d, steps


def pipe(model, score=f_classif):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("sel", SelectKBest(score, k=K_SELECT)), ("m", model)])


# ---------------------------------------------------------------- 2. CLASSIFICATION
def run_classification(d):
    X, y = d[FEATS], d["occ"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    mi = mutual_info_classif(SimpleImputer(strategy="median").fit_transform(Xtr), ytr, random_state=SEED)
    fs = pipe(GaussianNB()).fit(Xtr, ytr).named_steps["sel"]
    feat = sorted([{"name": f, "mi": rnd(m), "selected": bool(s)} for f, m, s in zip(FEATS, mi, fs.get_support())],
                  key=lambda r: -r["mi"])

    grid = GridSearchCV(pipe(KNeighborsClassifier()), {"m__n_neighbors": [1, 3, 5, 7, 11, 15, 21, 31, 51]},
                        scoring="f1_macro", cv=5).fit(Xtr, ytr)
    k_curve = [{"k": int(k), "f1": rnd(s)} for k, s in zip(grid.cv_results_["param_m__n_neighbors"], grid.cv_results_["mean_test_score"])]
    models = {"Logistic Regression": pipe(LogisticRegression(max_iter=2000)),
              f"KNN (k={grid.best_params_['m__n_neighbors']})": grid.best_estimator_,
              "Gaussian Naive Bayes": pipe(GaussianNB())}
    dd = d.sort_values("ts"); cut = int(len(dd) * .75)
    out, fitted = [], {}
    for name, p in models.items():
        t0 = time.time(); p.fit(Xtr, ytr); ft = time.time() - t0
        pred = p.predict(Xte)
        tp = clone(p).fit(dd[FEATS][:cut], dd["occ"][:cut]).predict(dd[FEATS][cut:])
        cv = cross_val_score(clone(p), Xtr, ytr, cv=5, scoring="f1_macro")
        out.append({"model": name, "accuracy": rnd(accuracy_score(yte, pred)),
                    "precision": rnd(precision_score(yte, pred, average="macro", zero_division=0)),
                    "recall": rnd(recall_score(yte, pred, average="macro", zero_division=0)),
                    "f1": rnd(f1_score(yte, pred, average="macro")),
                    "cv_f1": rnd(cv.mean()), "cv_sd": rnd(cv.std()),
                    "time_split_f1": rnd(f1_score(dd["occ"][cut:], tp, average="macro", labels=sorted(set(dd["occ"][cut:])), zero_division=0)),
                    "fit_s": rnd(ft, 3), "cm": confusion_matrix(yte, pred, labels=[0, 1, 2, 3]).tolist()})
        fitted[name] = p
    best = max(out, key=lambda r: r["f1"])["model"]

    return {"table": out, "best": best, "features": feat, "k_curve": k_curve,
            "n_train": len(Xtr), "n_test": len(Xte)}, fitted[best], best


# ---------------------------------------------------------------- 3. REGRESSION
def run_regression(d):
    X, y = d[FEATS], d["occ"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    grid = GridSearchCV(pipe(KNeighborsRegressor(), f_regression), {"m__n_neighbors": [3, 5, 7, 11, 15, 21, 31]},
                        scoring="neg_mean_absolute_error", cv=5).fit(Xtr, ytr)
    models = {"Linear Regression": pipe(LinearRegression(), f_regression), "Ridge (alpha=1)": pipe(Ridge(1.0), f_regression),
              f"KNN Regressor (k={grid.best_params_['m__n_neighbors']})": grid.best_estimator_}
    rows, fitted = [], {}
    for n, p in models.items():
        p.fit(Xtr, ytr); pr = np.clip(p.predict(Xte), 0, 3)
        rows.append({"model": n, "mae": rnd(mean_absolute_error(yte, pr)), "rmse": rnd(mean_squared_error(yte, pr) ** .5),
                     "r2": rnd(r2_score(yte, pr)), "rounded_acc": rnd(accuracy_score(yte, np.rint(pr)))})
        fitted[n] = (p, pr)
    best = min(rows, key=lambda r: r["mae"])["model"]
    sel = np.random.default_rng(1).choice(len(yte), 220, replace=False)
    return {"table": rows, "best": best, "scatter": {"true": yte.values[sel].tolist(), "pred": fitted[best][1][sel].round(3).tolist()}}, fitted[best][0]


# ---------------------------------------------------------------- 4. CLUSTERING (labels never used for fitting)
def run_clustering(d):
    Xc = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(d[RAW]))
    y = d["occ"].values
    rng = np.random.default_rng(SEED); sub = rng.choice(len(Xc), 4000, replace=False)
    Xs, ys = Xc[sub], y[sub]
    sweep = []
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=SEED).fit(Xc)
        hl = AgglomerativeClustering(k, linkage="ward").fit_predict(Xs)
        sweep.append({"k": k, "km_inertia": rnd(km.inertia_, 1), "km_sil": rnd(silhouette_score(Xc, km.labels_, sample_size=4000, random_state=SEED)),
                      "km_db": rnd(davies_bouldin_score(Xc, km.labels_)), "hc_sil": rnd(silhouette_score(Xs, hl)), "hc_db": rnd(davies_bouldin_score(Xs, hl))})
    K = 4   # four occupancy levels (0-3) -> fair comparison against the ground truth
    km = KMeans(K, n_init=10, random_state=SEED).fit(Xc)
    lab = {"K-Means": km.predict(Xs), "Hierarchical (Ward)": AgglomerativeClustering(K, linkage="ward").fit_predict(Xs)}
    comp = [{"algo": a, "silhouette": rnd(silhouette_score(Xs, l)), "davies_bouldin": rnd(davies_bouldin_score(Xs, l)),
             "ari": rnd(adjusted_rand_score(ys, l)), "nmi": rnd(normalized_mutual_info_score(ys, l))} for a, l in lab.items()]
    pc = PCA(2, random_state=SEED).fit(Xc); P = pc.transform(Xs)[:900]
    scatter = {"x": P[:, 0].round(3).tolist(), "y": P[:, 1].round(3).tolist(), "true": ys[:900].tolist(),
               **{a: l[:900].tolist() for a, l in lab.items()}, "var": pc.explained_variance_ratio_.round(3).tolist()}

    # deployed model: K-Means with k=3 -> "usage states" ranked by average occupancy
    pre = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]).fit(d[RAW])
    k3 = KMeans(3, n_init=10, random_state=SEED).fit(pre.transform(d[RAW]))
    d["cl"] = k3.labels_
    order = d.groupby("cl")["occ"].mean().sort_values().index.tolist()
    names = ["Vacant", "Light use", "Busy"]; rank = {int(c): names[i] for i, c in enumerate(order)}
    d["state"] = d["cl"].map(rank)
    prof = [{"state": names[i], "share": rnd(100 * (d.cl == c).mean(), 1), "occ": rnd(d[d.cl == c].occ.mean(), 2), "light": rnd(d[d.cl == c].light_mean.mean(), 1),
             "co2": rnd(d[d.cl == c].S5_CO2.mean(), 0), "sound": rnd(d[d.cl == c].sound_mean.mean(), 3)} for i, c in enumerate(order)]
    return {"sweep": sweep, "compare": comp, "scatter": scatter, "profile": prof, "k_compare": K}, (pre, k3, rank)


# ---------------------------------------------------------------- 5. USAGE PATTERNS ("ghost" detection)
def run_patterns(d):
    thr = max(5.0, 0.3 * d.loc[d.occ > 0, "light_mean"].median())
    d["lit"], d["vac"] = d.light_mean > thr, d.occ == 0
    d["ghost"] = d.lit & d.vac
    hrs = lambda m: rnd(m.sum() * 30 / 3600, 1)
    hourly = d.groupby("hour").agg(occ=("occ", "mean"), ghost=("ghost", "mean"), lit=("lit", "mean"), n=("occ", "size"))
    st = pd.crosstab(d.hour, d.state, normalize="index").reindex(columns=["Vacant", "Light use", "Busy"], fill_value=0)
    grid = d.groupby(["day", "hour"]).agg(occ=("occ", "mean"), ghost=("ghost", "mean"), n=("occ", "size")).reset_index()
    grid = grid[grid.n >= 20]
    order = d.drop_duplicates("day")["day"].tolist()
    g_hr = hourly[hourly.n >= 100].sort_values("ghost", ascending=False)
    peak = hourly[hourly.n >= 100].sort_values("occ", ascending=False).head(3).index.tolist()
    ghost_h, lit_h = hrs(d.ghost), hrs(d.lit)
    insights = [f"Lights were on for {lit_h} h but the room was empty for {ghost_h} h of that ({rnd(100 * ghost_h / max(lit_h, 1e-9), 0):.0f}%).",
                f"Ghost time peaks around {', '.join(f'{h:02d}:00' for h in g_hr.head(3).index)}.",
                f"Heaviest use is around {', '.join(f'{h:02d}:00' for h in sorted(peak))}; plan cleaning and HVAC boost for those windows.",
                f"The room is empty {rnd(100 * d.vac.mean(), 0):.0f}% of the time - a candidate for auto-off lighting and setback HVAC."]
    return {"threshold_lux": rnd(thr, 1), "total_h": hrs(d.occ >= 0), "occupied_h": hrs(~d.vac), "lit_h": lit_h, "ghost_h": ghost_h,
            "hourly": [{"hour": int(h), "occ": rnd(r.occ, 3), "ghost": rnd(r.ghost, 3), "lit": rnd(r.lit, 3), "n": int(r.n),
                        "vacant": rnd(st.loc[h, "Vacant"], 3), "light": rnd(st.loc[h, "Light use"], 3), "busy": rnd(st.loc[h, "Busy"], 3)} for h, r in hourly.iterrows()],
            "days": order, "grid": [{"day": r.day, "hour": int(r.hour), "occ": rnd(r.occ, 2), "ghost": rnd(r.ghost, 2)} for r in grid.itertuples()],
            "insights": insights}


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    raw, source = load()
    d, steps = preprocess(raw)
    print(f"{source}: {len(d)} rows"); t0 = time.time()
    cls, cls_model, cls_name = run_classification(d); print("classification done", rnd(time.time() - t0, 1), "s", cls_name)
    reg, reg_model = run_regression(d); print("regression done")
    clu, (pre, k3, rank) = run_clustering(d); print("clustering done")
    pat = run_patterns(d); print("patterns done")

    A = aggregates(d)
    base = d.loc[d.occ > 0, RAW].median()
    ranges = {c: [rnd(A[c].quantile(.01), 2), rnd(A[c].quantile(.99), 2)] for c in AGG}
    samples = {int(k): A[d.occ == k].sample(min(60, int((d.occ == k).sum())), random_state=SEED).round(3).to_dict("records") for k in range(4)}
    joblib.dump({"cls": cls_model, "reg": reg_model, "pre": pre, "km": k3, "rank": rank, "base": base,
                 "thr": pat["threshold_lux"]}, "models/model.joblib")
    results = {"meta": {"source": source, "rows": len(d), "start": str(d.ts.min()), "end": str(d.ts.max()),
                        "class_counts": d.occ.value_counts().sort_index().tolist(), "steps": steps, "sensors": len(RAW), "k_select": K_SELECT},
               "classification": cls, "regression": reg, "clustering": clu, "patterns": pat, "ranges": ranges, "samples": samples}
    json.dump(results, open("models/results.json", "w"))
    print("saved models/model.joblib + models/results.json")
