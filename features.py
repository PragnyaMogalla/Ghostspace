"""Shared feature definitions: used by training AND the API so they never drift apart."""
import numpy as np, pandas as pd

T = [f"S{i}_Temp" for i in range(1, 5)]
L = [f"S{i}_Light" for i in range(1, 5)]
S = [f"S{i}_Sound" for i in range(1, 5)]
RAW = T + L + S + ["S5_CO2", "S5_CO2_Slope", "S6_PIR", "S7_PIR"]
ENG = ["temp_mean", "temp_spread", "light_mean", "light_max",
       "sound_mean", "sound_max", "pir_sum", "co2_log"]
FEATS = RAW + ENG
AGG = ["temp", "light", "sound", "co2", "slope", "pir"]   # the 6 controls shown in the UI


def engineer(df):
    d = df.copy()
    d["temp_mean"] = d[T].mean(1)
    d["temp_spread"] = d[T].max(1) - d[T].min(1)
    d["light_mean"] = d[L].mean(1)
    d["light_max"] = d[L].max(1)
    d["sound_mean"] = d[S].mean(1)
    d["sound_max"] = d[S].max(1)
    d["pir_sum"] = d["S6_PIR"] + d["S7_PIR"]
    d["co2_log"] = np.log1p(d["S5_CO2"])
    return d


def aggregates(d):
    """Collapse a dataframe of raw sensors into the 6 UI-level controls."""
    return pd.DataFrame({"temp": d[T].mean(1), "light": d[L].mean(1), "sound": d[S].mean(1),
                         "co2": d["S5_CO2"], "slope": d["S5_CO2_Slope"],
                         "pir": d["S6_PIR"] + d["S7_PIR"]})


def from_aggregates(a, base):
    """Rebuild a full 16-sensor reading from 6 aggregate values, using a typical 'occupied' row as template."""
    r = base.copy()
    r[T] = base[T] - base[T].mean() + a["temp"]
    bl = base[L].mean()
    r[L] = base[L] * (a["light"] / bl) if bl > 1 else a["light"]
    bs = base[S].mean()
    r[S] = base[S] * (a["sound"] / bs) if bs > 1e-6 else a["sound"]
    r["S5_CO2"], r["S5_CO2_Slope"] = a["co2"], a["slope"]
    p = int(round(a["pir"]))
    r["S6_PIR"], r["S7_PIR"] = (1.0 if p >= 1 else 0.0), (1.0 if p >= 2 else 0.0)
    return r
