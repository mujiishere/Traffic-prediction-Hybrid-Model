"""
Traffic Hybrid Speed Prediction
--------------------------------
This script implements a hybrid traffic prediction model combining:
1. Historical sliding-window pattern matching
2. Bayesian residual correction
3. Final hybrid prediction = historical + Bayesian residual

Dataset: METR-LA (publicly available traffic speed dataset)
Density is estimated using Greenshields model since METR-LA contains speed only.
"""

# -----------------------------------------------
# Imports & Fixes for NumPy compatibility issues
# -----------------------------------------------
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# Fix deprecated NumPy attributes (important for pgmpy)
if not hasattr(np, "product"): np.product = np.prod
if not hasattr(np, "bool"): np.bool = bool
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "object"): np.object = object

print("NumPy version:", np.__version__)


# -----------------------------
# Load and preprocess METR-LA
# -----------------------------
CSV_PATH = "METR-LA.csv"
df_all = pd.read_csv(CSV_PATH)

# Rename first column to timestamp
first_col = df_all.columns[0]
df_all = df_all.rename(columns={first_col: "timestamp"})

# Parse timestamp (day-first format)
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], dayfirst=True, errors="coerce")

# Convert all sensor columns to numeric
sensor_cols = [c for c in df_all.columns if c != "timestamp"]
for col in sensor_cols:
    df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

# Select 3 sensors arbitrarily (A, B, C)
A_col, B_col, C_col = sensor_cols[:3]

# Build main dataframe
df = pd.DataFrame({
    "speed_A": df_all[A_col],
    "speed_B": df_all[B_col],
    "speed_C": df_all[C_col],
    "timestamp": df_all["timestamp"]
})


# ---------------------------------------------
# Synthetic density using Greenshields model
# ---------------------------------------------
v_free = 65      # assumed free-flow speed (mph)
k_jam = 180      # assumed jam density (veh/km)

df["density_A"] = k_jam * (1 - df["speed_A"]/v_free)
df["density_B"] = k_jam * (1 - df["speed_B"]/v_free)
df["density_C"] = k_jam * (1 - df["speed_C"]/v_free)


# ---------------------------------------------
# Create t + 1 hour prediction targets
# ---------------------------------------------
H = 12             # 12 × 5min = 1 hour
prefix_len = 36    # 36 × 5min = 3 hours history

df["speed_A_tplus"]   = df["speed_A"].shift(-H)
df["density_A_tplus"] = df["density_A"].shift(-H)

df = df.dropna().reset_index(drop=True)


# ---------------------------------------------
# Sliding window helper
# ---------------------------------------------
def sliding(series, L):
    arr = np.asarray(series)
    return np.lib.stride_tricks.sliding_window_view(arr, L)[:len(arr)-L-H+1]

speed_windows = sliding(df["speed_A"], prefix_len)
dens_windows  = sliding(df["density_A"], prefix_len)


# ---------------------------------------------
# Historical predictor (pattern matching)
# ---------------------------------------------
def historical_predict(ps, pd):
    ps_norm = (ps - ps.mean()) / (ps.std() + 1e-9)
    pd_norm = (pd - pd.mean()) / (pd.std() + 1e-9)

    sw_norm = (speed_windows - speed_windows.mean(1, keepdims=True)) / (speed_windows.std(1, keepdims=True) + 1e-9)
    dw_norm = (dens_windows  - dens_windows.mean(1, keepdims=True))  / (dens_windows.std(1, keepdims=True)  + 1e-9)

    dist = np.linalg.norm(sw_norm - ps_norm, 1) + np.linalg.norm(dw_norm - pd_norm, 1)
    best = np.argmin(dist)
    idx = best + prefix_len

    return df["speed_A"].iloc[idx], df["density_A"].iloc[idx]


# ---------------------------------------------
# Generate historical predictions
# ---------------------------------------------
N = len(df) - prefix_len - 1
hist_sp_preds = []
hist_den_preds = []

for i in range(N):
    ps = df["speed_A"].values[i:i+prefix_len]
    pd = df["density_A"].values[i:i+prefix_len]
    hsp, hden = historical_predict(ps, pd)
    hist_sp_preds.append(hsp)
    hist_den_preds.append(hden)

# Align for residual calculations
df_res = df.iloc[:N].copy()
df_res["hist_sp"]  = hist_sp_preds
df_res["hist_den"] = hist_den_preds

true_sp  = df_res["speed_A_tplus"].values
true_den = df_res["density_A_tplus"].values


# ---------------------------------------------
# Compute residuals (true - historical)
# ---------------------------------------------
df_res["residual_speed"]   = df_res["speed_A_tplus"]   - df_res["hist_sp"]
df_res["residual_density"] = df_res["density_A_tplus"] - df_res["hist_den"]

# Normalize residuals and discretize
res_sp_norm  = (df_res["residual_speed"] - df_res["residual_speed"].mean()) / df_res["residual_speed"].std()
res_den_norm = (df_res["residual_density"] - df_res["residual_density"].mean()) / df_res["residual_density"].std()

n_bins = 12
km_res_sp  = KMeans(n_clusters=n_bins, n_init=10).fit(res_sp_norm.values.reshape(-1,1))
km_res_den = KMeans(n_clusters=n_bins, n_init=10).fit(res_den_norm.values.reshape(-1,1))

df_res["res_sp_disc"]  = km_res_sp.predict(res_sp_norm.values.reshape(-1,1))
df_res["res_den_disc"] = km_res_den.predict(res_den_norm.values.reshape(-1,1))


# ---------------------------------------------
# Discretize current speed/density
# ---------------------------------------------
km_speed = KMeans(n_clusters=n_bins, n_init=10).fit(df_res["speed_A"].values.reshape(-1,1))
km_dens  = KMeans(n_clusters=n_bins, n_init=10).fit(df_res["density_A"].values.reshape(-1,1))

df_res["speed_t_disc"] = km_speed.predict(df_res["speed_A"].values.reshape(-1,1))
df_res["dens_t_disc"]  = km_dens.predict(df_res["density_A"].values.reshape(-1,1))


# ---------------------------------------------
# Learn Bayesian CPT: P(residual | current bin)
# ---------------------------------------------
from collections import defaultdict
alpha = 1.0

counts_res_sp  = defaultdict(lambda: np.zeros(n_bins))
counts_res_den = defaultdict(lambda: np.zeros(n_bins))
counts_parent  = defaultdict(float)

for _, r in df_res.iterrows():
    key = (int(r["speed_t_disc"]), int(r["dens_t_disc"]))
    counts_res_sp[key][int(r["res_sp_disc"])] += 1
    counts_res_den[key][int(r["res_den_disc"])] += 1
    counts_parent[key] += 1

# Compute CPTs
cpt_res_sp = {}
cpt_res_den = {}

for k, total in counts_parent.items():
    denom = total + alpha*n_bins
    cpt_res_sp[k]  = (counts_res_sp[k]  + alpha) / denom
    cpt_res_den[k] = (counts_res_den[k] + alpha) / denom

# Global fallback distribution
global_res_sp  = np.mean(np.stack(list(cpt_res_sp.values())), axis=0)
global_res_den = np.mean(np.stack(list(cpt_res_den.values())), axis=0)


# ----------------------------------------------------------
# Bayesian prediction of residuals using learned CPTs
# ----------------------------------------------------------
def bn_predict_residual(sp, dn):
    s_bin = int(km_speed.predict([[sp]])[0])
    d_bin = int(km_dens.predict([[dn]])[0])
    key = (s_bin, d_bin)

    sp_probs  = cpt_res_sp.get(key, global_res_sp)
    den_probs = cpt_res_den.get(key, global_res_den)

    sp_bin  = int(np.argmax(sp_probs))
    den_bin = int(np.argmax(den_probs))

    sp_res  = df_res.loc[df_res["res_sp_disc"] == sp_bin,  "residual_speed"].median()
    den_res = df_res.loc[df_res["res_den_disc"] == den_bin,"residual_density"].median()

    return sp_res, den_res


# ----------------------------------------------------------
# Hybrid Prediction = Historical + BN Residual
# ----------------------------------------------------------
bn_sp_preds = []
bn_den_preds = []
hyb_sp_preds = []
hyb_den_preds = []

for i in range(N):
    sp_now = df_res["speed_A"].iloc[i]
    dn_now = df_res["density_A"].iloc[i]

    rsp, rden = bn_predict_residual(sp_now, dn_now)
    bn_sp_preds.append(rsp)
    bn_den_preds.append(rden)

    # Hybrid = historical + correction
    hyb_sp_preds.append(hist_sp_preds[i] + rsp)
    hyb_den_preds.append(hist_den_preds[i] + rden)


# ----------------------------------------------------------
# Evaluation: sMAPE + Hybrid Congestion FP/FN
# ----------------------------------------------------------
def smape(y, p):
    y = np.array(y)
    p = np.array(p)
    denom = (np.abs(y) + np.abs(p)) / 2
    denom[denom == 0] = 1e-9
    return 100 * np.mean(np.abs(y - p) / denom)

# Compute sMAPE
sm_hist = smape(true_sp, hist_sp_preds)
true_residual_sp = true_sp - np.array(hist_sp_preds)
sm_bn = smape(true_residual_sp, bn_sp_preds)
sm_hyb = smape(true_sp, hyb_sp_preds)

# Congestion classification threshold
thr_speed = 35
true_cls = (true_sp < thr_speed).astype(int)
hyb_cls  = (np.array(hyb_sp_preds) < thr_speed).astype(int)

tn, fp, fn, tp = confusion_matrix(true_cls, hyb_cls, labels=[0,1]).ravel()

fp_hyb = fp / (fp + tn + 1e-9) * 100
fn_hyb = fn / (fn + tp + 1e-9) * 100

# Print results
print("=================================")
print("        MODEL PERFORMANCE        ")
print("=================================")
print("Historical sMAPE:           ", round(sm_hist, 4))
print("Bayesian Residual sMAPE:    ", round(sm_bn, 4))
print("Hybrid Model sMAPE:         ", round(sm_hyb, 4))
print("---------------------------------")
print("Hybrid FP(%):", round(fp_hyb, 4))
print("Hybrid FN(%):", round(fn_hyb, 4))
print("=================================")
