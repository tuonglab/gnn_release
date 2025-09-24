import numpy as np
import pandas as pd

# Settings
csv_path = "/scratch/project/tcr_ml/gnn_release/clonal_frequency_scoring/20240530_WGS_20240530_sc_PICA0001-PICA0007_PMID_97-101_Pool_3_2_merged.csv"
method = "softmax"          # "power" or "softmax"
beta = 0.9                # used if method == "power"
tau = 1.5                 # used if method == "softmax"
alpha = 0.6               # noisy OR sensitivity
gamma = 0.2               # logit bump strength. 0 means off
eps = 1e-12

# Math helpers
def logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Load
df = pd.read_csv(csv_path)
mean_prob_unweighted = df["prob"].astype(float).mean()

w = df["CloneFreq"].astype(float).to_numpy()
p = df["prob"].astype(float).to_numpy()

# Normalize frequencies
w = np.clip(w, 0.0, None)
if np.all(w == 0):
    w = np.ones_like(w)
w_norm = w / (w.sum() + eps)
logw = np.log(w_norm + eps)

# Priority weights a_i from frequencies
if method == "power":
    a_unnorm = np.power(w_norm + eps, beta)
elif method == "softmax":
    a_unnorm = np.exp(logw / tau)
else:
    raise ValueError("method must be 'power' or 'softmax'")
a = a_unnorm / (a_unnorm.sum() + eps)

# Optional logit bump of per clone probabilities
p = np.clip(p, 0.0, 1.0)
if gamma != 0.0:
    h = (logw - logw.mean()) / (logw.std(ddof=0) + eps)  # z score of log w
    p_star = sigmoid(logit(p, eps) + gamma * h)
else:
    p_star = p

# Noisy OR aggregation in log space for stability
log_terms = np.log(1.0 - p_star + eps)
log_prod = np.sum((alpha * a) * log_terms)
P_cancer = 1.0 - np.exp(log_prod)

# Optional small signal linear approximation
linear_mean = float(np.sum(a * p_star))

print("n_clones:", len(w))
print("P_cancer:", float(P_cancer))
print("linear_mean_a_pstar:", linear_mean)

# If you want per sequence outputs, attach to df:
df_out = df.copy()
df_out["w_norm"] = w_norm
df_out["a_priority"] = a
df_out["p_star"] = p_star
# df_out now contains per sequence fields you can save or inspect
print("Unweighted mean probability (no transformation, no frequency):", mean_prob_unweighted)
