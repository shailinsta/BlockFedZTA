import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import ttest_rel, t
import xgboost as xgb

# =========================
# LOAD DATA
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/Final_5Class_IDS.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset not found!")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

X_raw = df.drop(columns=["label"])
y_raw = df["label"]

X_raw = X_raw.loc[:, X_raw.nunique() > 1]
X_raw = X_raw.replace([np.inf, -np.inf], 0).fillna(0).clip(-1e6, 1e6)

le = LabelEncoder()
y_enc = le.fit_transform(y_raw)
n_classes = len(np.unique(y_enc))
CLASS_NAMES = list(le.classes_)

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_raw)

# =========================
# SETTINGS
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
RESULTS = {}

DRIFT_CFG = {
    "NO DRIFT":     dict(noise=0.00, blackout=0.00, flip=0.00),
    "MILD DRIFT":   dict(noise=0.04, blackout=0.08, flip=0.02),
    "SEVERE DRIFT": dict(noise=0.08, blackout=0.12, flip=0.04),
}

# =========================
# 95% CONFIDENCE INTERVAL
# =========================
def compute_ci(data, confidence=0.95):
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    t_value = t.ppf((1 + confidence) / 2., n-1)
    margin = t_value * (std / np.sqrt(n))

    return mean, mean - margin, mean + margin

def compute_ci_pm(data, confidence=0.95):
    mean, low, high = compute_ci(data, confidence)
    margin = (high - low) / 2
    return mean, margin

# =========================
# FUNCTIONS
# =========================
def poison_labels(y, ratio=0.3):
    y = y.copy()
    n = int(len(y) * ratio)
    if n > 0:
        idx = np.random.choice(len(y), n, replace=False)
        y[idx] = np.random.permutation(y[idx])
    return y

def degrade(X, y, cid, cfg):
    X = X.copy(); y = y.copy()

    if cfg["noise"] > 0:
        X += np.random.normal(0, cfg["noise"] * (1 + 0.5*cid), X.shape)

    if cfg["blackout"] > 0:
        k = int(X.shape[1] * cfg["blackout"])
        X[:, :k] = 0

    if cid == 1 and cfg["flip"] > 0:
        y = poison_labels(y, cfg["flip"])

    return X, y

def build_model(cid, cfg):
    if cid == 0 or cfg["noise"] == 0:
        return xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=150,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            verbosity=0,
            random_state=42
        )
    else:
        return xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.75,
            colsample_bytree=0.75,
            tree_method="hist",
            verbosity=0,
            random_state=42
        )

def compute_trust(model, X_val, y_val, prev_trust):
    probs = model.predict_proba(X_val)
    pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_val, pred)
    conf = np.mean(np.max(probs, axis=1))

    return 0.7*acc + 0.2*conf + 0.1*prev_trust

# =========================
# MAIN EXPERIMENT
# =========================
def run_experiment(name, cfg, NUM_CLIENTS=8):

    print(f"\n===== {name} =====")

    accs, fedavg_accs = [], []

    for fold, (tr, te) in enumerate(skf.split(X_sc, y_enc), 1):

        print(f"\nFold {fold}")

        prev_trusts = np.ones(NUM_CLIENTS) * 0.5

        Xtr, Xte = X_sc[tr], X_sc[te]
        ytr, yte = y_enc[tr], y_enc[te]

        splits = np.array_split(range(len(Xtr)), NUM_CLIENTS)

        models, trust_scores = [], []

        for cid, idx in enumerate(splits):

            X_loc = Xtr[idx]
            y_loc = ytr[idx]

            for c in range(n_classes):
                if c not in y_loc:
                    X_loc = np.vstack([X_loc, np.zeros((1, X_loc.shape[1]))])
                    y_loc = np.append(y_loc, c)

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_loc, y_loc, test_size=0.2,
                stratify=y_loc, random_state=fold
            )

            if cid > 0:
                X_tr, y_tr = degrade(X_tr, y_tr, cid, cfg)

            model = build_model(cid, cfg)
            model.fit(X_tr, y_tr)

            trust = compute_trust(model, X_val, y_val, prev_trusts[cid])
            prev_trusts[cid] = trust

            print(f"Client {cid} Trust: {trust:.4f}")

            models.append(model)
            trust_scores.append(trust)

        t_arr = np.array(trust_scores)
        t_arr = (t_arr - t_arr.min()) / (t_arr.max() - t_arr.min() + 1e-9)
        t_arr = t_arr ** 8
        w = t_arr / (t_arr.sum() + 1e-9)

        probs = [m.predict_proba(Xte) for m in models]

        trust_pred = np.argmax(sum(wi*p for wi,p in zip(w, probs)), axis=1)
        fedavg_pred = np.argmax(np.mean(probs, axis=0), axis=1)

        t_acc = accuracy_score(yte, trust_pred)
        f_acc = accuracy_score(yte, fedavg_pred)

        print(f"Fold {fold} → Trust={t_acc:.4f} | FedAvg={f_acc:.4f}")

        accs.append(t_acc)
        fedavg_accs.append(f_acc)

    RESULTS[name] = {
        "trust": accs,
        "fed": fedavg_accs
    }

# =========================
# RUN
# =========================
run_experiment("NO DRIFT", DRIFT_CFG["NO DRIFT"])
run_experiment("MILD DRIFT", DRIFT_CFG["MILD DRIFT"])
run_experiment("SEVERE DRIFT", DRIFT_CFG["SEVERE DRIFT"])

# =========================
# FINAL SUMMARY (± FORMAT)
# =========================
print("\n" + "="*70)
print("95% CONFIDENCE INTERVAL (CI)")
print("="*70)

for s in RESULTS:

    t_mean, t_margin = compute_ci_pm(RESULTS[s]["trust"])
    f_mean, f_margin = compute_ci_pm(RESULTS[s]["fed"])

    print(f"\n{s}")
    print(f"Trust  : {t_mean:.4f} ± {t_margin:.4f} (95% CI)")
    print(f"FedAvg : {f_mean:.4f} ± {f_margin:.4f} (95% CI)")

print("="*70)

# =========================
# T-TEST + CI
# =========================
print("\n" + "="*70)
print(f"{'SCENARIO':<15} {'METHOD':<12} {'ACCURACY (95% CI)':<30}")
print("="*70)

for s in RESULTS:

    trust = np.array(RESULTS[s]["trust"])
    fed   = np.array(RESULTS[s]["fed"])

    t_mean, t_margin = compute_ci_pm(trust)
    f_mean, f_margin = compute_ci_pm(fed)

    t_stat, p_val = ttest_rel(trust, fed)

    if p_val < 0.01:
        sig = "Highly Significant"
    elif p_val < 0.05:
        sig = "Significant"
    else:
        sig = "Not Significant"

    print(f"{s:<15} {'Trust':<12} {t_mean:.4f} ± {t_margin:.4f}")
    print(f"{'':<15} {'FedAvg':<12} {f_mean:.4f} ± {f_margin:.4f}")
    print(f"{'':<15} {'T-stat':<12} {t_stat:.4f}")
    print(f"{'':<15} {'P-value':<12} {p_val:.6f}")
    print(f"{'':<15} {'Result':<12} {sig}")
    print("-"*70)

print("="*70)
print("Statistical validation complete.")