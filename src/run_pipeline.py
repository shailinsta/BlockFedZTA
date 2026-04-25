import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb


# =========================
# LOAD
# =========================
print("Loading dataset...")
df = pd.read_csv(r"C:\Users\Tahzib\Desktop\BlockFedZTA-main\BlockFedZTA-main\data\Final_5Class_IDS\Final_5Class_IDS.csv")


X_raw = df.drop(columns=["label"])
y_raw = df["label"]

X_raw = X_raw.loc[:, X_raw.nunique() > 1]
X_raw = X_raw.replace([np.inf, -np.inf], 0).fillna(0).clip(-1e6, 1e6)

le = LabelEncoder()
y_enc = le.fit_transform(y_raw)
n_classes = len(np.unique(y_enc))
CLASS_NAMES = list(le.classes_)

print("Classes:", CLASS_NAMES)
print("Features:", X_raw.shape[1])


# =========================
# SCALE
# =========================
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_raw)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

RESULTS = {}


# =========================
# DRIFT CONFIG
# =========================
DRIFT_CFG = {
    "NO DRIFT":     dict(noise=0.00, blackout=0.00, flip=0.00),
    "MILD DRIFT":   dict(noise=0.04, blackout=0.08, flip=0.02),
    "SEVERE DRIFT": dict(noise=0.08, blackout=0.12, flip=0.04),
}


# =========================
# POISONING
# =========================
def poison_labels(y, ratio=0.3):
    y = y.copy()
    n = int(len(y) * ratio)
    if n > 0:
        idx = np.random.choice(len(y), n, replace=False)
        y[idx] = np.random.permutation(y[idx])
    return y


# =========================
# DEGRADATION
# =========================
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


# =========================
# MODEL
# =========================
def build_model(cid, cfg):
    if cid == 0 or cfg["noise"] == 0:
        return xgb.XGBClassifier(
            objective="multi:softprob", num_class=n_classes,
            n_estimators=150, max_depth=7,
            learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, tree_method="hist",
            verbosity=0, random_state=42
        )
    else:
        return xgb.XGBClassifier(
            objective="multi:softprob", num_class=n_classes,
            n_estimators=100, max_depth=5,
            learning_rate=0.08, subsample=0.75,
            colsample_bytree=0.75, tree_method="hist",
            verbosity=0, random_state=42
        )


# =========================
# EXPERIMENT
# =========================
def run_experiment(name, cfg, NUM_CLIENTS=8):

    print("\n" + "="*60)
    print(f"SCENARIO: {name} | CLIENTS: {NUM_CLIENTS}")
    print("="*60)

    accs, fedavg_accs = [], []
    prev_trusts = np.ones(NUM_CLIENTS) * 0.5

    for fold, (tr, te) in enumerate(skf.split(X_sc, y_enc), 1):

        print(f"\n--- FOLD {fold} ---")

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

            probs = model.predict_proba(X_val)
            pred = np.argmax(probs, axis=1)

            acc = accuracy_score(y_val, pred)
            conf = np.mean(np.max(probs, axis=1))

            trust = 0.7*acc + 0.2*conf + 0.1*prev_trusts[cid]
            prev_trusts[cid] = trust

            print(f"Client {cid:<2} | Trust: {trust:.4f}")

            models.append(model)
            trust_scores.append(trust)

        t = np.array(trust_scores)
        t = (t - t.min()) / (t.max() - t.min() + 1e-9)
        t = t ** 8
        w = t / (t.sum() + 1e-9)

        probs = [m.predict_proba(Xte) for m in models]

        trust_probs = sum(wi * p for wi, p in zip(w, probs))
        trust_pred = np.argmax(trust_probs, axis=1)

        fedavg_probs = np.mean(probs, axis=0)
        fedavg_pred = np.argmax(fedavg_probs, axis=1)

        t_acc = accuracy_score(yte, trust_pred)
        f_acc = accuracy_score(yte, fedavg_pred)

        print(f"Result → Trust={t_acc:.4f} | FedAvg={f_acc:.4f}")

        accs.append(t_acc)
        fedavg_accs.append(f_acc)

    mean_t = np.mean(accs)
    mean_f = np.mean(fedavg_accs)

    print("\n" + "-"*60)
    print(f"FINAL RESULT ({name})")
    print(f"Trust Mean : {mean_t:.4f}")
    print(f"FedAvg Mean: {mean_f:.4f}")
    print("-"*60)

    RESULTS[name] = (mean_t, mean_f)


# =========================
# RUN BASE
# =========================
run_experiment("NO DRIFT", DRIFT_CFG["NO DRIFT"])
run_experiment("MILD DRIFT", DRIFT_CFG["MILD DRIFT"])
run_experiment("SEVERE DRIFT", DRIFT_CFG["SEVERE DRIFT"])


# =========================
# SCALABILITY
# =========================
CLIENTS = [3, 5, 8, 10, 15]

for n in CLIENTS:
    run_experiment(f"{n}_CLIENTS", DRIFT_CFG["SEVERE DRIFT"], NUM_CLIENTS=n)


# =========================
# FINAL SUMMARY
# =========================
print("\n" + "="*60)
print(f"{'SCENARIO':<20} {'TRUST':<10} {'FEDAVG':<10}")
print("="*60)

for k in RESULTS:
    t, f = RESULTS[k]
    print(f"{k:<20} {t:.4f}     {f:.4f}")

print("="*60)