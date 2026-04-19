import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb


# ---------- LOAD ----------
print("Loading dataset...")
df = pd.read_csv(r"C:\Users\Tahzib\Desktop\BlockFedZTA\data\raw\Final_5Class_IDS.csv")

X_raw = df.drop(columns=["label"])
y_raw = df["label"]

X_raw = X_raw.loc[:, X_raw.nunique() > 1]
X_raw = X_raw.replace([np.inf, -np.inf], 0).fillna(0).clip(-1e6, 1e6)

le = LabelEncoder()
y_enc = le.fit_transform(y_raw)
n_classes = len(np.unique(y_enc))
CLASS_NAMES = list(le.classes_)

print("Classes:", CLASS_NAMES)
print("Feature count:", X_raw.shape[1])


# ---------- SCALING ----------
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_raw)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

RESULTS = {}


# ---------- DRIFT ----------
DRIFT_CFG = {
    "NO DRIFT":     dict(noise=0.00, blackout=0.00, flip=0.00),
    "MILD DRIFT":   dict(noise=0.04, blackout=0.08, flip=0.02),
    "SEVERE DRIFT": dict(noise=0.08, blackout=0.12, flip=0.04),
}


def degrade(X, y, cid, cfg):
    X = X.copy(); y = y.copy()

    if cfg["noise"] > 0:
        scale = cfg["noise"] * (1 + 0.5 * cid)
        X += np.random.normal(0, scale, X.shape)

    if cfg["blackout"] > 0:
        n_feat = int(X.shape[1] * cfg["blackout"] * (1 + 0.3 * cid))
        X[:, :n_feat] = 0

    if cfg["flip"] > 0:
        ratio = cfg["flip"] * (1 + 0.5 * cid)
        n_flip = int(len(y) * ratio)
        idx = np.random.choice(len(y), n_flip, replace=False)
        y[idx] = np.random.randint(0, n_classes, size=n_flip)

    return X, y


# ---------- MODEL ----------
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


# ---------- EXPERIMENT ----------
def run_experiment(scenario_name):
    cfg = DRIFT_CFG[scenario_name]

    print("\n" + "="*50)
    print(f"  Scenario: {scenario_name}")
    print("="*50)

    accs, f1s = [], []

    for fold, (tr, te) in enumerate(skf.split(X_sc, y_enc), 1):

        Xtr, Xte = X_sc[tr], X_sc[te]
        ytr, yte = y_enc[tr], y_enc[te]

        splits = np.array_split(range(len(Xtr)), 3)

        models, trust_scores = [], []

        for cid, idx in enumerate(splits):

            X_loc = Xtr[idx]
            y_loc = ytr[idx]

            # ensure all classes exist
            for c in range(n_classes):
                if c not in y_loc:
                    X_loc = np.vstack([X_loc, np.zeros(X_loc.shape[1])])
                    y_loc = np.append(y_loc, c)

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_loc, y_loc,
                test_size=0.2,
                stratify=y_loc,
                random_state=fold
            )

            # apply drift only to clients 1 & 2
            if cid > 0:
                X_tr, y_tr = degrade(X_tr, y_tr, cid, cfg)

            model = build_model(cid, cfg)
            model.fit(X_tr, y_tr)

            trust = accuracy_score(
                y_val,
                np.argmax(model.predict_proba(X_val), axis=1)
            )

            print(f"  Fold {fold} | Client {cid} | Trust: {trust:.4f}")

            models.append(model)
            trust_scores.append(trust)

        # ---------- AGGREGATION ----------
        w = np.array(trust_scores)
        w = w / (w.sum() + 1e-9)

        probs = [m.predict_proba(Xte) for m in models]
        final = sum(wi * p for wi, p in zip(w, probs))

        pred = np.argmax(final, axis=1)

        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="weighted")

        print(f"  >>> Fold {fold}: ACC={acc:.4f} | F1={f1:.4f}")

        accs.append(acc)
        f1s.append(f1)

    print(f"\n  MEAN ACC = {np.mean(accs):.4f}  |  MEAN F1 = {np.mean(f1s):.4f}")

    RESULTS[scenario_name] = (np.mean(accs), np.mean(f1s))


# ---------- RUN ----------
run_experiment("NO DRIFT")
run_experiment("MILD DRIFT")
run_experiment("SEVERE DRIFT")


# ---------- SUMMARY ----------
print("\n" + "="*50)
print("  ABLATION SUMMARY")
print("="*50)

for k in RESULTS:
    print(f"  {k:<14}  ACC={RESULTS[k][0]:.4f}  F1={RESULTS[k][1]:.4f}")