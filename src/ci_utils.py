# ci_utils.py

import numpy as np
from scipy.stats import t
import pandas as pd


# =========================
# CORE FUNCTION
# =========================
def compute_ci(data, confidence=0.95):
    """
    Compute mean and confidence interval (CI).

    Parameters:
        data (list or array): input values
        confidence (float): confidence level (default 0.95)

    Returns:
        mean (float)
        margin (float): CI half-width
    """
    data = np.array(data)
    n = len(data)

    mean = np.mean(data)
    std = np.std(data, ddof=1)  # sample std

    t_val = t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_val * (std / np.sqrt(n))

    return mean, margin


# =========================
# PRINT FUNCTION
# =========================
def print_ci_results(name, trust, fed):
    """
    Print formatted CI results.
    """
    t_mean, t_ci = compute_ci(trust)
    f_mean, f_ci = compute_ci(fed)

    print("\n" + "=" * 60)
    print(f"{name}")
    print("=" * 60)
    print(f"Trust  : {t_mean:.4f} ± {t_ci:.4f} (95% CI)")
    print(f"FedAvg : {f_mean:.4f} ± {f_ci:.4f} (95% CI)")
    print("=" * 60)


# =========================
# RETURN DICTIONARY
# =========================
def get_ci_dict(trust, fed):
    """
    Return CI results as dictionary.
    """
    t_mean, t_ci = compute_ci(trust)
    f_mean, f_ci = compute_ci(fed)

    return {
        "trust_mean": round(t_mean, 4),
        "trust_ci": round(t_ci, 4),
        "fed_mean": round(f_mean, 4),
        "fed_ci": round(f_ci, 4),
    }


# =========================
# SAVE TO CSV (OPTIONAL)
# =========================
def save_ci_to_csv(results, filename="ci_results.csv"):
    """
    Save CI results to CSV.

    Parameters:
        results (dict): RESULTS dictionary from pipeline
        filename (str): output file
    """
    rows = []

    for scenario in results:
        trust = results[scenario]["trust"]
        fed = results[scenario]["fed"]

        t_mean, t_ci = compute_ci(trust)
        f_mean, f_ci = compute_ci(fed)

        rows.append({
            "Scenario": scenario,
            "Trust Mean": round(t_mean, 4),
            "Trust CI": round(t_ci, 4),
            "FedAvg Mean": round(f_mean, 4),
            "FedAvg CI": round(f_ci, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)

    print(f"\nSaved CI results to {filename}")