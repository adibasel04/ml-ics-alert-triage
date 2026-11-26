"""
baselines.py
Two simple approaches to compare the Transformer against.

1) Rule-based baseline: a fixed lookup by event type, like a basic SIEM rule.
   It does not look at process values, login counts or whether the user is known,
   so it misses context the other models can learn.

2) Random Forest (scikit-learn): learns from hand-built features of each alert.

Usage:  python src/baselines.py      (trains the Random Forest and saves it)
"""

import os
import pickle

from sklearn.ensemble import RandomForestClassifier

import config as C
from log_parser import load_split

RF_PATH = os.path.join(C.MODEL_DIR, "random_forest.pkl")

# 1) Rule-based baseline

RULES = {
    "read_registers": ("low", ["monitor"]),
    "heartbeat": ("low", ["monitor"]),
    "comm_timeout": ("medium", ["check_network"]),
    "failed_login": ("medium", ["verify_user"]),
    "config_download": ("medium", ["review_change"]),
    "setpoint_change": ("medium", ["review_change"]),
    "unauthorized_write": ("high", ["verify_write", "block_source"]),
    "port_scan": ("high", ["block_source"]),
    "firmware_update": ("high", ["review_change"]),
    "safety_override": ("critical", ["notify_engineer", "isolate_device"]),
}


def rule_predict(fields):
    """Look up severity by event type only. Unknown events default to medium."""
    return RULES.get(fields["event"], ("medium", ["verify_user"]))



# 2) Random Forest
EVENTS = list(RULES)


def features(fields):
    """Turn parsed fields into a numeric feature vector."""
    v = [1.0 if fields["event"] == e else 0.0 for e in EVENTS]          # one-hot event
    v.append(1.0 if fields["user"] in C.KNOWN_USERS else 0.0)            # known user?
    v.append(1.0 if fields["src"] in C.KNOWN_SOURCES else 0.0)           # known source?
    v.append(1.0 if fields["protocol"] == "dnp3" else 0.0)
    try:
        count = float(fields["count"])
    except ValueError:
        count = 0.0
    v.append(count)
    # how far the value is outside the device's safe range (0 if inside or missing)
    out_of_range = 0.0
    if fields["value"] not in ("-", "") and fields["device"] in C.DEVICES:
        try:
            val = float(fields["value"])
            low, high = C.DEVICES[fields["device"]]["safe_range"]
            out_of_range = max(low - val, val - high, 0.0)
        except ValueError:
            pass
    v.append(out_of_range)
    v.append(1.0 if fields["value"] not in ("-", "") else 0.0)           # has a value?
    return v


def train_rf(train_rows, val_rows):
    """Try a few forest sizes and keep the one with the best validation accuracy."""
    X = [features(r["fields"]) for r in train_rows]
    y = [r["severity"] for r in train_rows]
    Xv = [features(r["fields"]) for r in val_rows]
    yv = [r["severity"] for r in val_rows]
    best, best_acc = None, -1
    for n in (50, 100, 200):
        for depth in (6, 10, None):
            rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=C.SEED)
            rf.fit(X, y)
            acc = rf.score(Xv, yv)
            if acc > best_acc:
                best, best_acc = rf, acc
    print(f"Random Forest best validation accuracy: {best_acc:.3f} "
          f"(trees={best.n_estimators}, max_depth={best.max_depth})")
    return best


def rf_predict(rf, fields):
    return rf.predict([features(fields)])[0]


def load_rf():
    with open(RF_PATH, "rb") as fh:
        return pickle.load(fh)


if __name__ == "__main__":
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    train = load_split(os.path.join(C.DATA_DIR, "train.csv"))
    val = load_split(os.path.join(C.DATA_DIR, "val.csv"))
    rf = train_rf(train, val)
    with open(RF_PATH, "wb") as fh:
        pickle.dump(rf, fh)
    rule_acc = sum(rule_predict(r["fields"])[0] == r["severity"] for r in val) / len(val)
    print(f"Rule-based validation accuracy: {rule_acc:.3f}")
    print(f"Saved Random Forest to {RF_PATH}")
